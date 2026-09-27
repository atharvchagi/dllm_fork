"""Run the multi-sample Qwen3-0.6B speculative top-1 analysis.

Run on physical GPUs 0-3 after activating the ``dllm`` environment:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/analyze_speculative_top1_agreement_multi.py \
  --sample-jsonl /absolute/path/to/samples.jsonl \
  --num-samples 128 --max-blocks-per-sample 16 --block-size 4 \
  --output /absolute/path/to/results.json
```
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import transformers

import dllm

import analyze_speculative_top1_agreement as single


DEFAULT_SAMPLE_JSONL = Path(
    "/nvme-data2/atharvchagi/dllm_fork/results/loophole-block4-gsm8k/"
    "bd3lm-b4-loophole-gsm8k-threshold085-len1024/"
    "samples_gsm8k_cot_2026-09-14T18-42-53.106915.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/atharvchagi/qwen3_0.6b_bd3lm_speculative_agreement/"
    "multi-128samples-16blocks.json"
)


def parse_args() -> argparse.Namespace:
    """Parse multi-sample experiment settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and recurrent Loophole second-pass top-1 "
            "proposals against an AR verifier over multiple samples."
        )
    )
    parser.add_argument("--verifier-checkpoint", default=single.DEFAULT_VERIFIER)
    parser.add_argument("--baseline-checkpoint", default=single.DEFAULT_BASELINE)
    parser.add_argument("--loophole-checkpoint", default=single.DEFAULT_LOOPHOLE)
    parser.add_argument("--sample-jsonl", type=Path, default=DEFAULT_SAMPLE_JSONL)
    parser.add_argument(
        "--sample-origin",
        choices=("auto", "loophole_generated", "ar_verifier_generated", "unspecified"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--allow-fewer-samples", action="store_true")
    parser.add_argument("--max-blocks-per-sample", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate settings that define the requested protocol."""
    if args.block_size != 4:
        raise ValueError("This experiment requires --block-size 4")
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2")
    if args.max_blocks_per_sample < 1:
        raise ValueError("--max-blocks-per-sample must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    if not args.sample_jsonl.is_absolute() or not args.sample_jsonl.is_file():
        raise ValueError("--sample-jsonl must be an existing absolute path")
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Load one flexible-extract row per lm-eval document."""
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as sample_file:
        for line_number, line in enumerate(sample_file, start=1):
            row = json.loads(line)
            if row.get("filter") != "flexible-extract":
                continue
            doc_id = int(row["doc_id"])
            if doc_id in rows:
                raise ValueError(f"Duplicate flexible-extract row for doc_id={doc_id}")
            prompt = row.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
            response = row.get("resps", [[""]])[0][0]
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise TypeError(f"Non-string prompt or response for doc_id={doc_id}")
            tokenized = row.get("tokenized_sample")
            if tokenized is not None:
                for key in ("prompt_token_ids", "response_token_ids"):
                    token_ids = tokenized.get(key)
                    if not isinstance(token_ids, list) or any(
                        not isinstance(token_id, int) or token_id < 0
                        for token_id in token_ids
                    ):
                        raise ValueError(f"Invalid saved {key} for doc_id={doc_id}")
            rows[doc_id] = {
                "doc_id": doc_id,
                "line_number": line_number,
                "prompt": prompt,
                "response": response,
                "tokenized_sample": tokenized,
                "generation": row.get("generation", {}),
            }
    if not rows:
        raise ValueError(f"No flexible-extract rows found in {path}")
    return rows


def response_token_start(tokenizer, prompt: str, response: str) -> tuple[list[int], int]:
    """Tokenize joined text and locate the first token wholly in the response."""
    text = prompt + response
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    boundary = len(prompt)
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start >= boundary and end > start:
            return input_ids, index
    raise ValueError("Response has no token wholly beyond the prompt boundary")


def prepare_samples(
    rows: dict[int, dict[str, Any]],
    tokenizer,
    *,
    num_samples: int,
    max_blocks_per_sample: int,
    block_size: int,
    seed: int,
    allow_fewer_samples: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a deterministic random set of eligible response-only samples."""
    candidate_ids = sorted(rows)
    random.Random(seed).shuffle(candidate_ids)
    selected = []
    skipped = []
    for doc_id in candidate_ids:
        row = rows[doc_id]
        try:
            if row.get("tokenized_sample") is not None:
                saved = row["tokenized_sample"]
                prompt_ids = saved["prompt_token_ids"]
                if tokenizer.encode(row["prompt"], add_special_tokens=False) != prompt_ids:
                    raise ValueError("Saved prompt IDs disagree with the shared tokenizer")
                input_ids = prompt_ids + saved["response_token_ids"]
                response_start = len(prompt_ids)
                token_source = "exact_saved_generation_ids"
            else:
                input_ids, response_start = response_token_start(
                    tokenizer, row["prompt"], row["response"]
                )
                token_source = "retokenized_text"
            block_starts, structural_skips = single.choose_block_starts(
                input_ids,
                tokenizer,
                block_size=block_size,
                minimum_prefix_tokens=response_start,
                max_blocks=max_blocks_per_sample,
            )
        except ValueError as error:
            skipped.append({"doc_id": doc_id, "reason": str(error)})
            continue
        text = row["prompt"] + row["response"]
        selected.append(
            {
                "doc_id": doc_id,
                "line_number": row["line_number"],
                "input_ids": input_ids,
                "response_token_start": response_start,
                "token_count": len(input_ids),
                "prompt_character_count": len(row["prompt"]),
                "response_character_count": len(row["response"]),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_ids_sha256": single.stable_json_hash(input_ids),
                "token_source": token_source,
                "generation": row.get("generation", {}),
                "block_starts": block_starts,
                "structural_blocks_skipped": structural_skips,
            }
        )
        if len(selected) == num_samples:
            break
    if len(selected) != num_samples and not (allow_fewer_samples and len(selected) >= 2):
        raise ValueError(
            f"Requested {num_samples} samples but found only {len(selected)} eligible rows"
        )
    return selected, skipped


def paired_counts(results: list[dict[str, Any]], context: str) -> dict[str, int]:
    """Count paired baseline/Loophole outcomes on eligible positions."""
    rows = [
        row
        for result in results
        if not result["excluded"]
        for row in result["rows"]
        if row["headline_eligible"]
    ]
    return {
        "both_match": sum(
            bool(row[context]["baseline"] and row[context]["loophole"])
            for row in rows
        ),
        "loop_only_match": sum(
            bool(not row[context]["baseline"] and row[context]["loophole"])
            for row in rows
        ),
        "base_only_match": sum(
            bool(row[context]["baseline"] and not row[context]["loophole"])
            for row in rows
        ),
        "neither_match": sum(
            bool(not row[context]["baseline"] and not row[context]["loophole"])
            for row in rows
        ),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results and add paired counts for the primary context."""
    summary = single.aggregate(results)
    summary["candidate_conditioned"].update(
        paired_counts(results, "candidate_conditioned_matches")
    )
    return summary


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove block-level traces from a per-sample summary."""
    return {
        "evaluated_blocks": summary["evaluated_blocks"],
        "excluded_blocks": summary["excluded_blocks"],
        "candidate_conditioned": summary["candidate_conditioned"],
        "teacher_forced_common_context": summary["teacher_forced_common_context"],
        "accepted_prefix_length": {
            model: {
                "total": values["total"],
                "mean": values["mean"],
            }
            for model, values in summary["accepted_prefix_length"].items()
        },
    }


def cluster_bootstrap(
    per_sample: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap pooled match-rate deltas by resampling whole samples."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(per_sample),
        size=(replicates, len(per_sample)),
        dtype=np.int32,
    )
    output = {
        "unit": "sample",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
    }
    for output_key, summary_key in (
        ("candidate_conditioned", "candidate_conditioned"),
        ("teacher_forced_common_context", "teacher_forced_common_context"),
    ):
        base = np.asarray(
            [item[summary_key]["base_top1_matches"] for item in per_sample],
            dtype=np.int64,
        )
        loop = np.asarray(
            [item[summary_key]["loop_top1_matches"] for item in per_sample],
            dtype=np.int64,
        )
        denominator = np.asarray(
            [item[summary_key]["eligible_positions"] for item in per_sample],
            dtype=np.int64,
        )
        deltas = (
            (loop[indices] - base[indices]).sum(axis=1)
            / denominator[indices].sum(axis=1)
        )
        lower, upper = np.quantile(deltas, [0.025, 0.975])
        output[output_key] = {
            "match_rate_delta_ci_lower": float(lower),
            "match_rate_delta_ci_upper": float(upper),
        }
    return output


def sample_win_counts(
    per_sample: list[dict[str, Any]], summary_key: str
) -> dict[str, int]:
    """Count samples where each model has more top-1 matches."""
    deltas = [
        item[summary_key]["loop_top1_matches"]
        - item[summary_key]["base_top1_matches"]
        for item in per_sample
    ]
    return {
        "loophole_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "baseline_wins": sum(delta < 0 for delta in deltas),
    }


def render_markdown(report: dict[str, Any], json_path: Path) -> str:
    """Render the multi-sample headline results and uncertainty."""
    results = report["results"]
    candidate = results["candidate_conditioned"]
    teacher = results["teacher_forced_common_context"]
    accepted = results["accepted_prefix_length"]
    bootstrap = report["uncertainty"]["sample_cluster_bootstrap"]
    candidate_ci = bootstrap["candidate_conditioned"]
    teacher_ci = bootstrap["teacher_forced_common_context"]
    candidate_wins = report["per_sample_comparison"]["candidate_conditioned"]
    teacher_wins = report["per_sample_comparison"]["teacher_forced_common_context"]
    origin = report["samples"].get("origin", "unspecified")
    origin_description = {
        "loophole_generated": "the Loophole GSM8K generation run",
        "ar_verifier_generated": "greedy generation by the same AR verifier",
        "unspecified": "the saved response JSONL",
    }[origin]
    lines = [
        "# Multi-sample Qwen3-0.6B speculative top-1 agreement",
        "",
        f"Full trace: `{json_path.resolve()}`",
        "",
        (
            f"Randomly selected {report['samples']['selected_count']} independent "
            f"GSM8K responses and evaluated {results['evaluated_blocks']} block-size-4 "
            f"targets ({candidate['eligible_positions']} second-pass positions)."
        ),
        "",
        "## Candidate-conditioned AR verification (primary)",
        "",
        "| Model | Top-1 matches | Eligible positions | Match rate |",
        "|---|---:|---:|---:|",
        (
            f"| Baseline BD3LM | {candidate['base_top1_matches']} | "
            f"{candidate['eligible_positions']} | {candidate['base_top1_match_rate']:.2%} |"
        ),
        (
            f"| Loophole with `h_1` | {candidate['loop_top1_matches']} | "
            f"{candidate['eligible_positions']} | {candidate['loop_top1_match_rate']:.2%} |"
        ),
        "",
        (
            f"Loophole minus baseline: **{candidate['match_count_delta']:+d} matches "
            f"({100 * candidate['match_rate_delta']:+.2f} percentage points)**; "
            f"sample-cluster bootstrap 95% CI "
            f"[{100 * candidate_ci['match_rate_delta_ci_lower']:+.2f}, "
            f"{100 * candidate_ci['match_rate_delta_ci_upper']:+.2f}] percentage points."
        ),
        (
            f"Paired positions: both={candidate['both_match']}, "
            f"Loophole-only={candidate['loop_only_match']}, "
            f"baseline-only={candidate['base_only_match']}, "
            f"neither={candidate['neither_match']}."
        ),
        (
            f"Per-sample outcomes: Loophole wins={candidate_wins['loophole_wins']}, "
            f"ties={candidate_wins['ties']}, baseline wins={candidate_wins['baseline_wins']}."
        ),
        "",
        "## Teacher-forced common-context diagnostic",
        "",
        "| Model | Top-1 matches | Eligible positions | Match rate |",
        "|---|---:|---:|---:|",
        (
            f"| Baseline BD3LM | {teacher['base_top1_matches']} | "
            f"{teacher['eligible_positions']} | {teacher['base_top1_match_rate']:.2%} |"
        ),
        (
            f"| Loophole with `h_1` | {teacher['loop_top1_matches']} | "
            f"{teacher['eligible_positions']} | {teacher['loop_top1_match_rate']:.2%} |"
        ),
        "",
        (
            f"Loophole minus baseline: **{teacher['match_count_delta']:+d} matches "
            f"({100 * teacher['match_rate_delta']:+.2f} percentage points)**; "
            f"sample-cluster bootstrap 95% CI "
            f"[{100 * teacher_ci['match_rate_delta_ci_lower']:+.2f}, "
            f"{100 * teacher_ci['match_rate_delta_ci_upper']:+.2f}] percentage points."
        ),
        (
            f"Paired positions: both={teacher['both_match']}, "
            f"Loophole-only={teacher['loop_only_match']}, "
            f"baseline-only={teacher['base_only_match']}, "
            f"neither={teacher['neither_match']}."
        ),
        (
            f"Per-sample outcomes: Loophole wins={teacher_wins['loophole_wins']}, "
            f"ties={teacher_wins['ties']}, baseline wins={teacher_wins['baseline_wins']}."
        ),
        "",
        "## Strict greedy accepted prefix",
        "",
        (
            f"Baseline mean: {accepted['baseline']['mean']:.3f}/4; "
            f"Loophole mean: {accepted['loophole']['mean']:.3f}/4."
        ),
        "",
        "## Scope",
        "",
        (
            f"Each response contributes at most "
            f"{report['protocol']['max_blocks_per_sample']} aligned response-only blocks. "
            "The confidence interval resamples complete responses, preserving within-response "
            f"block dependence. The clean responses came from {origin_description}, "
            "so this measures verifier agreement on that fixed response distribution."
        ),
        (
            "Both second-pass models receive the identical half-filled block: two Loophole "
            "first-pass top-1 tokens and two masks, with the same clean causal prefix. "
            "The second Loophole pass additionally receives the aligned first-pass state."
        ),
        "",
    ]
    if origin == "ar_verifier_generated":
        consistency = report["ar_generation_consistency"]
        lines.extend(
            [
                "## AR generation consistency",
                "",
                (
                    f"Recomputed clean-prefix AR top-1 agrees with the saved generated token "
                    f"at {consistency['all_target_top1_matches']}/"
                    f"{consistency['all_target_positions']} target positions and "
                    f"{consistency['eligible_top1_matches']}/"
                    f"{consistency['eligible_positions']} headline positions. "
                    "Generation uses a KV cache; verification recomputes unpadded full "
                    "prefixes, so small BF16/kernel argmax differences are possible."
                ),
                "",
            ]
        )
        saved_matches = consistency.get("saved_generated_token_matches")
        if saved_matches is not None:
            lines.extend(
                [
                    (
                        "Direct second-pass matches to the exact saved greedy AR tokens: "
                        f"baseline={saved_matches['baseline']}, "
                        f"Loophole={saved_matches['loophole']} out of "
                        f"{consistency['eligible_positions']} headline positions. "
                        "These are saved generation IDs, distinct from recomputed verifier top-1."
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    """Run sharded multi-sample evaluation and write one merged report."""
    args = parse_args()
    validate_args(args)
    rank, world_size, device = single.initialize_distributed(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, args.dtype)
    started_at = time.perf_counter()

    baseline_tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=args.baseline_checkpoint
    )
    loophole_tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=args.loophole_checkpoint
    )
    verifier_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.verifier_checkpoint
    )
    tokenizer_metadata = single.validate_tokenizers(
        baseline_tokenizer,
        loophole_tokenizer,
        verifier_tokenizer,
    )
    rows = load_rows(args.sample_jsonl)
    origins = {row["generation"].get("origin", "unspecified") for row in rows.values()}
    sample_origin = args.sample_origin
    if sample_origin == "auto":
        sample_origin = next(iter(origins)) if len(origins) == 1 else "unspecified"
    if sample_origin == "ar_verifier_generated":
        for row in rows.values():
            generation = row["generation"]
            if generation.get("checkpoint") != args.verifier_checkpoint:
                raise ValueError("AR response generator and verifier checkpoints differ")
            if generation.get("tokenizer_fingerprint") != tokenizer_metadata["fingerprints"]["verifier"]:
                raise ValueError("AR generation tokenizer fingerprint differs from verifier")
            if row["tokenized_sample"] is None:
                raise ValueError("AR-generated samples must preserve their exact token IDs")
    samples, selection_skips = prepare_samples(
        rows,
        baseline_tokenizer,
        num_samples=args.num_samples,
        max_blocks_per_sample=args.max_blocks_per_sample,
        block_size=args.block_size,
        seed=args.seed,
        allow_fewer_samples=args.allow_fewer_samples,
    )
    tasks = [
        (sample["doc_id"], block_start)
        for sample in samples
        for block_start in sample["block_starts"]
    ]
    local_tasks = tasks[rank::world_size]
    if rank == 0:
        print(
            f"world_size={world_size} samples={len(samples)} blocks={len(tasks)} "
            f"positions={2 * len(tasks)}",
            flush=True,
        )

    baseline_model, baseline_config = single.load_diffusion_model(
        args.baseline_checkpoint,
        expected_loophole_enabled=False,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    loophole_model, loophole_config = single.load_diffusion_model(
        args.loophole_checkpoint,
        expected_loophole_enabled=True,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    verifier_model, verifier_config = single.load_verifier_model(
        args.verifier_checkpoint,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    if not (
        baseline_config.vocab_size
        == loophole_config.vocab_size
        == verifier_config.vocab_size
    ):
        raise ValueError("Model vocabulary sizes differ")
    if any(
        token_id >= baseline_config.vocab_size
        for sample in samples
        for token_id in sample["input_ids"]
    ):
        raise ValueError("Saved sample includes a token ID outside the model vocabulary")
    if sample_origin == "ar_verifier_generated" and any(
        sample["generation"].get("revision") != getattr(verifier_config, "_commit_hash", None)
        for sample in samples
    ):
        raise ValueError("AR response generator and verifier revisions differ")

    sample_by_id = {sample["doc_id"]: sample for sample in samples}
    tensor_by_id = {
        doc_id: torch.tensor(
            sample_by_id[doc_id]["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        for doc_id in {doc_id for doc_id, _ in local_tasks}
    }
    local_results = []
    progress_interval = max(1, len(local_tasks) // 10)
    for local_index, (doc_id, block_start) in enumerate(local_tasks, start=1):
        result = single.evaluate_block(
            clean_ids=tensor_by_id[doc_id],
            block_start=block_start,
            block_size=args.block_size,
            baseline_model=baseline_model,
            loophole_model=loophole_model,
            verifier_model=verifier_model,
            tokenizer=baseline_tokenizer,
        )
        result["doc_id"] = doc_id
        for row in result.get("rows", []):
            row["doc_id"] = doc_id
        local_results.append(result)
        if local_index % progress_interval == 0 or local_index == len(local_tasks):
            print(
                f"rank={rank} completed={local_index}/{len(local_tasks)}",
                flush=True,
            )

    local_payload = {
        "rank": rank,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "task_count": len(local_tasks),
        "results": local_results,
        "runtime_seconds": time.perf_counter() - started_at,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    gathered: list[dict[str, Any] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    if world_size > 1:
        dist.gather_object(local_payload, gathered, dst=0)
    else:
        gathered = [local_payload]

    if rank == 0:
        assert gathered is not None
        worker_payloads = [payload for payload in gathered if payload is not None]
        all_results = sorted(
            [
                result
                for payload in worker_payloads
                for result in payload["results"]
            ],
            key=lambda result: (result["doc_id"], result["block_start"]),
        )
        global_summary = summarize_results(all_results)
        per_sample = []
        for sample in samples:
            doc_results = [
                result
                for result in all_results
                if result["doc_id"] == sample["doc_id"]
            ]
            compact = compact_summary(summarize_results(doc_results))
            compact["doc_id"] = sample["doc_id"]
            per_sample.append(compact)
        bootstrap = cluster_bootstrap(
            per_sample,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        )
        model_info = {
            "verifier": single.model_metadata(
                args.verifier_checkpoint, verifier_model, verifier_config
            ),
            "baseline": single.model_metadata(
                args.baseline_checkpoint, baseline_model, baseline_config
            ),
            "loophole": single.model_metadata(
                args.loophole_checkpoint, loophole_model, loophole_config
            ),
        }
        sample_metadata = [
            {key: value for key, value in sample.items() if key != "input_ids"}
            for sample in samples
        ]
        report = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "qwen3_0.6b_bd3lm_speculative_top1_agreement_multi_sample",
            "protocol": {
                "block_size": args.block_size,
                "max_blocks_per_sample": args.max_blocks_per_sample,
                "sample_selection": "uniform_without_replacement_after_seeded_shuffle",
                "response_only": True,
                "initial_target_mask_fraction": 1.0,
                "first_pass_fill_fraction": 0.5,
                "first_pass_proposer": "loophole",
                "second_pass_models": ["baseline", "loophole_with_h1"],
                "common_half_filled_input": True,
                "first_pass_hidden_state": "null",
                "right_shift_logits": True,
                "right_shift_loophole_state": True,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "candidate_conditioned_verification": True,
                "teacher_forced_common_context_diagnostic": True,
                "headline_positions": "R_only",
                "seed": args.seed,
            },
            "models": model_info,
            "tokenizer": tokenizer_metadata,
            "samples": {
                "source_path": str(args.sample_jsonl.resolve()),
                "source_filter": "flexible-extract",
                "origin": sample_origin,
                "available_count": len(rows),
                "requested_count": args.num_samples,
                "selected_count": len(samples),
                "selected": sample_metadata,
                "selection_skips": selection_skips,
            },
            "execution": {
                "launch": (
                    "torchrun_data_parallel" if world_size > 1 else "single_process"
                ),
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "workers": [
                    {key: value for key, value in payload.items() if key != "results"}
                    for payload in worker_payloads
                ],
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "dtype": args.dtype,
                "attention_implementation": args.attn_implementation,
                "slurm_used": False,
                "slurm_note": "srun is unavailable on this host",
            },
            "results": global_summary,
            "per_sample": per_sample,
            "per_sample_comparison": {
                "candidate_conditioned": sample_win_counts(
                    per_sample, "candidate_conditioned"
                ),
                "teacher_forced_common_context": sample_win_counts(
                    per_sample, "teacher_forced_common_context"
                ),
            },
            "uncertainty": {"sample_cluster_bootstrap": bootstrap},
        }
        generation_metadata_path = args.sample_jsonl.with_suffix(".metadata.json")
        if generation_metadata_path.is_file():
            with generation_metadata_path.open(encoding="utf-8") as handle:
                report["samples"]["generation_provenance"] = json.load(handle)
        if sample_origin == "ar_verifier_generated":
            all_rows = global_summary["per_position"]
            eligible_rows = [row for row in all_rows if row["headline_eligible"]]
            report["ar_generation_consistency"] = {
                "all_target_positions": len(all_rows),
                "all_target_top1_matches": sum(
                    row["clean"]["id"] == row["teacher_forced_verifier"]["id"]
                    for row in all_rows
                ),
                "eligible_positions": len(eligible_rows),
                "eligible_top1_matches": sum(
                    row["clean"]["id"] == row["teacher_forced_verifier"]["id"]
                    for row in eligible_rows
                ),
                "saved_generated_token_matches": {
                    model: sum(
                        row[f"{model}_second_pass"]["id"] == row["clean"]["id"]
                        for row in eligible_rows
                    )
                    for model in ("baseline", "loophole")
                },
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        markdown_path = args.output.with_suffix(".md")
        markdown_path.write_text(
            render_markdown(report, args.output), encoding="utf-8"
        )
        print(f"Wrote {args.output.resolve()}", flush=True)
        print(f"Wrote {markdown_path.resolve()}", flush=True)
        print(
            json.dumps(global_summary["candidate_conditioned"], indent=2),
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
