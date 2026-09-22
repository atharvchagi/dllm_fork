"""Run the Qwen3-0.6B BD3LM/Loophole top-1 verifier experiment.

Run on physical GPUs 0-3 after activating the ``dllm`` environment:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/analyze_speculative_top1_agreement.py \
  --block-size 4 \
  --output /nvme-data2/atharvchagi/dllm_fork/results/qwen3_0.6b_bd3lm_speculative_agreement/results.json
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
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

import dllm
from dllm.core.samplers.bd3lm import _prepare_for_sampling


DEFAULT_VERIFIER = "Qwen/Qwen3-0.6B-Base"
DEFAULT_BASELINE = "divelab/Qwen3-0.6B-bd3lm-block4-len4096-baseline"
DEFAULT_LOOPHOLE = "divelab/Qwen3-0.6B-bd3lm-block4-len4096-loophole"
DEFAULT_OUTPUT = Path(
    "/nvme-data2/atharvchagi/dllm_fork/results/"
    "qwen3_0.6b_bd3lm_speculative_agreement/results.json"
)
DEFAULT_SAMPLE_TEXT = """Question: Janet's ducks lay 16 eggs per day. She eats
three eggs for breakfast and uses four eggs to bake muffins. She sells every
remaining egg for two dollars. How much money does Janet earn each day?

Solution: Janet begins with 16 eggs. After breakfast she has 16 - 3 = 13 eggs.
After baking she has 13 - 4 = 9 eggs. Selling 9 eggs for 2 dollars each gives
9 times 2 = 18 dollars. Therefore Janet earns 18 dollars each day."""


def parse_args() -> argparse.Namespace:
    """Parse experiment settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare second-pass baseline and Loophole BD3LM top-1 proposals "
            "against a Qwen3 autoregressive verifier."
        )
    )
    parser.add_argument("--verifier-checkpoint", default=DEFAULT_VERIFIER)
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE)
    parser.add_argument("--loophole-checkpoint", default=DEFAULT_LOOPHOLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-jsonl",
        type=Path,
        help=(
            "Optional lm-eval samples JSONL. The prompt and response from the "
            "requested flexible-extract row form the clean sample."
        ),
    )
    parser.add_argument("--doc-id", type=int, default=0)
    parser.add_argument("--sample-text", default=None)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--minimum-prefix-tokens", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=64)
    parser.add_argument("--max-sample-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject settings that do not implement the requested experiment."""
    if args.block_size != 4:
        raise ValueError("This run requires --block-size 4")
    if args.minimum_prefix_tokens < 1:
        raise ValueError("minimum-prefix-tokens must be positive")
    if args.max_blocks < 1:
        raise ValueError("max-blocks must be positive")
    if args.max_sample_tokens <= args.minimum_prefix_tokens + args.block_size:
        raise ValueError("max-sample-tokens is too small for one target block")
    if args.sample_jsonl is not None and args.sample_text is not None:
        raise ValueError("Pass only one of --sample-jsonl and --sample-text")
    if args.sample_jsonl is not None and not args.sample_jsonl.is_file():
        raise FileNotFoundError(f"Sample JSONL does not exist: {args.sample_jsonl}")
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def initialize_distributed(device_name: str) -> tuple[int, int, torch.device]:
    """Initialize torchrun workers and return rank, world size, and local device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device_name == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    if device_name == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
    return rank, world_size, device


def load_lm_eval_sample(path: Path, doc_id: int) -> tuple[str, dict[str, Any]]:
    """Load one prompt-plus-response sample from an lm-eval JSONL file."""
    with path.open(encoding="utf-8") as sample_file:
        for line_number, line in enumerate(sample_file, start=1):
            row = json.loads(line)
            if row.get("filter") != "flexible-extract":
                continue
            if int(row.get("doc_id", -1)) != doc_id:
                continue
            prompt = row.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
            response = row.get("resps", [[""]])[0][0]
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise TypeError("lm-eval prompt and response must be strings")
            text = prompt + response
            if not text.strip():
                raise ValueError("Selected lm-eval sample is empty")
            return text, {
                "kind": "lm_eval_prompt_plus_response",
                "path": str(path.resolve()),
                "line_number": line_number,
                "doc_id": doc_id,
                "filter": "flexible-extract",
            }
    raise ValueError(
        f"No flexible-extract row for doc_id={doc_id} was found in {path}"
    )


def load_sample(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Resolve the requested clean sample and its provenance."""
    if args.sample_jsonl is not None:
        return load_lm_eval_sample(args.sample_jsonl, args.doc_id)
    if args.sample_text is not None:
        if not args.sample_text.strip():
            raise ValueError("--sample-text cannot be empty")
        return args.sample_text, {"kind": "command_line_text"}
    return DEFAULT_SAMPLE_TEXT, {"kind": "built_in_text"}


def stable_json_hash(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic serialization."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash the complete token-to-ID mapping."""
    return stable_json_hash(sorted(tokenizer.get_vocab().items()))


def validate_tokenizers(
    baseline_tokenizer,
    loophole_tokenizer,
    verifier_tokenizer,
) -> dict[str, Any]:
    """Require shared token IDs and permit BD3LM's diffusion-only mask alias."""
    baseline_vocab = baseline_tokenizer.get_vocab()
    loophole_vocab = loophole_tokenizer.get_vocab()
    verifier_vocab = verifier_tokenizer.get_vocab()
    if baseline_vocab != loophole_vocab:
        raise ValueError("Baseline and Loophole tokenizer vocabularies differ")
    baseline_only = sorted(set(baseline_vocab) - set(verifier_vocab))
    verifier_only = sorted(set(verifier_vocab) - set(baseline_vocab))
    mismatched_ids = sorted(
        token
        for token in set(baseline_vocab) & set(verifier_vocab)
        if baseline_vocab[token] != verifier_vocab[token]
    )
    permitted_baseline_only = (
        [baseline_tokenizer.mask_token]
        if baseline_tokenizer.mask_token is not None
        else []
    )
    mask_alias_is_valid = (
        baseline_only == permitted_baseline_only
        and baseline_tokenizer.mask_token_id is not None
        and baseline_vocab[baseline_tokenizer.mask_token]
        == baseline_tokenizer.mask_token_id
    )
    if verifier_only or mismatched_ids or (baseline_only and not mask_alias_is_valid):
        raise ValueError(
            "Verifier vocabulary is not exactly aligned with BD3LM: "
            f"baseline_only={baseline_only[:10]}, "
            f"verifier_only={verifier_only[:10]}, "
            f"mismatched_ids={mismatched_ids[:10]}"
        )
    fingerprints = {
        "baseline": tokenizer_fingerprint(baseline_tokenizer),
        "loophole": tokenizer_fingerprint(loophole_tokenizer),
        "verifier": tokenizer_fingerprint(verifier_tokenizer),
    }
    return {
        "baseline_tokenizer_entries": len(baseline_vocab),
        "verifier_tokenizer_entries": len(verifier_vocab),
        "verifier_tokenizer_base_vocab_size": verifier_tokenizer.vocab_size,
        "fingerprints": fingerprints,
        "shared_token_ids_match": not verifier_only and not mismatched_ids,
        "exact_mapping_match": baseline_vocab == verifier_vocab,
        "permitted_baseline_only_aliases": baseline_only,
        "mask_token": baseline_tokenizer.mask_token,
        "mask_token_id": baseline_tokenizer.mask_token_id,
    }


def load_diffusion_model(
    checkpoint: str,
    *,
    expected_loophole_enabled: bool,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
):
    """Load one A2D checkpoint and validate its Loophole capability flag."""
    config = transformers.AutoConfig.from_pretrained(checkpoint)
    actual = bool(getattr(config, "loophole_enabled", False))
    if actual != expected_loophole_enabled:
        raise ValueError(
            f"{checkpoint} has loophole_enabled={actual}; expected "
            f"{expected_loophole_enabled}"
        )
    model = dllm.utils.get_model(
        model_name_or_path=checkpoint,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.eval()
    if model.device != device:
        model.to(device)
    return model, config


def load_verifier_model(
    checkpoint: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
):
    """Load the causal autoregressive verifier."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.eval().to(device)
    return model, model.config


def model_metadata(checkpoint: str, model, config) -> dict[str, Any]:
    """Build serializable checkpoint metadata."""
    return {
        "checkpoint": checkpoint,
        "revision": getattr(config, "_commit_hash", None),
        "model_type": config.model_type,
        "model_class": type(model).__name__,
        "vocab_size": int(config.vocab_size),
        "loophole_enabled": bool(getattr(config, "loophole_enabled", False)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def choose_block_starts(
    input_ids: list[int],
    tokenizer,
    *,
    block_size: int,
    minimum_prefix_tokens: int,
    max_blocks: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Choose aligned target blocks that contain no structural special tokens."""
    first_start = (
        (minimum_prefix_tokens + block_size - 1) // block_size
    ) * block_size
    structural_ids = {
        token_id
        for token_id in (
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.mask_token_id,
        )
        if token_id is not None
    }
    starts = []
    skipped = []
    for start in range(first_start, len(input_ids) - block_size + 1, block_size):
        target = input_ids[start : start + block_size]
        found = sorted(structural_ids.intersection(target))
        if found:
            skipped.append({"block_start": start, "structural_token_ids": found})
            continue
        starts.append(start)
        if len(starts) == max_blocks:
            break
    if not starts:
        raise ValueError("The sample contains no eligible aligned target blocks")
    return starts, skipped


def make_bd3lm_inputs(
    input_ids: torch.Tensor,
    *,
    block_size: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor | bool]:
    """Construct full-sequence generation-time BD3LM attention inputs."""
    attention_mask, position_ids = _prepare_for_sampling(
        x=input_ids,
        block_size=block_size,
        pad_token_id=pad_token_id,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
    }


def right_shift_target(
    tensor: torch.Tensor,
    *,
    block_start: int,
    block_size: int,
) -> torch.Tensor:
    """Select preceding-position values aligned to one target block."""
    if block_start < 1:
        raise ValueError("A nonempty prefix is required for right shifting")
    shifted = tensor[:, block_start - 1 : block_start + block_size - 1]
    if shifted.shape[1] != block_size:
        raise ValueError(
            f"Expected {block_size} shifted positions, got {shifted.shape[1]}"
        )
    return shifted


def align_loophole_state(
    raw_state: torch.Tensor,
    *,
    block_start: int,
    block_size: int,
) -> torch.Tensor:
    """Create a null-prefix state with the active block globally right-shifted."""
    state_input = torch.zeros_like(raw_state)
    state_input[:, block_start : block_start + block_size] = right_shift_target(
        raw_state,
        block_start=block_start,
        block_size=block_size,
    )
    return state_input


def top1(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32-softmax top-1 token IDs and probabilities."""
    probabilities = F.softmax(logits.float(), dim=-1)
    confidence, token_ids = probabilities.max(dim=-1)
    return token_ids, confidence


def token_record(tokenizer, token_id: int) -> dict[str, Any]:
    """Record an authoritative token ID plus two readable representations."""
    return {
        "id": token_id,
        "token": tokenizer.convert_ids_to_tokens(token_id),
        "text": tokenizer.decode([token_id], skip_special_tokens=False),
    }


@torch.inference_mode()
def verifier_top1(
    verifier,
    prefix: torch.Tensor,
    candidate_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a complete candidate block with causal, preceding-token alignment."""
    full_ids = torch.cat([prefix, candidate_block], dim=1)
    outputs = verifier(input_ids=full_ids, use_cache=False)
    logits = right_shift_target(
        outputs.logits,
        block_start=prefix.shape[1],
        block_size=candidate_block.shape[1],
    )
    return top1(logits)


def accepted_prefix_length(candidate: torch.Tensor, verifier_ids: torch.Tensor) -> int:
    """Count consecutive left-to-right greedy matches before first rejection."""
    accepted = 0
    for candidate_id, verifier_id in zip(
        candidate[0].tolist(), verifier_ids[0].tolist()
    ):
        if candidate_id != verifier_id:
            break
        accepted += 1
    return accepted


@torch.inference_mode()
def evaluate_block(
    *,
    clean_ids: torch.Tensor,
    block_start: int,
    block_size: int,
    baseline_model,
    loophole_model,
    verifier_model,
    tokenizer,
) -> dict[str, Any]:
    """Evaluate one all-mask to half-filled to verified block trajectory."""
    prefix = clean_ids[:, :block_start]
    clean_block = clean_ids[:, block_start : block_start + block_size]
    mask_id = int(tokenizer.mask_token_id)
    noisy_block = torch.full_like(clean_block, mask_id)
    noisy_ids = torch.cat([prefix, noisy_block], dim=1)
    noisy_inputs = make_bd3lm_inputs(
        noisy_ids,
        block_size=block_size,
        pad_token_id=int(tokenizer.pad_token_id),
    )

    first_outputs = loophole_model(
        **noisy_inputs,
        loophole_state=None,
        return_loophole_state=True,
        loophole_enabled=True,
    )
    raw_state = getattr(first_outputs, "loophole_state", None)
    if raw_state is None:
        raise RuntimeError("First Loophole pass returned no recurrent state")
    first_logits = right_shift_target(
        first_outputs.logits,
        block_start=block_start,
        block_size=block_size,
    )
    first_ids, first_confidence = top1(first_logits)
    fill_count = (block_size + 1) // 2
    fill_indices = sorted(
        range(block_size),
        key=lambda index: (-float(first_confidence[0, index]), index),
    )[:fill_count]
    fill_indices = sorted(fill_indices)
    remaining_indices = [
        index for index in range(block_size) if index not in fill_indices
    ]

    structural_fill_ids = {
        int(token_id)
        for token_id in (tokenizer.mask_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    invalid_fills = [
        index
        for index in fill_indices
        if int(first_ids[0, index]) in structural_fill_ids
    ]
    if invalid_fills:
        return {
            "excluded": True,
            "reason": "first_pass_structural_fill",
            "block_start": block_start,
            "invalid_fill_indices": invalid_fills,
            "first_pass_ids": first_ids[0].cpu().tolist(),
            "first_pass_confidence": first_confidence[0].cpu().tolist(),
        }

    partial_block = noisy_block.clone()
    partial_block[:, fill_indices] = first_ids[:, fill_indices]
    partial_ids = torch.cat([prefix, partial_block], dim=1)
    partial_inputs = make_bd3lm_inputs(
        partial_ids,
        block_size=block_size,
        pad_token_id=int(tokenizer.pad_token_id),
    )

    baseline_outputs = baseline_model(**partial_inputs)
    baseline_logits = right_shift_target(
        baseline_outputs.logits,
        block_start=block_start,
        block_size=block_size,
    )
    baseline_ids, baseline_confidence = top1(baseline_logits)

    state_input = align_loophole_state(
        raw_state.detach(),
        block_start=block_start,
        block_size=block_size,
    )
    loophole_outputs = loophole_model(
        **partial_inputs,
        loophole_state=state_input,
        return_loophole_state=False,
        loophole_enabled=True,
    )
    loophole_logits = right_shift_target(
        loophole_outputs.logits,
        block_start=block_start,
        block_size=block_size,
    )
    loophole_ids, loophole_confidence = top1(loophole_logits)

    baseline_candidate = partial_block.clone()
    loophole_candidate = partial_block.clone()
    baseline_candidate[:, remaining_indices] = baseline_ids[:, remaining_indices]
    loophole_candidate[:, remaining_indices] = loophole_ids[:, remaining_indices]

    baseline_verifier_ids, baseline_verifier_confidence = verifier_top1(
        verifier_model,
        prefix,
        baseline_candidate,
    )
    loophole_verifier_ids, loophole_verifier_confidence = verifier_top1(
        verifier_model,
        prefix,
        loophole_candidate,
    )
    teacher_forced_ids, teacher_forced_confidence = verifier_top1(
        verifier_model,
        prefix,
        clean_block,
    )

    rows = []
    for index in range(block_size):
        is_remaining = index in remaining_indices
        base_candidate_match = (
            int(baseline_ids[0, index]) == int(baseline_verifier_ids[0, index])
            if is_remaining
            else None
        )
        loop_candidate_match = (
            int(loophole_ids[0, index]) == int(loophole_verifier_ids[0, index])
            if is_remaining
            else None
        )
        base_teacher_forced_match = (
            int(baseline_ids[0, index]) == int(teacher_forced_ids[0, index])
            if is_remaining
            else None
        )
        loop_teacher_forced_match = (
            int(loophole_ids[0, index]) == int(teacher_forced_ids[0, index])
            if is_remaining
            else None
        )
        rows.append(
            {
                "block_start": block_start,
                "block_offset": index,
                "absolute_position": block_start + index,
                "membership": "R" if is_remaining else "F",
                "headline_eligible": is_remaining,
                "clean": token_record(tokenizer, int(clean_block[0, index])),
                "first_pass": {
                    **token_record(tokenizer, int(first_ids[0, index])),
                    "confidence": float(first_confidence[0, index]),
                },
                "baseline_second_pass": {
                    **token_record(tokenizer, int(baseline_ids[0, index])),
                    "confidence": float(baseline_confidence[0, index]),
                },
                "loophole_second_pass": {
                    **token_record(tokenizer, int(loophole_ids[0, index])),
                    "confidence": float(loophole_confidence[0, index]),
                },
                "baseline_conditioned_verifier": {
                    **token_record(tokenizer, int(baseline_verifier_ids[0, index])),
                    "confidence": float(baseline_verifier_confidence[0, index]),
                },
                "loophole_conditioned_verifier": {
                    **token_record(tokenizer, int(loophole_verifier_ids[0, index])),
                    "confidence": float(loophole_verifier_confidence[0, index]),
                },
                "teacher_forced_verifier": {
                    **token_record(tokenizer, int(teacher_forced_ids[0, index])),
                    "confidence": float(teacher_forced_confidence[0, index]),
                },
                "candidate_conditioned_matches": {
                    "baseline": base_candidate_match,
                    "loophole": loop_candidate_match,
                },
                "teacher_forced_matches": {
                    "baseline": base_teacher_forced_match,
                    "loophole": loop_teacher_forced_match,
                },
            }
        )

    return {
        "excluded": False,
        "block_start": block_start,
        "fill_indices": fill_indices,
        "remaining_indices": remaining_indices,
        "clean_block": clean_block[0].cpu().tolist(),
        "partial_block": partial_block[0].cpu().tolist(),
        "baseline_candidate": baseline_candidate[0].cpu().tolist(),
        "loophole_candidate": loophole_candidate[0].cpu().tolist(),
        "accepted_prefix_length": {
            "baseline": accepted_prefix_length(
                baseline_candidate, baseline_verifier_ids
            ),
            "loophole": accepted_prefix_length(
                loophole_candidate, loophole_verifier_ids
            ),
        },
        "rows": rows,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate block traces into the requested counts and paired diagnostics."""
    included = [result for result in results if not result["excluded"]]
    excluded = [result for result in results if result["excluded"]]
    rows = [row for result in included for row in result["rows"]]
    eligible = [row for row in rows if row["headline_eligible"]]
    if not eligible:
        raise RuntimeError("No eligible second-pass positions remained")

    def count(context: str, model: str) -> int:
        return sum(bool(row[context][model]) for row in eligible)

    candidate_base = count("candidate_conditioned_matches", "baseline")
    candidate_loop = count("candidate_conditioned_matches", "loophole")
    teacher_base = count("teacher_forced_matches", "baseline")
    teacher_loop = count("teacher_forced_matches", "loophole")
    denominator = len(eligible)

    candidate = {
        "eligible_positions": denominator,
        "base_top1_matches": candidate_base,
        "loop_top1_matches": candidate_loop,
        "base_top1_match_rate": candidate_base / denominator,
        "loop_top1_match_rate": candidate_loop / denominator,
        "match_count_delta": candidate_loop - candidate_base,
        "match_rate_delta": (candidate_loop - candidate_base) / denominator,
    }
    teacher_forced = {
        "eligible_positions": denominator,
        "base_top1_matches": teacher_base,
        "loop_top1_matches": teacher_loop,
        "base_top1_match_rate": teacher_base / denominator,
        "loop_top1_match_rate": teacher_loop / denominator,
        "match_count_delta": teacher_loop - teacher_base,
        "match_rate_delta": (teacher_loop - teacher_base) / denominator,
        "both_match": sum(
            row["teacher_forced_matches"]["baseline"]
            and row["teacher_forced_matches"]["loophole"]
            for row in eligible
        ),
        "loop_only_match": sum(
            not row["teacher_forced_matches"]["baseline"]
            and row["teacher_forced_matches"]["loophole"]
            for row in eligible
        ),
        "base_only_match": sum(
            row["teacher_forced_matches"]["baseline"]
            and not row["teacher_forced_matches"]["loophole"]
            for row in eligible
        ),
        "neither_match": sum(
            not row["teacher_forced_matches"]["baseline"]
            and not row["teacher_forced_matches"]["loophole"]
            for row in eligible
        ),
    }
    accepted = {
        model: {
            "total": sum(
                result["accepted_prefix_length"][model] for result in included
            ),
            "mean": sum(
                result["accepted_prefix_length"][model] for result in included
            )
            / len(included),
            "per_block": [
                result["accepted_prefix_length"][model] for result in included
            ],
        }
        for model in ("baseline", "loophole")
    }
    return {
        "evaluated_blocks": len(included),
        "excluded_blocks": len(excluded),
        "exclusions": excluded,
        "candidate_conditioned": candidate,
        "teacher_forced_common_context": teacher_forced,
        "accepted_prefix_length": accepted,
        "blocks": included,
        "per_position": rows,
    }


def render_markdown(report: dict[str, Any], json_path: Path) -> str:
    """Render a compact human-readable result report."""
    candidate = report["results"]["candidate_conditioned"]
    teacher = report["results"]["teacher_forced_common_context"]
    accepted = report["results"]["accepted_prefix_length"]
    return "\n".join(
        [
            "# Qwen3-0.6B BD3LM speculative top-1 agreement results",
            "",
            f"Full trace: `{json_path.resolve()}`",
            "",
            (
                f"Evaluated {report['results']['evaluated_blocks']} block-size-4 "
                f"targets and {candidate['eligible_positions']} remaining-mask positions."
            ),
            "",
            "## Candidate-conditioned AR verification",
            "",
            "| Model | Top-1 matches | Eligible positions | Match rate |",
            "|---|---:|---:|---:|",
            (
                f"| Baseline BD3LM | {candidate['base_top1_matches']} | "
                f"{candidate['eligible_positions']} | "
                f"{candidate['base_top1_match_rate']:.2%} |"
            ),
            (
                f"| Loophole with `h_1` | {candidate['loop_top1_matches']} | "
                f"{candidate['eligible_positions']} | "
                f"{candidate['loop_top1_match_rate']:.2%} |"
            ),
            "",
            (
                f"Loophole minus baseline: **{candidate['match_count_delta']:+d} "
                f"matches ({candidate['match_rate_delta']:+.2%})**."
            ),
            "",
            "## Teacher-forced common-context diagnostic",
            "",
            "| Model | Top-1 matches | Eligible positions | Match rate |",
            "|---|---:|---:|---:|",
            (
                f"| Baseline BD3LM | {teacher['base_top1_matches']} | "
                f"{teacher['eligible_positions']} | "
                f"{teacher['base_top1_match_rate']:.2%} |"
            ),
            (
                f"| Loophole with `h_1` | {teacher['loop_top1_matches']} | "
                f"{teacher['eligible_positions']} | "
                f"{teacher['loop_top1_match_rate']:.2%} |"
            ),
            "",
            (
                f"Paired counts: both={teacher['both_match']}, "
                f"Loophole-only={teacher['loop_only_match']}, "
                f"baseline-only={teacher['base_only_match']}, "
                f"neither={teacher['neither_match']}."
            ),
            "",
            "## Strict greedy accepted prefix",
            "",
            (
                f"Baseline mean: {accepted['baseline']['mean']:.3f}/4; "
                f"Loophole mean: {accepted['loophole']['mean']:.3f}/4."
            ),
            "",
        ]
    )


def write_report(report: dict[str, Any], output: Path) -> None:
    """Write the complete JSON trace and a sibling Markdown summary."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(render_markdown(report, output), encoding="utf-8")
    print(f"Wrote {output.resolve()}", flush=True)
    print(f"Wrote {markdown_path.resolve()}", flush=True)


def main() -> None:
    """Run sharded block evaluation and write the merged result on rank zero."""
    args = parse_args()
    validate_args(args)
    rank, world_size, device = initialize_distributed(args.device)
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
    tokenizer_metadata = validate_tokenizers(
        baseline_tokenizer,
        loophole_tokenizer,
        verifier_tokenizer,
    )
    if baseline_tokenizer.mask_token_id is None:
        raise ValueError("The BD3LM tokenizer has no mask token")
    if baseline_tokenizer.pad_token_id is None:
        raise ValueError("The BD3LM tokenizer has no pad token")

    sample_text, sample_source = load_sample(args)
    sample_ids = baseline_tokenizer.encode(
        sample_text,
        add_special_tokens=False,
    )[: args.max_sample_tokens]
    block_starts, skipped_structural = choose_block_starts(
        sample_ids,
        baseline_tokenizer,
        block_size=args.block_size,
        minimum_prefix_tokens=args.minimum_prefix_tokens,
        max_blocks=args.max_blocks,
    )
    local_block_starts = block_starts[rank::world_size]
    if rank == 0:
        print(
            f"world_size={world_size} device_count={torch.cuda.device_count()} "
            f"sample_tokens={len(sample_ids)} blocks={len(block_starts)}",
            flush=True,
        )

    baseline_model, baseline_config = load_diffusion_model(
        args.baseline_checkpoint,
        expected_loophole_enabled=False,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    loophole_model, loophole_config = load_diffusion_model(
        args.loophole_checkpoint,
        expected_loophole_enabled=True,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    verifier_model, verifier_config = load_verifier_model(
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

    clean_ids = torch.tensor(
        sample_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    local_results = []
    for local_index, block_start in enumerate(local_block_starts, start=1):
        local_results.append(
            evaluate_block(
                clean_ids=clean_ids,
                block_start=block_start,
                block_size=args.block_size,
                baseline_model=baseline_model,
                loophole_model=loophole_model,
                verifier_model=verifier_model,
                tokenizer=baseline_tokenizer,
            )
        )
        print(
            f"rank={rank} completed={local_index}/{len(local_block_starts)} "
            f"block_start={block_start}",
            flush=True,
        )

    local_payload = {
        "rank": rank,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "block_starts": local_block_starts,
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
            key=lambda result: result["block_start"],
        )
        aggregated = aggregate(all_results)
        model_info = {
            "verifier": model_metadata(
                args.verifier_checkpoint, verifier_model, verifier_config
            ),
            "baseline": model_metadata(
                args.baseline_checkpoint, baseline_model, baseline_config
            ),
            "loophole": model_metadata(
                args.loophole_checkpoint, loophole_model, loophole_config
            ),
        }
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "qwen3_0.6b_bd3lm_speculative_top1_agreement",
            "protocol": {
                "block_size": args.block_size,
                "initial_target_mask_fraction": 1.0,
                "first_pass_fill_fraction": 0.5,
                "first_pass_proposer": "loophole",
                "second_pass_models": ["baseline", "loophole_with_h1"],
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
            "sample": {
                "source": sample_source,
                "sha256": hashlib.sha256(sample_text.encode("utf-8")).hexdigest(),
                "text": sample_text,
                "token_count_before_truncation": len(
                    baseline_tokenizer.encode(sample_text, add_special_tokens=False)
                ),
                "evaluated_token_count": len(sample_ids),
                "minimum_prefix_tokens": args.minimum_prefix_tokens,
                "requested_max_blocks": args.max_blocks,
                "selected_block_starts": block_starts,
                "structural_blocks_skipped": skipped_structural,
            },
            "execution": {
                "launch": "torchrun_data_parallel" if world_size > 1 else "single_process",
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "workers": worker_payloads,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "dtype": args.dtype,
                "attention_implementation": args.attn_implementation,
                "slurm_used": False,
                "slurm_note": "srun is unavailable on this host",
            },
            "results": aggregated,
        }
        write_report(report, args.output)
        print(json.dumps(aggregated["candidate_conditioned"], indent=2), flush=True)
        print(
            json.dumps(aggregated["teacher_forced_common_context"], indent=2),
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
