"""Generate frozen Qwen3-0.6B Loophole trajectories for offline distillation.

Run one GPU shard after activating the ``dllm`` environment:
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/generate_off_policy_traces.py --shard-index 0 --num-shards 64``.
Finalize all completed shards without a GPU with ``--finalize-only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from datasets import load_dataset

import dllm
from dllm.data.offpolicy_distillation import (
    DEFAULT_TRACE_FRACTIONS,
    TEACHER_HEAD_FILENAME,
    TRACE_FORMAT_VERSION,
    ThresholdTraceCollector,
    save_teacher_head,
    save_trace_tensor_shard,
    trim_snapshot_targets,
)


DEFAULT_TEACHER = Path(
    "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init/"
    "mdlm-opdlm-loophole-epoch5-len512-mask151669/checkpoint-final"
)
DEFAULT_OUTPUT = Path(
    "/nvme-data/atharvchagi/qwen3_0.6b_loophole_offpolicy"
)
CONFIDENCE_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    """Parse trace-generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-path", default="divelab/opdlm_train_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=8192,
        help="Reduce long-prompt batches to keep padded canvas tokens under this cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject configurations outside the fixed experiment protocol."""
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    for name in (
        "batch_size",
        "max_batch_tokens",
        "max_prompt_tokens",
        "max_new_tokens",
        "steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.block_size != args.max_new_tokens:
        raise ValueError("This experiment requires one full response block")
    if args.confidence_threshold != CONFIDENCE_THRESHOLD:
        raise ValueError(
            f"All experiment inference must use threshold {CONFIDENCE_THRESHOLD}"
        )
    if not args.finalize_only and not args.teacher_checkpoint.is_dir():
        raise FileNotFoundError(args.teacher_checkpoint)


def normalized_prompt(text: str) -> str:
    """Normalize whitespace and case for stable prompt deduplication."""
    return re.sub(r"\s+", " ", text).strip().lower()


def stable_hash(text: str, seed: int = 42) -> str:
    """Return a stable SHA-256 identifier independent of Python hash seeding."""
    return hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one artifact for the finalized manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(tokenizer, question: str) -> list[int]:
    """Render an explicit Qwen thinking-mode user prompt."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,
    )


def prepare_examples(dataset, tokenizer, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Deduplicate, length-filter, split, and shard the OPDLM prompt mix."""
    required = {"question", "domain", "source"}
    missing = required - set(dataset.column_names)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    unique: dict[str, dict[str, Any]] = {}
    for source_index, row in enumerate(dataset):
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            continue
        normalized = normalized_prompt(question)
        unique.setdefault(
            normalized,
            {
                "source_index": source_index,
                "question": question.strip(),
                "normalized_prompt": normalized,
                "domain": str(row["domain"]),
                "source": str(row["source"]),
                "ground_truth_answer": str(row.get("ground_truth_answer", "")),
                "tests_json": str(row.get("tests_json", "")),
            },
        )

    eligible = []
    for row in unique.values():
        prompt_ids = render_prompt(tokenizer, row["question"])
        if len(prompt_ids) > args.max_prompt_tokens:
            continue
        row = dict(row)
        row["prompt_ids"] = prompt_ids
        row["trace_id"] = stable_hash(row["normalized_prompt"], args.seed)
        eligible.append(row)

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        strata[(row["domain"], row["source"])].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: stable_hash(row["trace_id"] + "\0split", args.seed))
        validation_count = max(1, round(0.05 * len(rows)))
        for index, row in enumerate(rows):
            row["split_name"] = (
                "validation" if index < validation_count else "train"
            )

    selected = [
        row
        for row in eligible
        if int(row["trace_id"], 16) % args.num_shards == args.shard_index
    ]
    return sorted(selected, key=lambda row: (len(row["prompt_ids"]), row["trace_id"]))


def first_stop_index(token_ids: list[int], stop_ids: set[int]) -> int:
    """Return the number of generated content tokens before the first stop."""
    return next(
        (index for index, token_id in enumerate(token_ids) if token_id in stop_ids),
        len(token_ids),
    )


def parse_reasoning(text: str) -> tuple[str, str]:
    """Split Qwen think tags when the generated response contains both markers."""
    match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
    if match is None:
        return text.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def degeneration_quality_flags(token_ids: list[int]) -> list[str]:
    """Flag highly repetitive generations without removing them from training."""
    if not token_ids:
        return []
    counts: dict[int, int] = defaultdict(int)
    for token_id in token_ids:
        counts[token_id] += 1
    flags = []
    if max(counts.values()) / len(token_ids) > 0.5:
        flags.append("dominant_token")
    if len(token_ids) >= 8:
        bigrams = list(zip(token_ids, token_ids[1:]))
        if len(set(bigrams)) / len(bigrams) < 0.1:
            flags.append("low_distinct_2")
    return flags


def per_sample_step_stats(step_metrics, row: int) -> dict[str, Any]:
    """Aggregate threshold-decoding diagnostics for one batch row."""
    active = [
        metric
        for metric in step_metrics
        if int(metric["remaining_masks"][row]) > 0
    ]
    threshold_tokens = sum(
        int(metric["threshold_accepted_tokens"][row]) for metric in active
    )
    fallback_tokens = sum(
        int(metric["below_threshold_fallback_tokens"][row]) for metric in active
    )
    transferred = threshold_tokens + fallback_tokens
    return {
        "decode_steps": len(active),
        "threshold_accepted_tokens": threshold_tokens,
        "scheduler_fallback_tokens": fallback_tokens,
        "threshold_accepted_fraction": (
            threshold_tokens / transferred if transferred else 0.0
        ),
        "scheduler_fallback_fraction": (
            fallback_tokens / transferred if transferred else 0.0
        ),
    }


def shard_paths(output_dir: Path, shard_index: int) -> tuple[Path, Path, Path]:
    """Return metadata, tensor, and manifest paths for one shard."""
    suffix = f"{shard_index:05d}"
    return (
        output_dir / f"metadata-{suffix}.parquet",
        output_dir / f"tensors-{suffix}.safetensors",
        output_dir / f"manifest-{suffix}.json",
    )


def completed_shard_is_valid(output_dir: Path, shard_index: int) -> bool:
    """Check whether a previously completed shard can be safely reused."""
    metadata_path, tensor_path, manifest_path = shard_paths(output_dir, shard_index)
    if not all(path.is_file() for path in (metadata_path, tensor_path, manifest_path)):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("metadata_sha256") == sha256_file(metadata_path)
        and manifest.get("tensor_sha256") == sha256_file(tensor_path)
    )


def generate_shard(args: argparse.Namespace) -> None:
    """Generate and atomically record one deterministic trace shard."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and completed_shard_is_valid(
        args.output_dir, args.shard_index
    ):
        print(f"Shard {args.shard_index} is already complete and valid", flush=True)
        return

    transformers.set_seed(args.seed)
    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=str(args.teacher_checkpoint)
    )
    dataset = load_dataset(args.dataset_path, split=args.split)
    examples = prepare_examples(dataset, tokenizer, args)
    if not examples:
        raise ValueError(f"No examples were assigned to shard {args.shard_index}")

    config = transformers.AutoConfig.from_pretrained(args.teacher_checkpoint)
    if not bool(getattr(config, "loophole_enabled", False)):
        raise ValueError("Teacher checkpoint is not Loopholing-enabled")
    model = dllm.utils.get_model(
        model_name_or_path=str(args.teacher_checkpoint),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
    stop_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, getattr(tokenizer, "eot_token_id", None))
        if token_id is not None
    }

    metadata_rows: list[dict[str, Any]] = []
    tensor_rows = []
    rejected = 0
    start = 0
    while start < len(examples):
        batch_size = min(args.batch_size, len(examples) - start)
        while batch_size > 1:
            longest_canvas = (
                len(examples[start + batch_size - 1]["prompt_ids"])
                + args.max_new_tokens
            )
            if batch_size * longest_canvas <= args.max_batch_tokens:
                break
            batch_size -= 1
        batch = examples[start : start + batch_size]
        prompts = [row["prompt_ids"] for row in batch]
        collector = ThresholdTraceCollector(
            prompt_lengths=[len(prompt) for prompt in prompts],
            initial_response_tokens=args.max_new_tokens,
            target_fractions=DEFAULT_TRACE_FRACTIONS,
        )
        outputs = sampler.sample(
            prompts,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_size=args.block_size,
            remasking="low_confidence",
            confidence_threshold=args.confidence_threshold,
            temperature=0.0,
            cfg_scale=0.0,
            stochastic_transfer=False,
            loophole_enabled=True,
            return_dict=True,
            return_history=False,
            return_step_entropy=True,
            distillation_step_callback=collector,
        )
        for row_index, source_row in enumerate(batch):
            prompt_length = len(prompts[row_index])
            generated = outputs.sequences[
                row_index, prompt_length : prompt_length + args.max_new_tokens
            ].tolist()
            completion_length = first_stop_index(generated, stop_ids)
            completion_ids = generated[:completion_length]
            completion_text = tokenizer.decode(
                completion_ids, skip_special_tokens=False
            ).strip()
            reasoning, answer = parse_reasoning(completion_text)
            has_masks = tokenizer.mask_token_id in generated
            snapshots = [
                trimmed
                for snapshot in collector.snapshots[row_index]
                if (
                    trimmed := trim_snapshot_targets(snapshot, completion_length)
                )
                is not None
            ]
            accepted = bool(
                completion_ids
                and completion_text
                and not has_masks
                and len(snapshots) == len(DEFAULT_TRACE_FRACTIONS)
            )
            quality_flags = degeneration_quality_flags(completion_ids)
            if completion_length == args.max_new_tokens:
                quality_flags.append("hit_max_tokens")
            if not completion_text:
                quality_flags.append("empty")
            if has_masks:
                quality_flags.append("remaining_masks")
            if len(snapshots) != len(DEFAULT_TRACE_FRACTIONS):
                quality_flags.append("incomplete_snapshot_levels")
            stats = per_sample_step_stats(outputs.step_metrics, row_index)
            metadata_index = len(metadata_rows)
            metadata_rows.append(
                {
                    **{
                        key: source_row[key]
                        for key in (
                            "trace_id",
                            "source_index",
                            "question",
                            "domain",
                            "source",
                            "ground_truth_answer",
                            "tests_json",
                            "split_name",
                        )
                    },
                    "prompt_token_ids": prompts[row_index],
                    "completion_token_ids": completion_ids,
                    "completion": completion_text,
                    "reasoning": reasoning,
                    "answer": answer,
                    "evaluation_text": source_row["question"]
                    + "\n\n"
                    + completion_text,
                    "accepted": accepted,
                    "quality_flags": quality_flags,
                    "snapshot_count": len(snapshots),
                    "completion_tokens": completion_length,
                    **stats,
                }
            )
            if not accepted:
                rejected += 1
                continue
            split_code = 0 if source_row["split_name"] == "train" else 1
            tensor_rows.extend(
                (metadata_index, split_code, snapshot) for snapshot in snapshots
            )
        print(
            f"shard={args.shard_index} generated={min(start + len(batch), len(examples))}/"
            f"{len(examples)} accepted={len(metadata_rows) - rejected}",
            flush=True,
        )
        start += len(batch)
        del outputs, collector

    if not tensor_rows:
        raise RuntimeError("The shard produced no accepted trace snapshots")
    metadata_path, tensor_path, manifest_path = shard_paths(
        args.output_dir, args.shard_index
    )
    metadata_tmp = metadata_path.with_suffix(".parquet.tmp")
    tensor_tmp = tensor_path.with_suffix(".safetensors.tmp")
    pq.write_table(pa.Table.from_pylist(metadata_rows), metadata_tmp)
    save_trace_tensor_shard(tensor_tmp, tensor_rows, config.hidden_size)
    metadata_tmp.replace(metadata_path)
    tensor_tmp.replace(tensor_path)

    if args.shard_index == 0:
        head_path = args.output_dir / TEACHER_HEAD_FILENAME
        head_tmp = head_path.with_suffix(".safetensors.tmp")
        save_teacher_head(head_tmp, model.lm_head.weight)
        head_tmp.replace(head_path)

    shard_manifest = {
        "format_version": TRACE_FORMAT_VERSION,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_examples": len(examples),
        "metadata_rows": len(metadata_rows),
        "accepted_examples": len(metadata_rows) - rejected,
        "rejected_examples": rejected,
        "snapshots": len(tensor_rows),
        "metadata_file": metadata_path.name,
        "tensor_file": tensor_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "tensor_sha256": sha256_file(tensor_path),
        "dataset_path": args.dataset_path,
        "dataset_fingerprint": dataset._fingerprint,
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "generation": {
            "enable_thinking": True,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "block_size": args.block_size,
            "remasking": "low_confidence",
            "confidence_threshold": args.confidence_threshold,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "stochastic_transfer": False,
            "loophole_enabled": True,
            "trace_fractions": list(DEFAULT_TRACE_FRACTIONS),
        },
        "seed": args.seed,
    }
    manifest_path.write_text(
        json.dumps(shard_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}", flush=True)


def finalize(args: argparse.Namespace) -> None:
    """Validate all shard checksums and write the dataset-level manifest."""
    manifests = []
    for shard_index in range(args.num_shards):
        if not completed_shard_is_valid(args.output_dir, shard_index):
            raise RuntimeError(f"Shard {shard_index} is missing or has a bad checksum")
        _, _, manifest_path = shard_paths(args.output_dir, shard_index)
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    head_path = args.output_dir / TEACHER_HEAD_FILENAME
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    reference = manifests[0]
    for manifest in manifests[1:]:
        for key in ("format_version", "num_shards", "dataset_path", "seed", "generation"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Shard manifests disagree on {key}")
    validation_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        table = pq.read_table(
            args.output_dir / manifest["metadata_file"],
            columns=["trace_id", "question", "domain", "split_name", "accepted"],
        )
        for row in table.to_pylist():
            if row["accepted"] and row["split_name"] == "validation":
                validation_by_domain[row["domain"]].append(row)
    prompt_rows = []
    for domain in ("code", "math", "science", "chat"):
        rows = sorted(validation_by_domain[domain], key=lambda row: row["trace_id"])
        if len(rows) < 32:
            raise ValueError(f"Validation split has fewer than 32 {domain} prompts")
        prompt_rows.extend(rows[:32])
    prompt_rows.sort(key=lambda row: (row["domain"], row["trace_id"]))
    prompt_file = args.output_dir / "throughput_prompts.jsonl"
    prompt_file.write_text(
        "".join(
            json.dumps(
                {
                    "trace_id": row["trace_id"],
                    "domain": row["domain"],
                    "question": row["question"],
                }
            )
            + "\n"
            for row in prompt_rows
        ),
        encoding="utf-8",
    )
    final_manifest = {
        "format_version": TRACE_FORMAT_VERSION,
        "dataset_path": reference["dataset_path"],
        "dataset_fingerprint": reference["dataset_fingerprint"],
        "teacher_checkpoint": reference["teacher_checkpoint"],
        "teacher_head_file": head_path.name,
        "teacher_head_sha256": sha256_file(head_path),
        "num_shards": args.num_shards,
        "metadata_shards": [manifest["metadata_file"] for manifest in manifests],
        "tensor_shards": [manifest["tensor_file"] for manifest in manifests],
        "throughput_prompt_file": prompt_file.name,
        "throughput_prompt_sha256": sha256_file(prompt_file),
        "source_examples": sum(manifest["source_examples"] for manifest in manifests),
        "accepted_examples": sum(
            manifest["accepted_examples"] for manifest in manifests
        ),
        "rejected_examples": sum(
            manifest["rejected_examples"] for manifest in manifests
        ),
        "snapshots": sum(manifest["snapshots"] for manifest in manifests),
        "generation": reference["generation"],
        "seed": reference["seed"],
        "shard_manifests": [f"manifest-{index:05d}.json" for index in range(args.num_shards)],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}", flush=True)


def main() -> None:
    """Generate one shard or finalize the complete trace dataset."""
    args = parse_args()
    validate_args(args)
    if args.finalize_only:
        finalize(args)
    else:
        generate_shard(args)


if __name__ == "__main__":
    main()
