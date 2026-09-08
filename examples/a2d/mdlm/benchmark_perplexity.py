"""Measure paired MDLM diffusion perplexity for two local checkpoints.

Run from the repository root after activating the ``dllm`` environment, for example:
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/benchmark_perplexity.py --baseline-checkpoint /path/to/baseline --loophole-checkpoint /path/to/loophole --output /path/to/report.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset

import dllm


DEFAULT_SEEDS = (314159, 271828, 161803)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare two MDLM checkpoints with identical Monte Carlo diffusion "
            "corruptions. Lower diffusion perplexity is better."
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--loophole-checkpoint", type=Path)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        metavar="NAME=PATH,loophole=BOOL",
        help=(
            "Named model arm; repeat for multi-model comparisons. Example: "
            "loophole_teacher=/path/to/checkpoint,loophole=true"
        ),
    )
    parser.add_argument(
        "--reference-model",
        default="loophole_teacher",
        help="Model name used as the reference for generic multi-arm deltas.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", default="wikitext")
    parser.add_argument("--dataset-name", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Optional files passed to datasets.load_dataset (for example Parquet shards).",
    )
    parser.add_argument(
        "--row-split",
        default=None,
        help="Optional value used to filter a Parquet 'split_name' column.",
    )
    parser.add_argument(
        "--accepted-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only trace-metadata rows whose accepted column is true.",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Monte Carlo seeds shared by both checkpoints.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Optional deterministic block subset size; the default evaluates all blocks.",
    )
    parser.add_argument("--subset-seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--baseline-inference-loophole",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Loopholing while evaluating the baseline-trained checkpoint. "
            "This adds a strictly zero-initialized adapter when the checkpoint lacks one."
        ),
    )
    parser.add_argument(
        "--loophole-inference-loophole",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Loopholing while evaluating the Loophole-trained checkpoint.",
    )
    parser.add_argument(
        "--insert-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert EOS between non-empty source records before fixed-length packing.",
    )
    return parser.parse_args()


def parse_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Normalize legacy two-arm arguments and generic named model specs."""
    if args.model_spec:
        if args.baseline_checkpoint is not None or args.loophole_checkpoint is not None:
            raise ValueError("Do not combine --model-spec with legacy checkpoint flags")
        models = []
        names = set()
        for raw in args.model_spec:
            try:
                name_path, raw_loophole = raw.rsplit(",loophole=", 1)
                name, raw_path = name_path.split("=", 1)
            except ValueError as error:
                raise ValueError(
                    f"Invalid model spec {raw!r}; expected NAME=PATH,loophole=BOOL"
                ) from error
            normalized = raw_loophole.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Invalid Loopholing value in model spec {raw!r}")
            if not name or name in names:
                raise ValueError(f"Model names must be non-empty and unique: {name!r}")
            names.add(name)
            models.append(
                {
                    "name": name,
                    "checkpoint": Path(raw_path),
                    "inference_loophole_enabled": normalized == "true",
                }
            )
        return models
    if args.baseline_checkpoint is None or args.loophole_checkpoint is None:
        raise ValueError(
            "Supply both legacy checkpoint flags or at least one --model-spec"
        )
    return [
        {
            "name": "baseline",
            "checkpoint": args.baseline_checkpoint,
            "inference_loophole_enabled": args.baseline_inference_loophole,
        },
        {
            "name": "loophole",
            "checkpoint": args.loophole_checkpoint,
            "inference_loophole_enabled": args.loophole_inference_loophole,
        },
    ]


def validate_args(args: argparse.Namespace) -> None:
    """Reject settings that do not define the training-aligned estimator."""
    models = parse_model_specs(args)
    for checkpoint in (model["checkpoint"] for model in models):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if args.sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("sequence-length and batch-size must be positive")
    if not 0.0 < args.time_epsilon < 1.0:
        raise ValueError("time-epsilon must be in (0, 1)")
    if not args.seeds:
        raise ValueError("At least one Monte Carlo seed is required")
    if args.max_blocks is not None and args.max_blocks <= 0:
        raise ValueError("max-blocks must be positive when supplied")
    if args.row_split is not None and args.data_files is None:
        raise ValueError("--row-split requires --data-files")
    if args.accepted_only and args.data_files is None:
        raise ValueError("--accepted-only requires --data-files")


def pack_dataset(
    args: argparse.Namespace,
    tokenizer,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Tokenize non-empty records and concatenate them into full token blocks."""
    load_kwargs = {"split": args.split}
    if args.data_files is not None:
        load_kwargs["data_files"] = args.data_files
    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name or None,
        **load_kwargs,
    )
    if args.row_split is not None:
        if "split_name" not in dataset.column_names:
            raise KeyError("--row-split requires a 'split_name' dataset column")
        dataset = dataset.filter(
            lambda row: row["split_name"] == args.row_split,
            desc=f"Selecting split_name={args.row_split}",
        )
    if args.accepted_only:
        if "accepted" not in dataset.column_names:
            raise KeyError("--accepted-only requires an 'accepted' dataset column")
        dataset = dataset.filter(
            lambda row: bool(row["accepted"]),
            desc="Selecting accepted generated traces",
        )
    if args.text_field not in dataset.column_names:
        raise KeyError(
            f"Text field {args.text_field!r} is absent; columns={dataset.column_names}"
        )

    raw_texts = dataset[args.text_field]
    texts = [text for text in raw_texts if isinstance(text, str) and text.strip()]
    eos_id = tokenizer.eos_token_id
    if args.insert_eos and eos_id is None:
        raise ValueError("--insert-eos requires a tokenizer EOS token")

    token_ids: list[int] = []
    tokenize_batch_size = 256
    for start in range(0, len(texts), tokenize_batch_size):
        encoded = tokenizer(
            texts[start : start + tokenize_batch_size],
            add_special_tokens=False,
        )["input_ids"]
        for row_ids in encoded:
            token_ids.extend(row_ids)
            if args.insert_eos and (not row_ids or row_ids[-1] != eos_id):
                token_ids.append(eos_id)

    usable_tokens = (len(token_ids) // args.sequence_length) * args.sequence_length
    if usable_tokens == 0:
        raise ValueError("Dataset did not produce one complete evaluation block")
    blocks = torch.tensor(
        token_ids[:usable_tokens], dtype=torch.long
    ).reshape(-1, args.sequence_length)
    total_blocks = len(blocks)

    if args.max_blocks is not None and args.max_blocks < total_blocks:
        generator = torch.Generator().manual_seed(args.subset_seed)
        indices = torch.randperm(total_blocks, generator=generator)[: args.max_blocks]
        blocks = blocks[indices]

    metadata = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "row_split": args.row_split,
        "accepted_only": args.accepted_only,
        "data_files": args.data_files,
        "text_field": args.text_field,
        "raw_records": len(raw_texts),
        "nonempty_records": len(texts),
        "insert_eos": args.insert_eos,
        "sequence_length": args.sequence_length,
        "available_full_blocks": total_blocks,
        "evaluated_blocks": len(blocks),
        "evaluated_tokens_per_seed": blocks.numel(),
        "dropped_tail_tokens": len(token_ids) - usable_tokens,
        "subset_seed": args.subset_seed if len(blocks) < total_blocks else None,
    }
    return blocks, metadata


@torch.inference_mode()
def forward_logits(model, noised_ids: torch.Tensor, loophole_enabled: bool) -> torch.Tensor:
    """Run either the baseline forward or detached two-pass Loopholing forward."""
    if not loophole_enabled:
        return model(input_ids=noised_ids).logits

    pseudo_outputs = model(
        input_ids=noised_ids,
        loophole_state=None,
        return_loophole_state=True,
        loophole_enabled=True,
    )
    loophole_state = getattr(pseudo_outputs, "loophole_state", None)
    if loophole_state is None:
        raise RuntimeError("Loopholing pseudo-forward returned no loophole_state")
    return model(
        input_ids=noised_ids,
        loophole_state=loophole_state.detach(),
        return_loophole_state=True,
        loophole_enabled=True,
    ).logits


@torch.inference_mode()
def evaluate_seed(
    model,
    blocks: torch.Tensor,
    tokenizer,
    *,
    batch_size: int,
    time_epsilon: float,
    seed: int,
    device: torch.device,
    loophole_enabled: bool,
) -> dict[str, Any]:
    """Evaluate one shared Monte Carlo corruption seed."""
    generator = torch.Generator(device=device).manual_seed(seed)
    weighted_nll_sum = 0.0
    raw_masked_nll_sum = 0.0
    masked_correct = 0
    masked_tokens = 0
    evaluated_tokens = 0
    started_at = time.perf_counter()

    for start in range(0, len(blocks), batch_size):
        clean_ids = blocks[start : start + batch_size].to(device, non_blocking=True)
        batch, length = clean_ids.shape
        t = time_epsilon + (1.0 - time_epsilon) * torch.rand(
            batch, device=device, generator=generator
        )
        masked = torch.rand(
            (batch, length), device=device, generator=generator
        ) < t.unsqueeze(1)
        noised_ids = torch.where(masked, tokenizer.mask_token_id, clean_ids)
        logits = forward_logits(model, noised_ids, loophole_enabled)

        masked_logits = logits[masked].float()
        masked_targets = clean_ids[masked]
        token_nll = F.cross_entropy(masked_logits, masked_targets, reduction="none")
        sample_weights = (1.0 / (t + 1e-6)).unsqueeze(1).expand_as(masked)
        weighted_nll_sum += (
            token_nll * sample_weights[masked].float()
        ).sum().item()
        raw_masked_nll_sum += token_nll.sum().item()
        masked_correct += (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
        masked_tokens += masked.sum().item()
        evaluated_tokens += clean_ids.numel()

    nll = weighted_nll_sum / evaluated_tokens
    return {
        "seed": seed,
        "diffusion_nll": nll,
        "diffusion_perplexity": math.exp(nll),
        "raw_masked_nll": raw_masked_nll_sum / masked_tokens,
        "masked_token_accuracy": masked_correct / masked_tokens,
        "masked_tokens": masked_tokens,
        "evaluated_tokens": evaluated_tokens,
        "runtime_seconds": time.perf_counter() - started_at,
    }


def evaluate_checkpoint(
    checkpoint: Path,
    blocks: torch.Tensor,
    tokenizer,
    args: argparse.Namespace,
    *,
    expected_checkpoint_loophole_enabled: bool,
    inference_loophole_enabled: bool,
) -> dict[str, Any]:
    """Load and evaluate a checkpoint, then release its device memory."""
    device = torch.device(args.device)
    checkpoint_config = transformers.AutoConfig.from_pretrained(str(checkpoint))
    checkpoint_loophole_enabled = bool(
        getattr(checkpoint_config, "loophole_enabled", False)
    )
    if checkpoint_loophole_enabled != expected_checkpoint_loophole_enabled:
        raise ValueError(
            f"Checkpoint {checkpoint} has loophole_enabled="
            f"{checkpoint_loophole_enabled}; expected "
            f"{expected_checkpoint_loophole_enabled}"
        )

    zero_initialized_adapter_added = (
        inference_loophole_enabled and not checkpoint_loophole_enabled
    )
    model_config = None
    if zero_initialized_adapter_added:
        checkpoint_config.loophole_enabled = True
        model_config = checkpoint_config

    model = dllm.utils.get_model(
        model_name_or_path=str(checkpoint),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        config=model_config,
    )
    model.eval()
    model_has_loophole_adapter = bool(
        getattr(model.config, "loophole_enabled", False)
    )
    if inference_loophole_enabled and not model_has_loophole_adapter:
        raise ValueError(
            f"Loopholing inference requested for {checkpoint}, but no adapter was loaded"
        )
    if zero_initialized_adapter_added:
        adapter = model.model.loophole_norm
        nonzero_parameters = sum(
            torch.count_nonzero(parameter).item() for parameter in adapter.parameters()
        )
        if nonzero_parameters != 0:
            raise RuntimeError(
                "The synthesized baseline Loopholing adapter was not exactly zero"
            )

    per_seed = []
    for seed in args.seeds:
        result = evaluate_seed(
            model,
            blocks,
            tokenizer,
            batch_size=args.batch_size,
            time_epsilon=args.time_epsilon,
            seed=seed,
            device=device,
            loophole_enabled=inference_loophole_enabled,
        )
        per_seed.append(result)
        print(
            f"{checkpoint.name} inference_loophole={inference_loophole_enabled} "
            f"seed={seed}: "
            f"NLL={result['diffusion_nll']:.6f}, "
            f"PPL={result['diffusion_perplexity']:.6f}, "
            f"accuracy={result['masked_token_accuracy']:.4%}",
            flush=True,
        )

    aggregate_nll = statistics.fmean(row["diffusion_nll"] for row in per_seed)
    aggregate = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_loophole_enabled": checkpoint_loophole_enabled,
        "inference_loophole_enabled": inference_loophole_enabled,
        "zero_initialized_adapter_added": zero_initialized_adapter_added,
        "model_forwards_per_batch": 2 if inference_loophole_enabled else 1,
        "diffusion_nll": aggregate_nll,
        "diffusion_perplexity": math.exp(aggregate_nll),
        "diffusion_nll_sample_stdev": (
            statistics.stdev(row["diffusion_nll"] for row in per_seed)
            if len(per_seed) > 1
            else 0.0
        ),
        "raw_masked_nll": statistics.fmean(
            row["raw_masked_nll"] for row in per_seed
        ),
        "masked_token_accuracy": statistics.fmean(
            row["masked_token_accuracy"] for row in per_seed
        ),
        "runtime_seconds": sum(row["runtime_seconds"] for row in per_seed),
        "per_seed": per_seed,
    }
    aggregate["logical_tokens_per_second"] = (
        len(args.seeds) * blocks.numel() / aggregate["runtime_seconds"]
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return aggregate


def main() -> None:
    """Run the paired benchmark and write a JSON report."""
    args = parse_args()
    validate_args(args)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_specs = parse_model_specs(args)
    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=str(model_specs[0]["checkpoint"])
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("Checkpoint tokenizer has no mask token")
    blocks, dataset_metadata = pack_dataset(args, tokenizer)
    print(
        f"Packed {dataset_metadata['evaluated_blocks']} blocks "
        f"({dataset_metadata['evaluated_tokens_per_seed']} tokens per seed)",
        flush=True,
    )

    model_results = {}
    for spec in model_specs:
        checkpoint_config = transformers.AutoConfig.from_pretrained(
            spec["checkpoint"]
        )
        result = evaluate_checkpoint(
            spec["checkpoint"],
            blocks,
            tokenizer,
            args,
            expected_checkpoint_loophole_enabled=bool(
                getattr(checkpoint_config, "loophole_enabled", False)
            ),
            inference_loophole_enabled=spec["inference_loophole_enabled"],
        )
        model_results[spec["name"]] = result

    reference_name = args.reference_model
    if reference_name not in model_results:
        reference_name = model_specs[-1]["name"]
    reference = model_results[reference_name]
    comparisons = {}
    for name, result in model_results.items():
        paired_nll_delta = [
            row["diffusion_nll"] - reference_row["diffusion_nll"]
            for row, reference_row in zip(
                result["per_seed"], reference["per_seed"], strict=True
            )
        ]
        comparisons[name] = {
            "reference_model": reference_name,
            "perplexity_delta_percent": 100.0
            * (result["diffusion_perplexity"] / reference["diffusion_perplexity"] - 1.0),
            "nll_delta": result["diffusion_nll"] - reference["diffusion_nll"],
            "masked_accuracy_delta": (
                result["masked_token_accuracy"] - reference["masked_token_accuracy"]
            ),
            "paired_nll_delta": paired_nll_delta,
        }
    report = {
        "metric": {
            "name": "MDLM diffusion perplexity",
            "definition": (
                "exp(mean scheduler-weighted masked-token NLL over all maskable "
                "tokens), matching dllm.core.trainers.MDLMTrainer/PPLMetric"
            ),
            "scheduler": "LinearAlphaScheduler",
            "loss_weight": "1 / (t + 1e-6)",
            "time_epsilon": args.time_epsilon,
            "seeds": args.seeds,
            "note": "This is not autoregressive next-token perplexity.",
        },
        "dataset": dataset_metadata,
        "execution": {
            "device": str(torch.device(args.device)),
            "gpu": (
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.device(args.device).type == "cuda"
                else None
            ),
            "dtype": "bfloat16",
            "batch_size": args.batch_size,
        },
        "models": model_results,
        "comparisons": comparisons,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
