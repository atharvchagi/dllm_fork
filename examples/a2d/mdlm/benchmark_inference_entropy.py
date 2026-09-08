"""Trace and plot predictive entropy during paired MDLM generation.

Run from the repository root after activating the ``dllm`` environment, for example:
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/benchmark_inference_entropy.py --baseline-checkpoint /path/to/baseline --loophole-checkpoint /path/to/loophole --output-dir /path/to/entropy-results``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset

import dllm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASELINE_LABEL = "Baseline trained + base inference"
LOOPHOLE_LABEL = "Loophole trained + Loopholing inference"
COLORS = {
    BASELINE_LABEL: "#4C78A8",
    LOOPHOLE_LABEL: "#F58518",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare masked-position predictive entropy across diffusion generation "
            "steps for the two end-to-end training/inference systems."
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--loophole-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", default="wikitext")
    parser.add_argument("--dataset-name", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--min-prompt-tokens", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Dynamically transfer every prediction at or above this probability; "
            "the diffusion schedule remains a minimum completion floor."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--log-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb-entity",
        default="atharv-chagi-texas-a-m-university",
    )
    parser.add_argument("--wandb-project", default="dllm-a2d-mdlm")
    parser.add_argument(
        "--wandb-run-name",
        default="qwen3-0.6b-mdlm-inference-entropy",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and experiment sizes."""
    for checkpoint in (args.baseline_checkpoint, args.loophole_checkpoint):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    for name in (
        "num_prompts",
        "min_prompt_tokens",
        "max_prompt_tokens",
        "batch_size",
        "max_new_tokens",
        "steps",
        "block_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.min_prompt_tokens > args.max_prompt_tokens:
        raise ValueError("min-prompt-tokens cannot exceed max-prompt-tokens")
    if args.block_size != args.max_new_tokens:
        raise ValueError(
            "This aligned step-wise experiment requires one generation block, so "
            "block-size must equal max-new-tokens"
        )
    if args.confidence_threshold is not None and not (
        0.0 <= args.confidence_threshold <= 1.0
    ):
        raise ValueError("confidence-threshold must be between 0 and 1")


def select_prompts(args: argparse.Namespace, tokenizer) -> tuple[list[dict], dict]:
    """Select a deterministic sample of tokenized external validation prompts."""
    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name or None,
        split=args.split,
    )
    if args.text_field not in dataset.column_names:
        raise KeyError(
            f"Text field {args.text_field!r} is absent; columns={dataset.column_names}"
        )

    indexed_texts = [
        (index, text)
        for index, text in enumerate(dataset[args.text_field])
        if isinstance(text, str) and text.strip()
    ]
    candidates = []
    tokenize_batch_size = 256
    for start in range(0, len(indexed_texts), tokenize_batch_size):
        batch = indexed_texts[start : start + tokenize_batch_size]
        encoded = tokenizer(
            [text for _, text in batch],
            add_special_tokens=False,
        )["input_ids"]
        for (source_index, text), token_ids in zip(batch, encoded, strict=True):
            if len(token_ids) < args.min_prompt_tokens:
                continue
            candidates.append(
                {
                    "source_index": source_index,
                    "text": text,
                    "input_ids": token_ids[: args.max_prompt_tokens],
                }
            )

    if len(candidates) < args.num_prompts:
        raise ValueError(
            f"Only {len(candidates)} eligible prompts for requested "
            f"num-prompts={args.num_prompts}"
        )
    selected = random.Random(args.seed).sample(candidates, args.num_prompts)
    for prompt_id, record in enumerate(selected):
        record["prompt_id"] = prompt_id

    metadata = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "text_field": args.text_field,
        "raw_records": len(dataset),
        "eligible_records": len(candidates),
        "selected_prompts": len(selected),
        "selection_seed": args.seed,
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "selected_source_indices": [row["source_index"] for row in selected],
    }
    return selected, metadata


def load_model(
    checkpoint: Path,
    *,
    expected_loophole_enabled: bool,
):
    """Load one checkpoint and validate its serialized Loopholing capability."""
    config = transformers.AutoConfig.from_pretrained(str(checkpoint))
    actual_loophole_enabled = bool(getattr(config, "loophole_enabled", False))
    if actual_loophole_enabled != expected_loophole_enabled:
        raise ValueError(
            f"Checkpoint {checkpoint} has loophole_enabled="
            f"{actual_loophole_enabled}; expected {expected_loophole_enabled}"
        )
    model = dllm.utils.get_model(
        model_name_or_path=str(checkpoint),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    return model


@torch.inference_mode()
def trace_model(
    label: str,
    checkpoint: Path,
    prompts: list[dict],
    tokenizer,
    args: argparse.Namespace,
    *,
    inference_loophole_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Generate from every prompt and collect per-prompt, per-step entropy."""
    model = load_model(
        checkpoint,
        expected_loophole_enabled=inference_loophole_enabled,
    )
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
    vocab_entropy = math.log(model.config.vocab_size)
    metric_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    batch_forward_steps: list[int] = []

    warmup_records = prompts[: args.batch_size]
    sampler.sample(
        [record["input_ids"] for record in warmup_records],
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_size=args.block_size,
        temperature=0.0,
        cfg_scale=0.0,
        stochastic_transfer=False,
        loophole_enabled=inference_loophole_enabled,
        return_dict=False,
        return_step_entropy=False,
        confidence_threshold=args.confidence_threshold,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for start in range(0, len(prompts), args.batch_size):
        batch_records = prompts[start : start + args.batch_size]
        output = sampler.sample(
            [record["input_ids"] for record in batch_records],
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_size=args.block_size,
            temperature=0.0,
            cfg_scale=0.0,
            stochastic_transfer=False,
            loophole_enabled=inference_loophole_enabled,
            return_dict=True,
            return_step_entropy=True,
            confidence_threshold=args.confidence_threshold,
        )
        if output.step_metrics is None or not output.step_metrics:
            raise RuntimeError(
                "The sampler returned no step metrics"
            )
        if len(output.step_metrics) > args.steps:
            raise RuntimeError(
                f"Sampler exceeded the {args.steps}-step budget with "
                f"{len(output.step_metrics)} steps"
            )
        batch_forward_steps.append(len(output.step_metrics))
        per_prompt_steps = {
            record["prompt_id"]: 0 for record in batch_records
        }

        for metrics in output.step_metrics:
            for row_index, record in enumerate(batch_records):
                remaining_masks = int(metrics["remaining_masks"][row_index].item())
                if remaining_masks == 0:
                    continue
                per_prompt_steps[record["prompt_id"]] = int(metrics["step"])
                entropy = float(metrics["mean_entropy_nats"][row_index].item())
                top1_probability = float(
                    metrics["mean_top1_probability"][row_index].item()
                )
                transferred_tokens = int(
                    metrics["transferred_tokens"][row_index].item()
                )
                threshold_accepted_tokens = int(
                    metrics["threshold_accepted_tokens"][row_index].item()
                )
                fallback_tokens = int(
                    metrics["below_threshold_fallback_tokens"][row_index].item()
                )
                metric_rows.append(
                    {
                        "model": label,
                        "prompt_id": record["prompt_id"],
                        "source_index": record["source_index"],
                        "step": int(metrics["step"]),
                        "remaining_masks": remaining_masks,
                        "remaining_mask_fraction": (
                            remaining_masks / args.max_new_tokens
                        ),
                        "entropy_nats": entropy,
                        "normalized_entropy": entropy / vocab_entropy,
                        "top1_probability": top1_probability,
                        "scheduled_minimum_tokens": int(
                            metrics["scheduled_minimum_tokens"][row_index].item()
                        ),
                        "transferred_tokens": transferred_tokens,
                        "remaining_masks_after": int(
                            metrics["remaining_masks_after"][row_index].item()
                        ),
                        "mean_transferred_confidence": float(
                            metrics["mean_transferred_confidence"][
                                row_index
                            ].item()
                        ),
                        "minimum_transferred_confidence": float(
                            metrics["minimum_transferred_confidence"][
                                row_index
                            ].item()
                        ),
                        "threshold_accepted_tokens": threshold_accepted_tokens,
                        "below_threshold_fallback_tokens": fallback_tokens,
                        "threshold_accepted_fraction": (
                            threshold_accepted_tokens / transferred_tokens
                        ),
                    }
                )

        for row_index, record in enumerate(batch_records):
            prompt_length = len(record["input_ids"])
            generated_ids = output.sequences[
                row_index,
                prompt_length : prompt_length + args.max_new_tokens,
            ].tolist()
            remaining_generated_masks = generated_ids.count(tokenizer.mask_token_id)
            generation_rows.append(
                {
                    "model": label,
                    "prompt_id": record["prompt_id"],
                    "source_index": record["source_index"],
                    "prompt": record["text"],
                    "prompt_token_count": prompt_length,
                    "decode_steps": per_prompt_steps[record["prompt_id"]],
                    "fully_decoded": remaining_generated_masks == 0,
                    "remaining_generated_masks": remaining_generated_masks,
                    "generated_token_ids": generated_ids,
                    "generated_text": tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ),
                }
            )

        completed = min(start + args.batch_size, len(prompts))
        print(f"{label}: traced {completed}/{len(prompts)} prompts", flush=True)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time
    logical_generated_tokens = sum(
        args.max_new_tokens - row["remaining_generated_masks"]
        for row in generation_rows
    )
    runtime = {
        "elapsed_seconds_with_entropy_instrumentation": elapsed_seconds,
        "logical_generated_tokens": logical_generated_tokens,
        "instrumented_logical_tokens_per_second": (
            logical_generated_tokens / elapsed_seconds
        ),
        "sampler_forward_steps_across_batches": int(sum(batch_forward_steps)),
        "mean_batch_forward_steps": float(np.mean(batch_forward_steps)),
        "maximum_batch_forward_steps": int(max(batch_forward_steps)),
    }

    del sampler, model
    gc.collect()
    torch.cuda.empty_cache()
    return metric_rows, generation_rows, runtime


def summarize_entropy(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate model curves and paired Loophole-minus-baseline deltas."""
    summary = (
        raw.groupby(["model", "step"], as_index=False)
        .agg(
            mean_entropy_nats=("entropy_nats", "mean"),
            std_entropy_nats=("entropy_nats", "std"),
            mean_normalized_entropy=("normalized_entropy", "mean"),
            mean_top1_probability=("top1_probability", "mean"),
            remaining_masks=("remaining_masks", "mean"),
            remaining_masks_after=("remaining_masks_after", "mean"),
            mean_transferred_tokens=("transferred_tokens", "mean"),
            mean_transferred_confidence=("mean_transferred_confidence", "mean"),
            threshold_accepted_tokens=("threshold_accepted_tokens", "sum"),
            below_threshold_fallback_tokens=(
                "below_threshold_fallback_tokens",
                "sum",
            ),
            prompt_count=("prompt_id", "count"),
        )
        .sort_values(["model", "step"])
    )
    summary["entropy_ci95"] = (
        1.96
        * summary["std_entropy_nats"]
        / np.sqrt(summary["prompt_count"])
    )

    paired = raw.pivot(
        index=["prompt_id", "source_index", "step"],
        columns="model",
        values="entropy_nats",
    ).reset_index()
    paired["loophole_minus_baseline_entropy_nats"] = (
        paired[LOOPHOLE_LABEL] - paired[BASELINE_LABEL]
    )
    paired = paired.dropna(
        subset=["loophole_minus_baseline_entropy_nats"]
    )
    paired_summary = (
        paired.groupby("step", as_index=False)
        .agg(
            mean_delta_nats=("loophole_minus_baseline_entropy_nats", "mean"),
            std_delta_nats=("loophole_minus_baseline_entropy_nats", "std"),
            prompt_count=("prompt_id", "count"),
        )
        .sort_values("step")
    )
    paired_summary["delta_ci95"] = (
        1.96
        * paired_summary["std_delta_nats"]
        / np.sqrt(paired_summary["prompt_count"])
    )

    per_prompt_delta = paired.groupby("prompt_id")[
        "loophole_minus_baseline_entropy_nats"
    ].mean()
    overall = {}
    for label in (BASELINE_LABEL, LOOPHOLE_LABEL):
        model_rows = raw[raw["model"] == label]
        prompt_means = model_rows.groupby("prompt_id")["entropy_nats"].mean()
        prompt_decode_steps = model_rows.groupby("prompt_id")["step"].max()
        total_transferred = int(model_rows["transferred_tokens"].sum())
        total_threshold_accepted = int(
            model_rows["threshold_accepted_tokens"].sum()
        )
        total_fallback = int(
            model_rows["below_threshold_fallback_tokens"].sum()
        )
        overall[label] = {
            "mean_entropy_nats_across_steps": float(prompt_means.mean()),
            "entropy_ci95_across_prompts": float(
                1.96 * prompt_means.std(ddof=1) / math.sqrt(len(prompt_means))
            ),
            "step_1_mean_entropy_nats": float(
                model_rows.loc[model_rows["step"] == 1, "entropy_nats"].mean()
            ),
            "final_step_mean_entropy_nats": float(
                model_rows.loc[
                    model_rows["step"] == model_rows["step"].max(),
                    "entropy_nats",
                ].mean()
            ),
            "mean_top1_probability_across_steps": float(
                model_rows["top1_probability"].mean()
            ),
            "mean_decode_steps": float(prompt_decode_steps.mean()),
            "median_decode_steps": float(prompt_decode_steps.median()),
            "p95_decode_steps": float(prompt_decode_steps.quantile(0.95)),
            "minimum_decode_steps": int(prompt_decode_steps.min()),
            "maximum_decode_steps": int(prompt_decode_steps.max()),
            "mean_tokens_transferred_per_active_step": float(
                model_rows["transferred_tokens"].mean()
            ),
            "mean_transferred_token_confidence": float(
                model_rows["mean_transferred_confidence"].mean()
            ),
            "total_transferred_tokens": total_transferred,
            "threshold_accepted_token_fraction": (
                total_threshold_accepted / total_transferred
            ),
            "below_threshold_fallback_token_fraction": (
                total_fallback / total_transferred
            ),
        }
    overall["paired_comparison"] = {
        "mean_loophole_minus_baseline_entropy_nats": float(
            per_prompt_delta.mean()
        ),
        "delta_ci95_across_prompts": float(
            1.96
            * per_prompt_delta.std(ddof=1)
            / math.sqrt(len(per_prompt_delta))
        ),
        "prompts_with_lower_loophole_entropy_fraction": float(
            (per_prompt_delta < 0).mean()
        ),
        "minimum_step_mean_delta_nats": float(
            paired_summary["mean_delta_nats"].min()
        ),
        "maximum_step_mean_delta_nats": float(
            paired_summary["mean_delta_nats"].max()
        ),
    }
    paired_decode_steps = (
        raw.groupby(["model", "prompt_id"])["step"]
        .max()
        .unstack("model")
    )
    decode_step_delta = (
        paired_decode_steps[LOOPHOLE_LABEL]
        - paired_decode_steps[BASELINE_LABEL]
    )
    overall["paired_comparison"].update(
        {
            "mean_loophole_minus_baseline_decode_steps": float(
                decode_step_delta.mean()
            ),
            "prompts_where_loophole_finished_in_fewer_steps_fraction": float(
                (decode_step_delta < 0).mean()
            ),
        }
    )
    return summary, paired_summary, overall


def configure_plot_style() -> None:
    """Apply a compact publication-oriented Matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 220,
        }
    )


def plot_entropy_curves(
    summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    output_dir: Path,
    confidence_threshold: float | None,
) -> list[Path]:
    """Create absolute-entropy and paired-delta plots with 95% intervals."""
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    entropy_axis, delta_axis = axes

    for label in (BASELINE_LABEL, LOOPHOLE_LABEL):
        rows = summary[summary["model"] == label]
        x = rows["step"].to_numpy()
        mean = rows["mean_entropy_nats"].to_numpy()
        ci95 = rows["entropy_ci95"].to_numpy()
        entropy_axis.plot(x, mean, label=label, color=COLORS[label], linewidth=2.2)
        entropy_axis.fill_between(
            x,
            mean - ci95,
            mean + ci95,
            color=COLORS[label],
            alpha=0.18,
            linewidth=0,
        )

    entropy_axis.set_title("Predictive entropy at remaining masked positions")
    entropy_axis.set_xlabel("Diffusion generation step")
    entropy_axis.set_ylabel("Mean entropy (nats; lower = sharper)")
    entropy_axis.legend(frameon=True)
    entropy_axis.set_xlim(1, int(summary["step"].max()))

    x = paired_summary["step"].to_numpy()
    delta = paired_summary["mean_delta_nats"].to_numpy()
    delta_ci95 = paired_summary["delta_ci95"].to_numpy()
    delta_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
    delta_axis.plot(x, delta, color="#7A5195", linewidth=2.2)
    delta_axis.fill_between(
        x,
        delta - delta_ci95,
        delta + delta_ci95,
        color="#7A5195",
        alpha=0.2,
        linewidth=0,
    )
    delta_axis.fill_between(
        x,
        delta,
        0.0,
        where=delta < 0,
        color="#2E8B57",
        alpha=0.12,
        interpolate=True,
    )
    delta_axis.set_title("Paired entropy change")
    delta_axis.set_xlabel("Diffusion generation step")
    delta_axis.set_ylabel("Loophole − baseline entropy (nats)")
    delta_axis.set_xlim(1, int(paired_summary["step"].max()))

    decoding_suffix = (
        ""
        if confidence_threshold is None
        else f" (confidence threshold = {confidence_threshold:.2f})"
    )
    figure.suptitle(
        "Qwen3-0.6B MDLM inference entropy on paired WikiText-2 prompts"
        + decoding_suffix,
        fontsize=14,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.01,
        "Shaded bands are 95% confidence intervals across prompts; negative deltas indicate sharper Loopholing predictions.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()
    png_path = output_dir / "inference_entropy_by_step.png"
    pdf_path = output_dir / "inference_entropy_by_step.pdf"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def plot_threshold_dynamics(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    confidence_threshold: float,
    max_new_tokens: int,
    num_prompts: int,
) -> list[Path]:
    """Plot confidence, dynamic transfer width, and completion progress."""
    configure_plot_style()
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.7))
    confidence_axis, transfer_axis, progress_axis = axes

    for label in (BASELINE_LABEL, LOOPHOLE_LABEL):
        rows = summary[summary["model"] == label]
        x = rows["step"].to_numpy()
        confidence_axis.plot(
            x,
            rows["mean_top1_probability"].to_numpy(),
            label=label,
            color=COLORS[label],
            linewidth=2.1,
        )
        transfer_axis.plot(
            x,
            rows["mean_transferred_tokens"].to_numpy(),
            label=label,
            color=COLORS[label],
            linewidth=2.1,
        )
        progress_axis.plot(
            x,
            rows["remaining_masks_after"].to_numpy() / max_new_tokens,
            label=f"{label}: masks left",
            color=COLORS[label],
            linewidth=2.1,
        )
        progress_axis.plot(
            x,
            rows["prompt_count"].to_numpy() / num_prompts,
            label=f"{label}: prompts active",
            color=COLORS[label],
            linestyle="--",
            linewidth=1.5,
        )

    confidence_axis.axhline(
        confidence_threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label=f"transfer threshold ({confidence_threshold:.2f})",
    )
    confidence_axis.set_title("Confidence at still-masked positions")
    confidence_axis.set_xlabel("Decoder iteration")
    confidence_axis.set_ylabel("Mean top-1 probability")
    confidence_axis.set_ylim(0.0, 1.02)
    confidence_axis.legend(frameon=True)

    transfer_axis.set_title("Dynamic parallel transfer width")
    transfer_axis.set_xlabel("Decoder iteration")
    transfer_axis.set_ylabel("Mean tokens committed per active prompt")
    transfer_axis.set_yscale("log", base=2)
    transfer_axis.legend(frameon=True)

    progress_axis.set_title("Decoding completion")
    progress_axis.set_xlabel("Decoder iteration")
    progress_axis.set_ylabel("Fraction")
    progress_axis.set_ylim(-0.02, 1.02)
    progress_axis.legend(frameon=True, fontsize=7.5)

    max_step = int(summary["step"].max())
    for axis in axes:
        axis.set_xlim(1, max_step)
    figure.suptitle(
        "Qwen3-0.6B MDLM confidence-threshold decoding dynamics",
        fontsize=14,
        y=1.02,
    )
    figure.tight_layout()
    png_path = output_dir / "confidence_threshold_dynamics.png"
    pdf_path = output_dir / "confidence_threshold_dynamics.pdf"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def plot_prompt_delta_distribution(raw: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the distribution of each prompt's step-averaged paired entropy delta."""
    configure_plot_style()
    paired = raw.pivot(
        index=["prompt_id", "step"],
        columns="model",
        values="entropy_nats",
    )
    prompt_delta = (
        paired[LOOPHOLE_LABEL] - paired[BASELINE_LABEL]
    ).groupby("prompt_id").mean()

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.hist(
        prompt_delta,
        bins=24,
        color="#7A5195",
        alpha=0.82,
        edgecolor="white",
    )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.3, label="No change")
    axis.axvline(
        prompt_delta.mean(),
        color="#D62728",
        linewidth=2,
        label=f"Mean = {prompt_delta.mean():+.3f} nats",
    )
    axis.set_title("Per-prompt average entropy change across generation")
    axis.set_xlabel("Loophole − baseline entropy (nats)")
    axis.set_ylabel("Prompt count")
    axis.legend(frameon=True)
    figure.tight_layout()
    path = output_dir / "inference_entropy_prompt_delta.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON objects one per line."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_to_wandb(
    args: argparse.Namespace,
    report: dict[str, Any],
    summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    output_files: list[Path],
) -> str:
    """Create a dedicated W&B evaluation run with curves, images, and artifacts."""
    import wandb

    tags = ["entropy", "loopholing", "qwen3-0.6b", "mdlm"]
    if args.confidence_threshold is not None:
        tags.append("confidence-threshold")
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        job_type="evaluation",
        tags=tags,
        config=report["protocol"],
    )
    report["wandb_run_url"] = run.url
    (args.output_dir / "inference_entropy_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    curve_table = wandb.Table(
        columns=["step", "model", "mean_entropy_nats", "entropy_ci95"]
    )
    for row in summary.itertuples(index=False):
        curve_table.add_data(
            int(row.step),
            row.model,
            float(row.mean_entropy_nats),
            float(row.entropy_ci95),
        )
    delta_table = wandb.Table(
        data=paired_summary[
            ["step", "mean_delta_nats", "delta_ci95"]
        ].values.tolist(),
        columns=["step", "mean_delta_nats", "delta_ci95"],
    )
    overall = report["summary"]
    log_payload = {
        "entropy/by_step_plot": wandb.Image(
            str(args.output_dir / "inference_entropy_by_step.png")
        ),
        "entropy/prompt_delta_plot": wandb.Image(
            str(args.output_dir / "inference_entropy_prompt_delta.png")
        ),
        "entropy/by_step_table": curve_table,
        "entropy/paired_delta_table": delta_table,
        "entropy/baseline_mean_nats": overall[BASELINE_LABEL][
            "mean_entropy_nats_across_steps"
        ],
        "entropy/loophole_mean_nats": overall[LOOPHOLE_LABEL][
            "mean_entropy_nats_across_steps"
        ],
        "entropy/loophole_minus_baseline_mean_nats": overall[
            "paired_comparison"
        ]["mean_loophole_minus_baseline_entropy_nats"],
        "entropy/prompts_with_lower_loophole_fraction": overall[
            "paired_comparison"
        ]["prompts_with_lower_loophole_entropy_fraction"],
        "decoding/baseline_mean_steps": overall[BASELINE_LABEL][
            "mean_decode_steps"
        ],
        "decoding/loophole_mean_steps": overall[LOOPHOLE_LABEL][
            "mean_decode_steps"
        ],
        "decoding/baseline_instrumented_tokens_per_second": overall[
            BASELINE_LABEL
        ]["runtime"]["instrumented_logical_tokens_per_second"],
        "decoding/loophole_instrumented_tokens_per_second": overall[
            LOOPHOLE_LABEL
        ]["runtime"]["instrumented_logical_tokens_per_second"],
    }
    if args.confidence_threshold is not None:
        log_payload["decoding/confidence_threshold_dynamics"] = wandb.Image(
            str(args.output_dir / "confidence_threshold_dynamics.png")
        )
        log_payload["decoding/confidence_threshold"] = args.confidence_threshold
    run.log(log_payload)
    artifact_name = (
        "qwen3-0.6b-mdlm-inference-entropy"
        if args.confidence_threshold is None
        else "qwen3-0.6b-mdlm-confidence-threshold-entropy"
    )
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation",
        metadata=report["protocol"],
    )
    for path in output_files:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run_url = run.url
    run.finish()
    return run_url


def main() -> None:
    """Run paired inference traces, save raw data, plot, and optionally log W&B."""
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=str(args.baseline_checkpoint)
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("Checkpoint tokenizer has no mask token")
    prompts, dataset_metadata = select_prompts(args, tokenizer)
    print(
        f"Selected {len(prompts)} paired prompts from "
        f"{dataset_metadata['eligible_records']} eligible records",
        flush=True,
    )

    baseline_metrics, baseline_generations, baseline_runtime = trace_model(
        BASELINE_LABEL,
        args.baseline_checkpoint,
        prompts,
        tokenizer,
        args,
        inference_loophole_enabled=False,
    )
    loophole_metrics, loophole_generations, loophole_runtime = trace_model(
        LOOPHOLE_LABEL,
        args.loophole_checkpoint,
        prompts,
        tokenizer,
        args,
        inference_loophole_enabled=True,
    )
    raw = pd.DataFrame(baseline_metrics + loophole_metrics)
    if raw.empty or raw["entropy_nats"].isna().any():
        raise RuntimeError(
            f"Expected finite metric rows, got {len(raw)} rows"
        )
    expected_prompt_pairs = {
        (label, prompt_id)
        for label in (BASELINE_LABEL, LOOPHOLE_LABEL)
        for prompt_id in range(args.num_prompts)
    }
    observed_prompt_pairs = set(zip(raw["model"], raw["prompt_id"], strict=True))
    if observed_prompt_pairs != expected_prompt_pairs:
        raise RuntimeError("Not every model/prompt pair produced entropy metrics")
    generations = baseline_generations + loophole_generations
    if not all(row["fully_decoded"] for row in generations):
        incomplete = sum(not row["fully_decoded"] for row in generations)
        raise RuntimeError(f"{incomplete} generations retained mask tokens")
    summary, paired_summary, overall = summarize_entropy(raw)
    overall[BASELINE_LABEL]["runtime"] = baseline_runtime
    overall[LOOPHOLE_LABEL]["runtime"] = loophole_runtime

    raw_path = args.output_dir / "inference_entropy_raw.csv"
    summary_path = args.output_dir / "inference_entropy_by_step.csv"
    paired_path = args.output_dir / "inference_entropy_paired_delta_by_step.csv"
    generations_path = args.output_dir / "inference_entropy_generations.jsonl"
    report_path = args.output_dir / "inference_entropy_report.json"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_summary.to_csv(paired_path, index=False)
    write_jsonl(
        generations_path,
        generations,
    )
    plot_paths = plot_entropy_curves(
        summary,
        paired_summary,
        args.output_dir,
        args.confidence_threshold,
    )
    plot_paths.append(plot_prompt_delta_distribution(raw, args.output_dir))
    if args.confidence_threshold is not None:
        plot_paths.extend(
            plot_threshold_dynamics(
                summary,
                args.output_dir,
                confidence_threshold=args.confidence_threshold,
                max_new_tokens=args.max_new_tokens,
                num_prompts=args.num_prompts,
            )
        )

    report = {
        "metric": {
            "name": "predictive entropy at currently masked generation positions",
            "definition": "mean(-sum_v p(v) log p(v)) in natural-log units",
            "units": "nats",
            "interpretation": "lower entropy means a sharper token distribution",
        },
        "models": {
            BASELINE_LABEL: str(args.baseline_checkpoint.resolve()),
            LOOPHOLE_LABEL: str(args.loophole_checkpoint.resolve()),
        },
        "dataset": dataset_metadata,
        "protocol": {
            "hardware": torch.cuda.get_device_name(torch.device(args.device)),
            "dtype": "bfloat16 model logits; float32 entropy computation",
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "diffusion_steps": args.steps,
            "block_size": args.block_size,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "stochastic_transfer": False,
            "decoder": (
                "static_scheduler_quota"
                if args.confidence_threshold is None
                else "confidence_threshold_with_scheduler_completion_floor"
            ),
            "confidence_threshold": args.confidence_threshold,
            "threshold_rule": (
                None
                if args.confidence_threshold is None
                else (
                    "At each iteration, commit every currently masked token whose "
                    "greedy prediction probability is at least the threshold. If "
                    "fewer qualify than the deterministic diffusion schedule's "
                    "quota, fill the shortfall with the highest-confidence tokens."
                )
            ),
            "paired_prompt_count": args.num_prompts,
            "confidence_intervals": "normal 95% intervals across paired prompts",
        },
        "summary": overall,
        "files": {
            "raw_csv": str(raw_path.resolve()),
            "step_summary_csv": str(summary_path.resolve()),
            "paired_delta_csv": str(paired_path.resolve()),
            "generations_jsonl": str(generations_path.resolve()),
            "entropy_plot_png": str(plot_paths[0].resolve()),
            "entropy_plot_pdf": str(plot_paths[1].resolve()),
            "prompt_delta_plot_png": str(plot_paths[2].resolve()),
        },
    }
    if args.confidence_threshold is not None:
        report["files"].update(
            {
                "threshold_dynamics_png": str(plot_paths[3].resolve()),
                "threshold_dynamics_pdf": str(plot_paths[4].resolve()),
            }
        )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    output_files = [
        raw_path,
        summary_path,
        paired_path,
        generations_path,
        report_path,
        *plot_paths,
    ]
    if args.log_wandb:
        report["wandb_run_url"] = log_to_wandb(
            args,
            report,
            summary,
            paired_summary,
            output_files,
        )
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(overall, indent=2), flush=True)
    print(f"Wrote {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
