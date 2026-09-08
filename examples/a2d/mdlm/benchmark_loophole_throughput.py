"""Benchmark threshold-decoded generation throughput for named MDLM arms.

Run from the repository root after activating the ``dllm`` environment, for example:
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/benchmark_loophole_throughput.py --baseline-checkpoint /path/to/baseline --loophole-checkpoint /path/to/loophole --output /path/to/report.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure MDLM generation throughput for a 2x2 Loopholing ablation."
    )
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--loophole-checkpoint", type=Path)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        metavar="NAME=PATH,loophole=BOOL",
        help="Named model arm; repeat for a multi-model comparison.",
    )
    parser.add_argument(
        "--reference-model",
        default="loophole_teacher",
        help="Model name used as the reference for throughput and latency deltas.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument(
        "--end-to-end-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark only baseline/off and Loophole/on instead of all four cells.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--prompt",
        default="Explain why masked diffusion language models can generate in parallel.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional JSONL file containing a prompt or question field per row.",
    )
    parser.add_argument("--prompt-count", type=int, default=128)
    return parser.parse_args()


def parse_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Normalize legacy two-checkpoint arguments and named model specs."""
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
            "name": "baseline_trained",
            "checkpoint": args.baseline_checkpoint,
            "inference_loophole_enabled": False,
        },
        {
            "name": "loophole_trained",
            "checkpoint": args.loophole_checkpoint,
            "inference_loophole_enabled": True,
        },
    ]


def validate_args(args: argparse.Namespace) -> None:
    """Validate checkpoint paths and positive benchmark sizes."""
    for checkpoint in (model["checkpoint"] for model in parse_model_specs(args)):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    for name in (
        "batch_size",
        "max_new_tokens",
        "steps",
        "block_size",
        "warmup_iterations",
        "trials",
        "prompt_count",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.confidence_threshold != 0.85:
        raise ValueError("This experiment requires confidence-threshold=0.85")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise FileNotFoundError(args.prompt_file)


def load_prompt_texts(args: argparse.Namespace) -> list[str]:
    """Load the fixed evaluation prompt suite or construct a legacy repeat batch."""
    if args.prompt_file is None:
        return [args.prompt for _ in range(args.batch_size)]
    prompts = []
    for line in args.prompt_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt", row.get("question"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Every prompt-file row needs a non-empty prompt/question")
        prompts.append(prompt.strip())
        if len(prompts) == args.prompt_count:
            break
    if len(prompts) != args.prompt_count:
        raise ValueError(
            f"Expected {args.prompt_count} prompts, found {len(prompts)} in {args.prompt_file}"
        )
    return prompts


def load_training_arm(
    checkpoint: Path,
    *,
    expected_checkpoint_loophole_enabled: bool,
    add_loophole_adapter: bool,
) -> tuple[torch.nn.Module, bool]:
    """Load an arm with a zero adapter added to the baseline architecture."""
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
        add_loophole_adapter and not checkpoint_loophole_enabled
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

    if zero_initialized_adapter_added:
        adapter = model.model.loophole_norm
        nonzero_parameters = sum(
            torch.count_nonzero(parameter).item() for parameter in adapter.parameters()
        )
        if nonzero_parameters != 0:
            raise RuntimeError("The synthesized baseline adapter was not exactly zero")
    return model, zero_initialized_adapter_added


@torch.inference_mode()
def benchmark_mode(
    model,
    tokenizer,
    args: argparse.Namespace,
    *,
    inference_loophole_enabled: bool,
) -> dict[str, Any]:
    """Benchmark iterative generation in one inference mode."""
    prompt_texts = load_prompt_texts(args)
    prompts = [
        tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in prompt_texts
    ]
    prompt_batches = [
        prompts[start : start + args.batch_size]
        for start in range(0, len(prompts), args.batch_size)
    ]
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
    sample_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_size": args.block_size,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "loophole_enabled": inference_loophole_enabled,
        "confidence_threshold": args.confidence_threshold,
        "return_dict": True,
        "return_history": False,
        "return_step_entropy": True,
    }

    for _ in range(args.warmup_iterations):
        output = sampler.sample(prompt_batches[0], **sample_kwargs)
        if torch.any(
            output.sequences[:, -args.max_new_tokens :] == tokenizer.mask_token_id
        ):
            raise RuntimeError("Warmup sampling left masked tokens in the output")
    torch.cuda.synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)

    trial_seconds = []
    trial_model_forwards = []
    final_outputs = []
    model_forward_count = 0

    def count_model_forward(_module, _args):
        nonlocal model_forward_count
        model_forward_count += 1

    forward_hook = model.register_forward_pre_hook(count_model_forward)
    for _ in range(args.trials):
        model_forward_count = 0
        torch.cuda.synchronize(model.device)
        started_at = time.perf_counter()
        trial_outputs = []
        for prompt_batch in prompt_batches:
            output = sampler.sample(prompt_batch, **sample_kwargs)
            trial_outputs.append(output)
            if torch.any(
                output.sequences[:, -args.max_new_tokens :]
                == tokenizer.mask_token_id
            ):
                raise RuntimeError("Sampling left masked tokens in the output")
        torch.cuda.synchronize(model.device)
        trial_seconds.append(time.perf_counter() - started_at)
        trial_model_forwards.append(model_forward_count)
        final_outputs = trial_outputs
    forward_hook.remove()

    median_seconds = statistics.median(trial_seconds)
    generated_tokens = len(prompts) * args.max_new_tokens
    output = final_outputs[0]
    generated_ids = output.sequences[0, -args.max_new_tokens :].tolist()
    generated_bigrams = list(zip(generated_ids, generated_ids[1:]))
    token_counts = Counter(generated_ids)
    active_metrics = [
        metric
        for batch_output in final_outputs
        for metric in batch_output.step_metrics
        if torch.any(metric["remaining_masks"] > 0)
    ]
    threshold_tokens = sum(
        int(metric["threshold_accepted_tokens"].sum()) for metric in active_metrics
    )
    fallback_tokens = sum(
        int(metric["below_threshold_fallback_tokens"].sum())
        for metric in active_metrics
    )
    committed_tokens = threshold_tokens + fallback_tokens
    return {
        "inference_loophole_enabled": inference_loophole_enabled,
        "model_forwards_per_diffusion_step": 1,
        "trial_seconds": trial_seconds,
        "trial_model_forwards": trial_model_forwards,
        "median_model_forwards": statistics.median(trial_model_forwards),
        "median_seconds": median_seconds,
        "p50_seconds": median_seconds,
        "p95_seconds": (
            statistics.quantiles(trial_seconds, n=20, method="inclusive")[18]
            if len(trial_seconds) > 1
            else trial_seconds[0]
        ),
        "median_seconds_per_model_forward": median_seconds
        / statistics.median(trial_model_forwards),
        "median_sequences_per_second": len(prompts) / median_seconds,
        "median_generated_tokens_per_second": generated_tokens / median_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 2**30,
        "remaining_masks": int(
            (
                output.sequences[:, -args.max_new_tokens :]
                == tokenizer.mask_token_id
            )
            .sum()
            .item()
        ),
        "threshold_accepted_tokens": threshold_tokens,
        "scheduler_fallback_tokens": fallback_tokens,
        "threshold_accepted_fraction": (
            threshold_tokens / committed_tokens if committed_tokens else 0.0
        ),
        "scheduler_fallback_fraction": (
            fallback_tokens / committed_tokens if committed_tokens else 0.0
        ),
        "sample_generation": {
            "token_ids": generated_ids,
            "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "distinct_1": len(token_counts) / len(generated_ids),
            "distinct_2": (
                len(set(generated_bigrams)) / len(generated_bigrams)
                if generated_bigrams
                else 0.0
            ),
            "dominant_token_fraction": max(token_counts.values())
            / len(generated_ids),
        },
    }


def benchmark_training_arm(
    name: str,
    checkpoint: Path,
    tokenizer,
    args: argparse.Namespace,
    *,
    checkpoint_loophole_enabled: bool,
    inference_modes: tuple[bool, ...] = (False, True),
) -> dict[str, Any]:
    """Benchmark inference off and on for one trained checkpoint."""
    model, zero_initialized_adapter_added = load_training_arm(
        checkpoint,
        expected_checkpoint_loophole_enabled=checkpoint_loophole_enabled,
        add_loophole_adapter=any(inference_modes),
    )
    modes = {}
    for inference_loophole_enabled in inference_modes:
        result = benchmark_mode(
            model,
            tokenizer,
            args,
            inference_loophole_enabled=inference_loophole_enabled,
        )
        modes["inference_off" if not inference_loophole_enabled else "inference_on"] = (
            result
        )
        print(
            f"{name} inference_loophole={inference_loophole_enabled}: "
            f"{result['median_generated_tokens_per_second']:.3f} generated tok/s",
            flush=True,
        )

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_loophole_enabled": checkpoint_loophole_enabled,
        "zero_initialized_adapter_added": zero_initialized_adapter_added,
        **modes,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    """Run named threshold-throughput arms and write a JSON report."""
    args = parse_args()
    validate_args(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    model_specs = parse_model_specs(args)
    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=str(model_specs[0]["checkpoint"])
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("Checkpoint tokenizer has no mask token")

    models = {}
    for spec in model_specs:
        checkpoint_config = transformers.AutoConfig.from_pretrained(
            spec["checkpoint"]
        )
        checkpoint_loophole_enabled = bool(
            getattr(checkpoint_config, "loophole_enabled", False)
        )
        if spec["inference_loophole_enabled"] and not checkpoint_loophole_enabled:
            raise ValueError(
                f"{spec['name']} requests Loopholing but its checkpoint has no adapter"
            )
        models[spec["name"]] = benchmark_training_arm(
            spec["name"],
            spec["checkpoint"],
            tokenizer,
            args,
            checkpoint_loophole_enabled=checkpoint_loophole_enabled,
            inference_modes=(spec["inference_loophole_enabled"],),
        )
    reference_name = args.reference_model
    if reference_name not in models:
        reference_name = model_specs[-1]["name"]

    def only_mode_result(model_result: dict[str, Any]) -> dict[str, Any]:
        mode_names = [
            name
            for name in ("inference_off", "inference_on")
            if name in model_result
        ]
        if len(mode_names) != 1:
            raise ValueError(
                "Named throughput arms must contain exactly one inference mode"
            )
        return model_result[mode_names[0]]

    reference = only_mode_result(models[reference_name])
    comparisons = {}
    for name, model_result in models.items():
        result = only_mode_result(model_result)
        comparisons[name] = {
            "reference_model": reference_name,
            "generated_tokens_per_second_delta_percent": 100.0
            * (
                result["median_generated_tokens_per_second"]
                / reference["median_generated_tokens_per_second"]
                - 1.0
            ),
            "p50_latency_delta_percent": 100.0
            * (result["p50_seconds"] / reference["p50_seconds"] - 1.0),
            "p95_latency_delta_percent": 100.0
            * (result["p95_seconds"] / reference["p95_seconds"] - 1.0),
            "peak_allocated_gib_delta": (
                result["peak_allocated_gib"] - reference["peak_allocated_gib"]
            ),
            "median_model_forwards_delta": (
                result["median_model_forwards"]
                - reference["median_model_forwards"]
            ),
        }
    report = {
        "hardware": torch.cuda.get_device_name(torch.device(args.device)),
        "dtype": "bfloat16",
        "protocol": {
            "batch_size": args.batch_size,
            "prompt_count": len(load_prompt_texts(args)),
            "prompt_file": str(args.prompt_file.resolve()) if args.prompt_file else None,
            "mean_prompt_tokens": statistics.fmean(
                len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                for prompt in load_prompt_texts(args)
            ),
            "new_tokens_per_sequence": args.max_new_tokens,
            "diffusion_steps": args.steps,
            "block_size": args.block_size,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "confidence_threshold": args.confidence_threshold,
            "decoder": "confidence_threshold_with_scheduler_completion_floor",
            "warmup_iterations": args.warmup_iterations,
            "trials": args.trials,
            "throughput_definition": "generated tokens / synchronized wall time",
            "end_to_end_only": args.end_to_end_only,
        },
        "models": models,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
