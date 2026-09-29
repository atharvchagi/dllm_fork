"""Evaluate the released OpenWebText LDDM-U checkpoint.

Run from the repository root after activating the ``dllm`` environment:
``python -m dllm.pipelines.lddm_u.eval --help``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import wandb
from filelock import FileLock
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from . import algo, dataloader


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / ".models" / "LDDM-U" / "lddm_u_owt_1M.ckpt"
)
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "lddm_u"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "uniform_loophole" / "lddm_u_owt"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def parse_args() -> argparse.Namespace:
    """Parse evaluation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "ppl", "both"), default="both")
    parser.add_argument("--algorithm", choices=("lddm_u", "udlm"), default="lddm_u")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset", default="openwebtext-split")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--sample-batches", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--ppl-batches", type=int, default=16)
    parser.add_argument("--ppl-warmup-batches", type=int, default=1)
    parser.add_argument("--ppl-mc-samples", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-project", default="uniform-loophole")
    parser.add_argument("--wandb-group", default="owt-uniform-checkpoints")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--diagnostic-accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report random-time denoising token accuracy alongside perplexity.",
    )
    parser.add_argument(
        "--generative-ppl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score generated samples with GPT-2 Large (slow and sample-count sensitive).",
    )
    parser.add_argument(
        "--float64-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the paper's float64 posterior calculation during sampling.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or misleading evaluation settings."""
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    for name in (
        "batch_size",
        "length",
        "steps",
        "sample_batches",
        "ppl_mc_samples",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.ppl_batches < 0:
        raise ValueError("--ppl-batches cannot be negative; use 0 for the full split")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative")
    if args.ppl_warmup_batches < 0:
        raise ValueError("--ppl-warmup-batches cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not args.device.startswith("cuda"):
        raise ValueError("The official LDDM implementation requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def _load_config(checkpoint: dict[str, Any], args: argparse.Namespace):
    """Combine checkpoint architecture settings with official LDDM-U configs."""
    original = checkpoint["hyper_parameters"]["config"]
    config = OmegaConf.create(OmegaConf.to_container(original, resolve=False))
    OmegaConf.set_struct(config, False)
    config.algo = OmegaConf.load(CONFIG_DIR / "algo" / f"{args.algorithm}.yaml")
    data_path = CONFIG_DIR / "data" / f"{args.dataset}.yaml"
    if not data_path.is_file():
        choices = sorted(path.stem for path in (CONFIG_DIR / "data").glob("*.yaml"))
        raise ValueError(f"Unknown dataset config {args.dataset!r}; choose from {choices}")
    config.data = OmegaConf.load(data_path)
    config.data.cache_dir = str(args.cache_dir)
    config.model.length = args.length
    config.loader.batch_size = args.batch_size
    config.loader.eval_batch_size = args.batch_size
    config.loader.global_batch_size = args.batch_size
    config.loader.eval_global_batch_size = args.batch_size
    config.loader.num_workers = args.num_workers
    config.trainer.devices = 1
    config.trainer.num_nodes = 1
    config.trainer.accumulate_grad_batches = 1
    if config.algo.loophole:
        config.algo.self_cond_rate = 1.0
    config.sampling.steps = args.steps
    config.sampling.predictor = "ancestral"
    config.sampling.use_float64 = args.float64_sampling
    config.eval.generate_samples = False
    config.eval.compute_generative_perplexity = args.generative_ppl
    return config


def load_model(args: argparse.Namespace):
    """Load the exact official architecture and apply the released EMA weights."""
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = _load_config(checkpoint, args)
    tokenizer = dataloader.get_tokenizer(config)
    model_class = algo.LDDM_U if args.algorithm == "lddm_u" else algo.UDLM
    model = model_class(config, tokenizer)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.ema.load_state_dict(checkpoint["ema"])
    model = model.to(torch.device(args.device))
    model.ema.copy_to(model._get_parameters())
    model.ema = None
    model.eval()
    del checkpoint
    return model, tokenizer, config


def _sync(device: torch.device) -> None:
    """Synchronize CUDA before reading a wall-clock timer."""
    torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_samples(model, tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    """Generate fixed-length samples and measure batch-one sampling throughput."""
    device = model.device
    for _ in range(args.warmup_batches):
        model.generate_samples(num_samples=args.batch_size, num_steps=args.steps)
    _sync(device)

    rows = []
    all_text = []
    total_seconds = 0.0
    for batch_index in range(args.sample_batches):
        _sync(device)
        started = time.perf_counter()
        samples, _, _ = model.generate_samples(
            num_samples=args.batch_size,
            num_steps=args.steps,
        )
        _sync(device)
        seconds = time.perf_counter() - started
        total_seconds += seconds
        text_samples = tokenizer.batch_decode(samples.detach().cpu())
        all_text.extend(text_samples)
        for sample_index, text in enumerate(text_samples):
            rows.append(
                {
                    "batch_index": batch_index,
                    "sample_index": sample_index,
                    "sampling_seconds_for_batch": seconds,
                    "text": text,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    sequences = args.sample_batches * args.batch_size
    final_tokens = sequences * args.length
    summary: dict[str, Any] = {
        "mode": "unconditional_generation",
        "algorithm": args.algorithm,
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": args.batch_size,
        "sequence_length": args.length,
        "sampling_steps": args.steps,
        "warmup_batches": args.warmup_batches,
        "measured_batches": args.sample_batches,
        "measured_sequences": sequences,
        "sampling_seconds": total_seconds,
        "sequences_per_second": sequences / total_seconds,
        "final_sequence_tokens_per_second": final_tokens / total_seconds,
        "denoising_positions_per_second": final_tokens * args.steps / total_seconds,
        "samples_path": str(sample_path.resolve()),
    }
    if args.generative_ppl:
        model.metrics.gen_ppl.reset()
        model.metrics.record_generative_perplexity(
            all_text,
            args.length,
            device=device,
        )
        summary["generative_perplexity_gpt2_large"] = float(
            model.metrics.gen_ppl.compute().item()
        )
        summary["generative_perplexity_note"] = (
            "Estimate uses only the generated samples in this run."
        )

    path = args.output_dir / "generation_throughput.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _validation_dataset(config, tokenizer):
    """Build the validation split selected by the official data config."""
    name = config.data.valid
    mode = "test" if name in ("text8", "lm1b", "ag_news") else "validation"
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f".{name}-{mode}-len{config.model.length}.lock"
    with FileLock(lock_path):
        return dataloader.get_dataset(
            name,
            tokenizer,
            wrap=config.data.wrap,
            mode=mode,
            cache_dir=config.data.cache_dir,
            insert_eos=config.data.insert_valid_eos,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("valid_revision", None),
        )


@torch.inference_mode()
def _denoising_accuracy(model, input_ids, valid_tokens) -> tuple[int, int]:
    """Measure clean-token recovery at one uniformly sampled diffusion time."""
    t = model._sample_t(input_ids.shape[0], accum_step=None)
    _, alpha_t = model.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    sigma = model._sigma_from_alphat(alpha_t)
    xt = model.q_xt(input_ids, alpha_t)
    if model.config.algo.loophole:
        _, latent = model.forward(xt, sigma, prev_latent=None)
        log_x_theta, _ = model.forward(xt, sigma, prev_latent=latent)
    else:
        log_x_theta, _ = model.forward(xt, sigma, prev_latent=None)
    predicted = log_x_theta.argmax(dim=-1)
    mask = valid_tokens.bool()
    return int(((predicted == input_ids) & mask).sum()), int(mask.sum())


@torch.inference_mode()
def evaluate_perplexity(
    model,
    tokenizer,
    config,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Estimate diffusion NLL/perplexity on a bounded validation sample."""
    dataset = _validation_dataset(config, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    total_accuracy_tokens = 0
    completed_batches = 0
    model_seconds = 0.0
    measured_sequence_evaluations = 0
    _sync(model.device)
    evaluation_started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if args.ppl_batches and completed_batches >= args.ppl_batches:
            break
        input_ids = batch["input_ids"].to(model.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
        if batch_index < args.ppl_warmup_batches:
            for _ in range(args.ppl_mc_samples):
                model._loss(input_ids, attention_mask)
        for _ in range(args.ppl_mc_samples):
            _sync(model.device)
            model_started = time.perf_counter()
            losses = model._loss(input_ids, attention_mask)
            _sync(model.device)
            batch_model_seconds = time.perf_counter() - model_started
            model_seconds += batch_model_seconds
            total_nll += float(losses.nlls.item())
            total_tokens += int(losses.num_tokens.item())
            measured_sequence_evaluations += input_ids.shape[0]
            if args.diagnostic_accuracy:
                correct, tokens = _denoising_accuracy(
                    model,
                    input_ids,
                    attention_mask,
                )
                total_correct += correct
                total_accuracy_tokens += tokens
        completed_batches += 1
        if completed_batches % 100 == 0:
            target = args.ppl_batches or len(loader)
            print(
                f"PPL_PROGRESS {completed_batches}/{target} "
                f"model_seconds={model_seconds:.1f}",
                flush=True,
            )
    _sync(model.device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    if completed_batches == 0 or total_tokens == 0:
        raise RuntimeError("Validation dataset yielded no batches")

    mean_nll = total_nll / total_tokens
    summary: dict[str, Any] = {
        "mode": "ppl",
        "algorithm": args.algorithm,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_config": args.dataset,
        "validation_dataset": config.data.valid,
        "batch_size": args.batch_size,
        "sequence_length": args.length,
        "warmup_batches": args.ppl_warmup_batches,
        "requested_batches": args.ppl_batches or "all",
        "batches": completed_batches,
        "monte_carlo_samples_per_batch": args.ppl_mc_samples,
        "tokens": total_tokens,
        "nll": mean_nll,
        "bits_per_token": mean_nll / math.log(2),
        "perplexity": math.exp(mean_nll),
        "model_seconds": model_seconds,
        "end_to_end_evaluation_seconds": evaluation_seconds,
        "dataset_sequences_per_second": measured_sequence_evaluations / model_seconds,
        "dataset_tokens_per_second": total_tokens / model_seconds,
        "throughput_timing_scope": (
            "CUDA-synchronized LDDM-U NLL calls on held-out dataset sequences; "
            "excludes loading, tokenization, transfers, warmup, and diagnostic accuracy."
        ),
    }
    if args.diagnostic_accuracy:
        summary["denoising_token_accuracy"] = total_correct / total_accuracy_tokens
        summary["denoising_accuracy_tokens"] = total_accuracy_tokens
        summary["denoising_accuracy_note"] = (
            "Random-time clean-token recovery diagnostic; this is not a paper task-accuracy metric."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "perplexity.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    """Load the released checkpoint once and run the requested evaluations."""
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=run_config,
    )
    try:
        model, tokenizer, config = load_model(args)
        combined = {}
        if args.mode in ("sample", "both"):
            combined["unconditional_generation"] = evaluate_samples(
                model, tokenizer, args
            )
        if args.mode in ("ppl", "both"):
            combined["ppl"] = evaluate_perplexity(
                model, tokenizer, config, args
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(combined, indent=2) + "\n",
            encoding="utf-8",
        )
        flat_summary = {
            f"{section}/{key}": value
            for section, values in combined.items()
            for key, value in values.items()
        }
        numeric_metrics = {
            key: value
            for key, value in flat_summary.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        run.log(numeric_metrics)
        run.summary.update(flat_summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
