"""Score paired lm-eval generations with BD3LM diffusion perplexity.

Run after activating the ``dllm`` environment, for example:
```
python \
  /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/benchmark_generation_perplexity.py \
  --help
```
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

import dllm
from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask


DEFAULT_SEEDS = (314159, 271828, 161803)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-score baseline and Loophole lm-eval responses with the "
            "BD3LM training-aligned diffusion objective."
        )
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--loophole-checkpoint", required=True)
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--loophole-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--time-epsilon", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate settings before loading either checkpoint."""
    for path in (args.baseline_samples, args.loophole_samples):
        if not path.is_file():
            raise FileNotFoundError(f"Sample file does not exist: {path}")
    if args.num_generations <= 0:
        raise ValueError("num-generations must be positive")
    if args.max_tokens < 2:
        raise ValueError("max-tokens must be at least two")
    if args.block_size <= 0:
        raise ValueError("block-size must be positive")
    if not 0.0 < args.time_epsilon < 1.0:
        raise ValueError("time-epsilon must be in (0, 1)")
    if not args.seeds:
        raise ValueError("at least one Monte Carlo seed is required")


def load_responses(path: Path, num_generations: int) -> list[dict[str, Any]]:
    """Load one flexible-extract row for each of the first paired documents."""
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as sample_file:
        for line in sample_file:
            row = json.loads(line)
            if row.get("filter") != "flexible-extract":
                continue
            response = row.get("resps", [[""]])[0][0]
            if isinstance(response, str) and response.strip():
                rows[int(row["doc_id"])] = {
                    "doc_id": int(row["doc_id"]),
                    "response": response,
                }
    selected = [rows[doc_id] for doc_id in sorted(rows)[:num_generations]]
    if len(selected) != num_generations:
        raise ValueError(
            f"Requested {num_generations} generations but found {len(selected)}"
        )
    return selected


def tokenize_responses(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_tokens: int,
    block_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Tokenize each response and EOS-pad it to its own block boundary."""
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    sequences = []
    lengths = []
    original_lengths = []
    truncated = 0
    for row in rows:
        token_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        original_lengths.append(len(token_ids))
        if len(token_ids) >= max_tokens:
            token_ids = token_ids[: max_tokens - 1]
            truncated += 1
        if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
            token_ids.append(tokenizer.eos_token_id)
        padding = (-len(token_ids)) % block_size
        token_ids.extend([tokenizer.eos_token_id] * padding)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        maskable_mask = torch.ones_like(input_ids, dtype=torch.bool)
        maskable_mask[0] = False
        sequences.append(
            {
                "input_ids": input_ids,
                "maskable_mask": maskable_mask,
            }
        )
        lengths.append(len(token_ids))
    return sequences, {
        "doc_ids": [row["doc_id"] for row in rows],
        "generations": len(rows),
        "max_tokens_before_block_padding": max_tokens,
        "truncated_generations": truncated,
        "original_token_lengths": original_lengths,
        "evaluated_sequence_lengths": lengths,
        "evaluated_tokens_per_seed": sum(
            int(sequence["maskable_mask"].sum().item()) for sequence in sequences
        ),
        "first_token_excluded_per_generation": True,
        "eos_block_padding": True,
    }


def make_bd3lm_inputs(
    clean_ids: torch.Tensor,
    noised_ids: torch.Tensor,
    *,
    block_size: int,
) -> dict[str, torch.Tensor]:
    """Construct the two-stream inputs and block attention used in training."""
    _, length = clean_ids.shape
    device = clean_ids.device
    input_ids = torch.cat([noised_ids, clean_ids], dim=1)
    attention_mask = _create_bd3lm_attention_mask(
        b=None,
        h=None,
        q_idx=torch.arange(2 * length, device=device)[:, None],
        kv_idx=torch.arange(2 * length, device=device)[None, :],
        block_size=block_size,
        n=length,
    ).unsqueeze(0).unsqueeze(0)
    base_positions = torch.arange(length, device=device).unsqueeze(0)
    position_ids = torch.cat([base_positions, base_positions], dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
    }


def align_loophole_state(loophole_state: torch.Tensor) -> torch.Tensor:
    """Right-shift the recurrent state independently in both BD3LM streams."""
    if loophole_state.shape[1] % 2:
        raise ValueError("Loophole state must contain equal x_t and x_0 streams")
    stream_length = loophole_state.shape[1] // 2
    aligned_streams = []
    for stream in loophole_state.split(stream_length, dim=1):
        aligned_streams.append(
            torch.cat([torch.zeros_like(stream[:, :1]), stream[:, :-1]], dim=1)
        )
    return torch.cat(aligned_streams, dim=1)


@torch.inference_mode()
def forward_logits(
    model,
    clean_ids: torch.Tensor,
    noised_ids: torch.Tensor,
    *,
    block_size: int,
    loophole_enabled: bool,
) -> torch.Tensor:
    """Run a baseline pass or detached two-pass Loophole training forward."""
    model_inputs = make_bd3lm_inputs(
        clean_ids,
        noised_ids,
        block_size=block_size,
    )
    if loophole_enabled:
        pseudo_outputs = model(
            **model_inputs,
            loophole_state=None,
            return_loophole_state=True,
            loophole_enabled=True,
        )
        loophole_state = getattr(pseudo_outputs, "loophole_state", None)
        if loophole_state is None:
            raise RuntimeError("Loophole pseudo-forward returned no hidden state")
        outputs = model(
            **model_inputs,
            loophole_state=align_loophole_state(loophole_state.detach()),
            return_loophole_state=True,
            loophole_enabled=True,
        )
    else:
        outputs = model(**model_inputs)

    length = clean_ids.shape[1]
    logits = outputs.logits[:, :length]
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.inference_mode()
def evaluate_seed(
    model,
    sequences: list[dict[str, torch.Tensor]],
    tokenizer,
    *,
    seed: int,
    time_epsilon: float,
    block_size: int,
    device: torch.device,
    loophole_enabled: bool,
) -> dict[str, Any]:
    """Evaluate one shared Monte Carlo corruption seed."""
    generator = torch.Generator(device=device).manual_seed(seed)
    weighted_nll_sum = 0.0
    raw_nll_sum = 0.0
    masked_correct = 0
    masked_tokens = 0
    evaluated_tokens = 0
    started_at = time.perf_counter()

    for sequence in sequences:
        clean_ids = sequence["input_ids"].unsqueeze(0).to(device)
        maskable = sequence["maskable_mask"].unsqueeze(0).to(device)
        t = time_epsilon + (1.0 - time_epsilon) * torch.rand(
            (),
            device=device,
            generator=generator,
        )
        masked = (
            torch.rand(
                clean_ids.shape,
                device=device,
                generator=generator,
            )
            < t
        ) & maskable
        noised_ids = torch.where(
            masked,
            tokenizer.mask_token_id,
            clean_ids,
        )
        logits = forward_logits(
            model,
            clean_ids,
            noised_ids,
            block_size=block_size,
            loophole_enabled=loophole_enabled,
        )
        masked_logits = logits[masked].float()
        masked_targets = clean_ids[masked]
        token_nll = F.cross_entropy(
            masked_logits,
            masked_targets,
            reduction="none",
        )
        weighted_nll_sum += (token_nll / (t + 1e-6)).sum().item()
        raw_nll_sum += token_nll.sum().item()
        masked_correct += (
            masked_logits.argmax(dim=-1) == masked_targets
        ).sum().item()
        masked_tokens += int(masked.sum().item())
        evaluated_tokens += int(maskable.sum().item())

    diffusion_nll = weighted_nll_sum / evaluated_tokens
    raw_masked_nll = raw_nll_sum / masked_tokens
    return {
        "seed": seed,
        "diffusion_nll": diffusion_nll,
        "diffusion_perplexity": math.exp(diffusion_nll),
        "raw_masked_nll": raw_masked_nll,
        "raw_masked_perplexity": math.exp(raw_masked_nll),
        "masked_token_accuracy": masked_correct / masked_tokens,
        "masked_tokens": masked_tokens,
        "evaluated_tokens": evaluated_tokens,
        "runtime_seconds": time.perf_counter() - started_at,
    }


def aggregate_seed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate shared-seed estimates in NLL space."""
    diffusion_nll = statistics.fmean(row["diffusion_nll"] for row in results)
    raw_masked_nll = statistics.fmean(row["raw_masked_nll"] for row in results)
    return {
        "diffusion_nll": diffusion_nll,
        "diffusion_perplexity": math.exp(diffusion_nll),
        "diffusion_nll_sample_stdev": (
            statistics.stdev(row["diffusion_nll"] for row in results)
            if len(results) > 1
            else 0.0
        ),
        "raw_masked_nll": raw_masked_nll,
        "raw_masked_perplexity": math.exp(raw_masked_nll),
        "masked_token_accuracy": statistics.fmean(
            row["masked_token_accuracy"] for row in results
        ),
        "runtime_seconds": sum(row["runtime_seconds"] for row in results),
        "per_seed": results,
    }


def evaluate_model(
    checkpoint: str,
    corpora: dict[str, list[dict[str, torch.Tensor]]],
    tokenizer,
    args: argparse.Namespace,
    *,
    expected_loophole_enabled: bool,
) -> dict[str, Any]:
    """Load one checkpoint and score both response corpora."""
    checkpoint_config = transformers.AutoConfig.from_pretrained(checkpoint)
    checkpoint_loophole_enabled = bool(
        getattr(checkpoint_config, "loophole_enabled", False)
    )
    if checkpoint_loophole_enabled != expected_loophole_enabled:
        raise ValueError(
            f"Checkpoint {checkpoint} has loophole_enabled="
            f"{checkpoint_loophole_enabled}; expected {expected_loophole_enabled}"
        )
    model = dllm.utils.get_model(
        model_name_or_path=checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    model_results = {}
    for corpus_name, sequences in corpora.items():
        per_seed = []
        for seed in args.seeds:
            result = evaluate_seed(
                model,
                sequences,
                tokenizer,
                seed=seed,
                time_epsilon=args.time_epsilon,
                block_size=args.block_size,
                device=torch.device(args.device),
                loophole_enabled=expected_loophole_enabled,
            )
            per_seed.append(result)
            print(
                f"model={checkpoint} corpus={corpus_name} seed={seed} "
                f"diffusion_ppl={result['diffusion_perplexity']:.6f} "
                f"raw_masked_ppl={result['raw_masked_perplexity']:.6f} "
                f"accuracy={result['masked_token_accuracy']:.4%}",
                flush=True,
            )
        model_results[corpus_name] = aggregate_seed_results(per_seed)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint": checkpoint,
        "loophole_enabled": expected_loophole_enabled,
        "corpora": model_results,
    }


def main() -> None:
    """Cross-score the selected generations and write a JSON report."""
    args = parse_args()
    validate_args(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=args.baseline_checkpoint
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("Checkpoint tokenizer has no mask token")

    sample_paths = {
        "baseline_generations": args.baseline_samples,
        "loophole_generations": args.loophole_samples,
    }
    corpora = {}
    corpus_metadata = {}
    for corpus_name, sample_path in sample_paths.items():
        rows = load_responses(sample_path, args.num_generations)
        sequences, metadata = tokenize_responses(
            rows,
            tokenizer,
            max_tokens=args.max_tokens,
            block_size=args.block_size,
        )
        corpora[corpus_name] = sequences
        corpus_metadata[corpus_name] = {
            "sample_path": str(sample_path.resolve()),
            **metadata,
        }

    models = {
        "baseline": evaluate_model(
            args.baseline_checkpoint,
            corpora,
            tokenizer,
            args,
            expected_loophole_enabled=False,
        ),
        "loophole": evaluate_model(
            args.loophole_checkpoint,
            corpora,
            tokenizer,
            args,
            expected_loophole_enabled=True,
        ),
    }
    report = {
        "metric": {
            "name": "BD3LM diffusion pseudo-perplexity",
            "definition": (
                "exp(mean scheduler-weighted masked-token NLL), using the "
                "training two-stream block attention and right-shifted logits"
            ),
            "note": "This is not autoregressive next-token perplexity.",
            "scheduler": "LinearAlphaScheduler",
            "loss_weight": "1 / (t + 1e-6)",
            "time_epsilon": args.time_epsilon,
            "seeds": args.seeds,
            "block_size": args.block_size,
            "loophole_forward": (
                "detached pseudo-forward followed by a loss-bearing forward "
                "with independently right-shifted x_t/x_0 hidden states"
            ),
        },
        "corpora": corpus_metadata,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
