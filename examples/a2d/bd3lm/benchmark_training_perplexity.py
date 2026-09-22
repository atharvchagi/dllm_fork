"""Compare BD3LM checkpoints on fixed training sequences.

Run after activating the ``dllm`` environment:
```
python \
  /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/benchmark_training_perplexity.py \
  --help
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import dllm
from dllm.utils.data import default_sft_map_fn
from benchmark_generation_perplexity import evaluate_model


DEFAULT_DATASET = (
    "Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy"
)
DEFAULT_SEEDS = (314159, 271828, 161803)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Score both checkpoints on identical, complete training examples "
            "with the BD3LM diffusion objective."
        )
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--loophole-checkpoint", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-predicted-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--time-epsilon", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate settings before loading the dataset or models."""
    if args.num_sequences <= 0:
        raise ValueError("num-sequences must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.min_predicted_tokens <= 0:
        raise ValueError("min-predicted-tokens must be positive")
    if args.block_size <= 0:
        raise ValueError("block-size must be positive")
    if not 0.0 < args.time_epsilon < 1.0:
        raise ValueError("time-epsilon must be in (0, 1)")
    if not args.seeds:
        raise ValueError("at least one Monte Carlo seed is required")


def load_training_sequences(
    args: argparse.Namespace,
    tokenizer,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Select complete short examples and reproduce SFT label/block handling."""
    dataset_dict = dllm.data.load_sft_dataset(args.dataset)
    if args.split not in dataset_dict:
        raise KeyError(
            f"Split {args.split!r} is absent; available={list(dataset_dict)}"
        )
    dataset = dataset_dict[args.split]
    sequences = []
    selected_indices = []
    original_lengths = []
    evaluated_lengths = []
    predicted_tokens = []

    for dataset_index, row in enumerate(dataset):
        mapped = default_sft_map_fn(
            row,
            tokenizer=tokenizer,
            mask_prompt_loss=True,
        )
        input_ids = list(mapped["input_ids"])
        labels = list(mapped["labels"])
        if len(input_ids) > args.max_tokens:
            continue
        response_tokens = sum(label != -100 for label in labels)
        if response_tokens < args.min_predicted_tokens:
            continue
        original_lengths.append(len(input_ids))
        padding = (-len(input_ids)) % args.block_size
        if padding:
            input_ids.extend([tokenizer.eos_token_id] * padding)
            labels.extend([tokenizer.eos_token_id] * padding)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        maskable_mask = torch.tensor(labels, dtype=torch.long) != -100
        if not maskable_mask.any():
            continue
        sequences.append(
            {
                "input_ids": input_tensor,
                "maskable_mask": maskable_mask,
            }
        )
        selected_indices.append(dataset_index)
        evaluated_lengths.append(len(input_ids))
        predicted_tokens.append(int(maskable_mask.sum().item()))
        if len(sequences) == args.num_sequences:
            break

    if len(sequences) != args.num_sequences:
        raise ValueError(
            f"Requested {args.num_sequences} complete sequences no longer than "
            f"{args.max_tokens} tokens, but found {len(sequences)}"
        )
    return sequences, {
        "dataset": args.dataset,
        "split": args.split,
        "dataset_rows": len(dataset),
        "selection": (
            "first rows whose complete chat-formatted prompt plus first response "
            "fits max_tokens and has at least min_predicted_tokens response labels"
        ),
        "selected_dataset_indices": selected_indices,
        "num_sequences": len(sequences),
        "max_tokens": args.max_tokens,
        "min_predicted_tokens": args.min_predicted_tokens,
        "truncated_sequences": 0,
        "original_token_lengths": original_lengths,
        "evaluated_sequence_lengths": evaluated_lengths,
        "predicted_response_tokens_per_sequence": predicted_tokens,
        "evaluated_tokens_per_seed": sum(predicted_tokens),
        "prompt_tokens_excluded": True,
        "first_response_selected": True,
        "eos_block_padding": True,
    }


def main() -> None:
    """Load fixed training examples, cross-score them, and write a report."""
    args = parse_args()
    validate_args(args)
    tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=args.baseline_checkpoint
    )
    if tokenizer.mask_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Checkpoint tokenizer must define mask and EOS tokens")
    sequences, dataset_metadata = load_training_sequences(args, tokenizer)
    corpora = {"fixed_training_sequences": sequences}
    evaluation_args = SimpleNamespace(
        seeds=args.seeds,
        time_epsilon=args.time_epsilon,
        block_size=args.block_size,
        device=args.device,
    )
    models = {
        "baseline": evaluate_model(
            args.baseline_checkpoint,
            corpora,
            tokenizer,
            evaluation_args,
            expected_loophole_enabled=False,
        ),
        "loophole": evaluate_model(
            args.loophole_checkpoint,
            corpora,
            tokenizer,
            evaluation_args,
            expected_loophole_enabled=True,
        ),
    }
    report = {
        "metric": {
            "name": "BD3LM diffusion pseudo-perplexity",
            "definition": (
                "exp(mean scheduler-weighted masked-token NLL over response "
                "labels), using training two-stream block attention and "
                "right-shifted logits"
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
        "dataset": dataset_metadata,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
