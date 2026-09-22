"""Generate exact-token AR verifier responses for the BD3LM agreement analysis.

Run on physical GPUs 0-3 after activating the ``dllm`` environment:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/generate_ar_verifier_responses.py \
  --selection-report /home/atharvchagi/qwen3_0.6b_bd3lm_speculative_agreement/multi-128samples-16blocks.json \
  --output /nvme-data2/atharvchagi/dllm_fork/results/qwen3_0.6b_bd3lm_speculative_agreement/ar-generated-128samples/samples_ar_verifier.jsonl
```

The JSONL retains raw prompt/response token IDs so downstream evaluation never
needs to decode and re-tokenize the verifier's greedy continuation.
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
import transformers

import analyze_speculative_top1_agreement as single


DEFAULT_SELECTION_REPORT = Path(
    "/home/atharvchagi/qwen3_0.6b_bd3lm_speculative_agreement/"
    "multi-128samples-16blocks.json"
)
DEFAULT_OUTPUT = Path(
    "/nvme-data2/atharvchagi/dllm_fork/results/"
    "qwen3_0.6b_bd3lm_speculative_agreement/ar-generated-128samples/"
    "samples_ar_verifier.jsonl"
)
STOP_STRINGS = ["\nQ:", "\nQuestion:", "</s>", "<|im_end|>"]


def parse_args() -> argparse.Namespace:
    """Parse AR response generation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=single.DEFAULT_VERIFIER)
    parser.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-style", choices=("qa", "source_chat"), default="qa")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_selected_rows(
    selection_report: Path, num_samples: int
) -> tuple[list[dict[str, Any]], Path]:
    """Reuse the exact document selection and prompts from a prior analysis."""
    with selection_report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    selected_ids = [
        int(sample["doc_id"])
        for sample in report["samples"]["selected"][:num_samples]
    ]
    if len(selected_ids) != num_samples or len(set(selected_ids)) != num_samples:
        raise ValueError("Selection report does not contain enough distinct samples")
    source_path = Path(report["samples"]["source_path"])
    selected_set = set(selected_ids)
    rows_by_id = {}
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = int(row["doc_id"])
            if row.get("filter") == "flexible-extract" and doc_id in selected_set:
                if doc_id in rows_by_id:
                    raise ValueError(f"Duplicate source document {doc_id}")
                rows_by_id[doc_id] = row
    missing = selected_set - set(rows_by_id)
    if missing:
        raise ValueError(f"Missing selected source documents: {sorted(missing)}")
    return [rows_by_id[doc_id] for doc_id in selected_ids], source_path


def trim_continuation(
    tokenizer,
    generated_ids: list[int],
    terminal_ids: set[int],
) -> tuple[list[int], str]:
    """Trim EOS/stop markers while retaining an exact generated token prefix."""
    terminal_index = next(
        (
            index
            for index, token_id in enumerate(generated_ids)
            if token_id in terminal_ids
        ),
        len(generated_ids),
    )
    content_ids = generated_ids[:terminal_index]
    reason = "terminal_token" if terminal_index < len(generated_ids) else "max_new_tokens"
    text = tokenizer.decode(
        content_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    matches = [
        (text.find(marker), marker)
        for marker in STOP_STRINGS
        if marker in text
    ]
    if matches:
        boundary, marker = min(matches)
        # A stop string may begin inside a token. Drop that whole token rather
        # than manufacturing a different token sequence by re-tokenizing text.
        safe_count = 0
        for count in range(1, len(content_ids) + 1):
            prefix = tokenizer.decode(
                content_ids[:count],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if len(prefix) > boundary:
                break
            safe_count = count
        content_ids = content_ids[:safe_count]
        reason = f"stop_string:{marker}"
    return content_ids, reason


def render_prompt(row: dict[str, Any], prompt_style: str) -> str:
    """Use plain QA for the AR base model or explicitly retain source formatting."""
    if prompt_style == "source_chat":
        return row["arguments"]["gen_args_0"]["arg_0"]
    question = row.get("doc", {}).get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing GSM8K question for doc_id={row['doc_id']}")
    return f"Q: {question.strip()}\nA:"


def repetition_flags(token_ids: list[int]) -> list[str]:
    """Flag degeneration for inspection without filtering generated responses."""
    if len(token_ids) < 16:
        return []
    counts: dict[int, int] = {}
    for token_id in token_ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    flags = []
    if max(counts.values()) / len(token_ids) > 0.5:
        flags.append("dominant_token")
    if len(set(zip(token_ids, token_ids[1:]))) / (len(token_ids) - 1) < 0.1:
        flags.append("low_distinct_2")
    return flags


@torch.inference_mode()
def generate_rows(
    rows: list[dict[str, Any]],
    *,
    model,
    tokenizer,
    device: torch.device,
    rank: int,
    batch_size: int,
    max_new_tokens: int,
    prompt_style: str,
    generator_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate a rank's selected responses in left-padded greedy batches."""
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    terminal_ids = {int(token_id) for token_id in eos_ids if token_id is not None}
    if tokenizer.eos_token_id is not None:
        terminal_ids.add(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        terminal_ids.add(int(vocab["<|im_end|>"]))
    output_rows = []
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompts = [render_prompt(row, prompt_style) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(device)
        prompt_width = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            use_cache=True,
            eos_token_id=sorted(terminal_ids),
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=STOP_STRINGS,
            tokenizer=tokenizer,
        )
        for index, row in enumerate(batch_rows):
            prompt_ids = encoded["input_ids"][index][
                encoded["attention_mask"][index].bool()
            ].cpu().tolist()
            raw_ids = generated[index, prompt_width:].cpu().tolist()
            response_ids, stop_reason = trim_continuation(
                tokenizer, raw_ids, terminal_ids
            )
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            output_rows.append(
                {
                    "doc_id": int(row["doc_id"]),
                    "doc": row.get("doc", {}),
                    "target": row.get("target"),
                    "arguments": {
                        "gen_args_0": {
                            "arg_0": prompts[index],
                            "arg_1": {"do_sample": False, "until": STOP_STRINGS},
                        }
                    },
                    "resps": [[response]],
                    "filter": "flexible-extract",
                    "tokenized_sample": {
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                    },
                    "generation": {
                        **generator_metadata,
                        "rank": rank,
                        "response_token_count": len(response_ids),
                        "stop_reason": stop_reason,
                        "quality_flags": repetition_flags(response_ids),
                        "token_ids_sha256": single.stable_json_hash(
                            prompt_ids + response_ids
                        ),
                        "text_sha256": hashlib.sha256(
                            (prompts[index] + response).encode("utf-8")
                        ).hexdigest(),
                    },
                }
            )
        print(
            f"rank={rank} generated={min(batch_start + batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
    return output_rows


def main() -> None:
    """Generate distributed responses and write a merged JSONL plus provenance."""
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Sample count, batch size, and token limit must be positive")
    if not args.selection_report.is_absolute() or not args.selection_report.is_file():
        raise ValueError("--selection-report must be an existing absolute path")
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute path")
    rank, world_size, device = single.initialize_distributed("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    started_at = time.perf_counter()
    selected_rows, source_path = load_selected_rows(
        args.selection_report, args.num_samples
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, config = single.load_verifier_model(
        args.checkpoint,
        dtype=getattr(torch, args.dtype),
        device=device,
        attn_implementation=args.attn_implementation,
    )
    generator_metadata = {
        "origin": "ar_verifier_generated",
        "checkpoint": args.checkpoint,
        "revision": getattr(config, "_commit_hash", None),
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "stop_strings": STOP_STRINGS,
        "use_cache": True,
        "prompt_style": args.prompt_style,
        "seed": args.seed,
        "tokenizer_fingerprint": single.tokenizer_fingerprint(tokenizer),
    }
    if rank == 0:
        print(
            f"world_size={world_size} samples={len(selected_rows)} "
            f"batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}",
            flush=True,
        )
    local_rows = generate_rows(
        selected_rows[rank::world_size],
        model=model,
        tokenizer=tokenizer,
        device=device,
        rank=rank,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        generator_metadata=generator_metadata,
    )
    local_payload = {
        "rank": rank,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "sample_count": len(local_rows),
        "runtime_seconds": time.perf_counter() - started_at,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "rows": local_rows,
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
        payloads = [payload for payload in gathered if payload is not None]
        rows_by_id = {
            row["doc_id"]: row
            for payload in payloads
            for row in payload["rows"]
        }
        ordered_rows = [rows_by_id[int(row["doc_id"])] for row in selected_rows]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_rows),
            encoding="utf-8",
        )
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": generator_metadata,
            "model": single.model_metadata(args.checkpoint, model, config),
            "selection_report": str(args.selection_report.resolve()),
            "prompt_source_path": str(source_path.resolve()),
            "same_questions_as_selection_report": True,
            "same_prompts_as_selection_report": args.prompt_style == "source_chat",
            "samples_path": str(args.output.resolve()),
            "sample_count": len(ordered_rows),
            "selected_doc_ids": [row["doc_id"] for row in ordered_rows],
            "response_token_counts": [
                len(row["tokenized_sample"]["response_token_ids"])
                for row in ordered_rows
            ],
            "quality_flagged_doc_ids": [
                row["doc_id"]
                for row in ordered_rows
                if row["generation"]["quality_flags"]
            ],
            "execution": {
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "batch_size": args.batch_size,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "slurm_used": False,
                "slurm_note": "srun is unavailable on this host",
                "workers": [
                    {key: value for key, value in payload.items() if key != "rows"}
                    for payload in payloads
                ],
            },
        }
        metadata_path = args.output.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output.resolve()}", flush=True)
        print(f"Wrote {metadata_path.resolve()}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
