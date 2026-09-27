"""Summarize a completed GSM8K baseline evaluation.

Run through the evaluation launcher:
python /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/summarize_eval.py RESULT_DIR CHECKPOINT BLOCK_SIZE NUM_GPUS BATCH_SIZE GPU_UUIDS
"""

import json
import os
import sys
from pathlib import Path

import dllm
import wandb


def main() -> None:
    """Write accuracy and end-to-end throughput metrics for one evaluation."""
    result_dir = Path(sys.argv[1])
    checkpoint = Path(sys.argv[2])
    block_size = int(sys.argv[3])
    num_gpus = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpu_uuids = sys.argv[6].split(",")
    if len(gpu_uuids) != num_gpus:
        raise ValueError(f"Expected {num_gpus} GPU UUIDs, received {len(gpu_uuids)}")

    result_files = list(result_dir.rglob("results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No lm-eval result JSON found under {result_dir}")
    result_file = max(result_files, key=lambda path: path.stat().st_mtime)
    result = json.loads(result_file.read_text(encoding="utf-8"))

    task = "gsm8k_cot"
    seconds = float(result["total_evaluation_time_seconds"])
    sample_count = int(result["n-samples"][task]["effective"])

    generated_tokens = None
    sample_files = list(result_dir.rglob(f"samples_{task}_*.jsonl"))
    if sample_files:
        sample_file = max(sample_files, key=lambda path: path.stat().st_mtime)
        tokenizer = dllm.utils.get_tokenizer(model_name_or_path=str(checkpoint))
        responses_by_doc = {}
        with sample_file.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                responses_by_doc.setdefault(sample["doc_id"], sample["resps"][0][0])
        generated_tokens = sum(
            len(tokenizer(response, add_special_tokens=False)["input_ids"])
            for response in responses_by_doc.values()
        )

    aggregate_sample_rate = sample_count / seconds
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "result_file": str(result_file.resolve()),
        "metrics": result["results"][task],
        "total_evaluation_time_seconds": seconds,
        "evaluated_samples": sample_count,
        "aggregate_samples_per_second": aggregate_sample_rate,
        "normalized_samples_per_second_per_gpu": aggregate_sample_rate / num_gpus,
        "generated_tokens": generated_tokens,
        "mean_generated_tokens_per_sample": (
            generated_tokens / sample_count if generated_tokens is not None else None
        ),
        "aggregate_output_tokens_per_second": (
            generated_tokens / seconds if generated_tokens is not None else None
        ),
        "normalized_output_tokens_per_second_per_gpu": (
            generated_tokens / seconds / num_gpus
            if generated_tokens is not None
            else None
        ),
        "num_gpus": num_gpus,
        "gpu_uuids": gpu_uuids,
        "batch_size_per_gpu": batch_size,
        "global_concurrent_batch_size": num_gpus * batch_size,
        "block_size": block_size,
        "max_new_tokens": 256,
        "denoising_steps": 256,
        "decoding": "static deterministic",
        "throughput_scope": (
            "four-GPU end-to-end lm-eval wall time, including model loading, "
            "generation, and scoring"
        ),
    }
    summary_path = result_dir / "throughput_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path.resolve()}")

    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if wandb_mode != "disabled":
        wandb_metrics = {
            f"{task}/{key}": value
            for key, value in result["results"][task].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        wandb_metrics.update(
            {
                "throughput/total_evaluation_time_seconds": seconds,
                "throughput/aggregate_samples_per_second": aggregate_sample_rate,
                "throughput/normalized_samples_per_second_per_gpu": (
                    aggregate_sample_rate / num_gpus
                ),
                "throughput/aggregate_output_tokens_per_second": summary[
                    "aggregate_output_tokens_per_second"
                ],
                "throughput/normalized_output_tokens_per_second_per_gpu": summary[
                    "normalized_output_tokens_per_second_per_gpu"
                ],
            }
        )
        with wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gsm8k-bd3lm-baseline"),
            name=(
                f"qwen3-0.6b-gsm8k-bd3lm-baseline-b{block_size}"
                "-eval-batch1-4gpu"
            ),
            job_type="eval",
            mode=wandb_mode,
            config={
                "checkpoint": str(checkpoint.resolve()),
                "block_size": block_size,
                "max_new_tokens": 256,
                "denoising_steps": 256,
                "num_gpus": num_gpus,
                "gpu_uuids": gpu_uuids,
                "batch_size_per_gpu": batch_size,
                "global_concurrent_batch_size": num_gpus * batch_size,
            },
        ) as run:
            run.log(wandb_metrics)
            run.save(str(summary_path.resolve()), base_path=str(result_dir.resolve()))


if __name__ == "__main__":
    main()
