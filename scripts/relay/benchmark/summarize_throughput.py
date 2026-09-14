"""Rebuild the 32-sample throughput comparison from saved summaries, without GPUs.

Run in the dllm environment:
python /nvme-data/neeleshgarg/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RELAY_RESULTS = ROOT / "results/qwen_0.6b_bd3lm_block4_k2_gsm8k/throughput_random32"
AR_RESULTS = ROOT / "results/qwen_0.6b_ar_gsm8k/throughput_random32"


def main():
    rows = []
    selected_ids = None
    for model, folder in [("bd3lm-relay-block4-k2", RELAY_RESULTS), ("Qwen3-0.6B-AR", AR_RESULTS)]:
        for path in sorted(folder.rglob("throughput_summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            ids = sorted(summary["sample_doc_ids"])
            if selected_ids is None:
                selected_ids = ids
            if ids != selected_ids or len(set(ids)) != 32 or summary["evaluated_samples"] != 32:
                raise ValueError(f"Expected the same 32 document IDs in {path}")
            result_path = Path(summary["result_file"])
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            args = result["config"]["model_args"]
            accuracy = summary["metrics"]["exact_match,flexible-extract"]
            rows.append(
                {
                    "model": model,
                    "run_id": path.parent.name,
                    "decoding": summary["decoding"],
                    "max_sequence_length": summary["max_sequence_length"],
                    "length_scope": "prompt_plus_output",
                    "num_gpus": summary["num_gpus"],
                    "batch_size_per_gpu": summary["batch_size_per_gpu"],
                    "evaluated_samples": summary["evaluated_samples"],
                    "generated_tokens": summary["generated_tokens"],
                    "mean_generated_tokens_per_sample": summary["mean_generated_tokens_per_sample"],
                    "total_evaluation_time_seconds": summary["total_evaluation_time_seconds"],
                    "output_tokens_per_second": summary["output_tokens_per_second"],
                    "samples_per_second": summary["samples_per_second"],
                    "correct_answers": round(accuracy * summary["evaluated_samples"]),
                    "accuracy_flexible_extract": accuracy,
                    "mean_active_denoising_nfe": summary.get("mean_active_denoising_nfe", ""),
                    "explicit_steps_argument": args.get("steps", "") if isinstance(args, dict) else "",
                    "throughput_scope": summary["throughput_scope"],
                    "token_count_scope": summary["token_count_scope"],
                    "notes": summary.get("notes", ""),
                    "summary_file": str(path.resolve()),
                    "result_file": str(result_path.resolve()),
                }
            )
    if not rows:
        raise FileNotFoundError("No throughput summaries found")
    output = RELAY_RESULTS / "comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
