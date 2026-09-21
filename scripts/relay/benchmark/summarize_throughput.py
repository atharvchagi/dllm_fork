"""Rebuild benchmark and full-evaluation summaries from saved files, without GPUs.

Run in the dllm environment:
python /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RELAY_RESULTS = ROOT / "results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k"
BASELINE_RESULTS = ROOT / "results/qwen_0.6b_bd3lm_block4_baseline_gsm8k/throughput_random32"
AR_RESULTS = ROOT / "results/qwen_0.6b_ar_gsm8k/throughput_random32"


def model_args_from_result(result):
    args = result["config"]["model_args"]
    if isinstance(args, str):
        from lm_eval.utils import simple_parse_args_string

        return simple_parse_args_string(args)
    return args


def summarize_full_eval():
    for epochs in (5, 8):
        folder = RELAY_RESULTS / f"{epochs}ep/full_eval"
        generation_runs = []
        for decoding in ("static", "dynamic"):
            path = folder / decoding / "throughput_summary.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            result_path = Path(summary["result_file"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            args = model_args_from_result(result)
            count = summary["evaluated_samples"]
            seconds = summary["total_evaluation_time_seconds"]
            tokens = summary["generated_tokens"]
            num_gpus = summary.get("num_gpus")
            # The original static summary explicitly records single-GPU timing.
            if num_gpus is None and summary["throughput_scope"] == "single-GPU end-to-end lm-eval wall time":
                num_gpus = 1
            accuracy = summary["metrics"]["exact_match,flexible-extract"]
            generation_runs.append(
                {
                    "decoding": decoding,
                    "evaluated_samples": count,
                    "num_gpus": num_gpus,
                    "batch_size_per_gpu": int(result["config"]["batch_size"]),
                    "block_size": args["block_size"],
                    "max_new_tokens": args.get("max_new_tokens"),
                    "max_total_tokens": args.get("max_length") if not args.get("max_new_tokens") else None,
                    "confidence_threshold": args.get("relay_unmask_threshold"),
                    "correct_answers": round(accuracy * count),
                    "accuracy_flexible_extract": accuracy,
                    "accuracy_strict_match": summary["metrics"]["exact_match,strict-match"],
                    "generated_tokens": tokens,
                    "mean_generated_tokens_per_sample": tokens / count,
                    "total_evaluation_time_seconds": seconds,
                    "samples_per_second": count / seconds,
                    "aggregate_output_tokens_per_second": tokens / seconds,
                    "normalized_output_tokens_per_second_per_gpu": tokens / seconds / num_gpus if num_gpus else None,
                    "mean_active_denoising_nfe": summary.get("mean_active_denoising_nfe"),
                    "result_file": str(result_path),
                    "source_summary": str(path),
                }
            )
        loss_path = folder / "perplexity/eval_results.json"
        loss_evaluation = None
        if loss_path.is_file():
            loss_evaluation = {
                "definition": "Teacher-forced masked-token loss evaluation; perplexity is exp(masked-token NLL), not AR sequence perplexity. eval_acc is token accuracy, not GSM8K answer accuracy.",
                "metrics": json.loads(loss_path.read_text(encoding="utf-8")),
                "source_file": str(loss_path),
            }
        if not generation_runs and loss_evaluation is None:
            continue
        output = folder / "summary.json"
        data = {
            "model": "qwen_0.6b_bd3lm_block4_k2",
            "model_epochs": epochs,
            "dataset": "GSM8K test",
            "scope": "Full-dataset evaluations only; excludes 32-sample benchmarks. Latest saved summary per decoding policy.",
            "throughput_definition": "Saved, stop-trimmed output text re-tokenized, divided by end-to-end lm-eval runtime including setup and scoring.",
            "comparison_note": "Static and dynamic runs may differ in GPU count, batch size, and token budget. Per-GPU throughput is an aggregate divided by GPU count, not a separate single-GPU measurement.",
            "generation_runs": generation_runs,
            "loss_evaluation": loss_evaluation,
        }
        output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote full evaluation summary to {output}")


def main():
    rows = []
    selected_ids = None
    experiments = [
        ("bd3lm-relay-block4-k2", epochs, RELAY_RESULTS / f"{epochs}ep/throughput_random32")
        for epochs in (5, 8)
    ]
    experiments.append(("bd3lm-baseline-block4", "", BASELINE_RESULTS))
    experiments.append(("Qwen3-0.6B-AR", "", AR_RESULTS))
    for model, epochs, folder in experiments:
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
            args = model_args_from_result(result)
            accuracy = summary["metrics"]["exact_match,flexible-extract"]
            rows.append(
                {
                    "model": model,
                    "model_epochs": epochs,
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
    output = RELAY_RESULTS / "throughput_analysis/comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {output}")
    summarize_full_eval()


if __name__ == "__main__":
    main()
