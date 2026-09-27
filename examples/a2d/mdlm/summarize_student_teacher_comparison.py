"""Summarize student-versus-teacher throughput and diffusion PPL, plus GSM8K if present.

Run with:
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/summarize_student_teacher_comparison.py --results-dir /nvme-data/atharvchagi/qwen3_0.6b_student_teacher_comparison``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_NAMES = ("distilled_student", "loophole_teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def throughput_row(report: dict[str, Any], name: str) -> dict[str, float]:
    model = report["models"][name]
    modes = [model[key] for key in ("inference_off", "inference_on") if key in model]
    if len(modes) != 1:
        raise ValueError(f"Expected one inference mode for {name}, found {len(modes)}")
    return modes[0]


def gsm8k_exact_match(report: dict[str, Any]) -> float:
    metrics = report["results"]["gsm8k_cot"]
    keys = [key for key in metrics if key.split(",", 1)[0] == "exact_match"]
    if not keys:
        raise KeyError(f"No exact_match metric in {sorted(metrics)}")
    return float(metrics[keys[0]])


def main() -> None:
    args = parse_args()
    throughput = read_json(args.results_dir / "throughput.json")
    perplexity = read_json(args.results_dir / "perplexity.json")
    gsm8k_paths = {
        "distilled_student": args.results_dir / "gsm8k_student.json",
        "loophole_teacher": args.results_dir / "gsm8k_teacher.json",
    }
    gsm8k_available = all(path.is_file() for path in gsm8k_paths.values())
    gsm8k = (
        {name: read_json(path) for name, path in gsm8k_paths.items()}
        if gsm8k_available
        else {}
    )

    rows = {}
    for name in MODEL_NAMES:
        speed = throughput_row(throughput, name)
        ppl = perplexity["models"][name]
        rows[name] = {
            "threshold_generated_tokens_per_second": speed[
                "median_generated_tokens_per_second"
            ],
            "threshold_p50_seconds": speed["p50_seconds"],
            "threshold_p95_seconds": speed["p95_seconds"],
            "gsm8k_exact_match": (
                gsm8k_exact_match(gsm8k[name]) if gsm8k_available else None
            ),
            "diffusion_perplexity": ppl["diffusion_perplexity"],
            "diffusion_nll": ppl["diffusion_nll"],
            "masked_token_accuracy": ppl["masked_token_accuracy"],
        }

    teacher = rows["loophole_teacher"]
    student = rows["distilled_student"]
    deltas = {
        "student_throughput_delta_percent": 100.0
        * (
            student["threshold_generated_tokens_per_second"]
            / teacher["threshold_generated_tokens_per_second"]
            - 1.0
        ),
        "student_gsm8k_delta_points": (
            100.0
            * (student["gsm8k_exact_match"] - teacher["gsm8k_exact_match"])
            if gsm8k_available
            else None
        ),
        "student_perplexity_delta_percent": 100.0
        * (student["diffusion_perplexity"] / teacher["diffusion_perplexity"] - 1.0),
    }
    summary = {
        "protocol": {
            "inference_decoder": "confidence threshold with scheduler completion floor",
            "confidence_threshold": 0.85,
            "gsm8k_task": "gsm8k_cot",
            "perplexity": "MATH-500 solution-trace MDLM diffusion perplexity",
        },
        "models": rows,
        "student_minus_teacher": deltas,
    }
    json_path = args.results_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Distilled student vs. Loophole teacher",
        "",
        "All generation uses confidence-threshold decoding at 0.85.",
        "",
        "| Model | Generated tok/s | GSM8K exact match | Diffusion PPL | Masked-token accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        row = rows[name]
        gsm8k_cell = (
            f"{row['gsm8k_exact_match']:.4%}" if gsm8k_available else "not run"
        )
        lines.append(
            f"| {name} | {row['threshold_generated_tokens_per_second']:.3f} "
            f"| {gsm8k_cell} "
            f"| {row['diffusion_perplexity']:.6f} "
            f"| {row['masked_token_accuracy']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Student minus teacher",
            "",
            f"- Throughput: {deltas['student_throughput_delta_percent']:+.2f}%",
            (
                f"- GSM8K: {deltas['student_gsm8k_delta_points']:+.2f} percentage points"
                if gsm8k_available
                else "- GSM8K: not run"
            ),
            f"- Diffusion PPL: {deltas['student_perplexity_delta_percent']:+.2f}%",
            "",
        ]
    )
    markdown_path = args.results_dir / "comparison_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path.resolve()}")
    print(f"Wrote {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
