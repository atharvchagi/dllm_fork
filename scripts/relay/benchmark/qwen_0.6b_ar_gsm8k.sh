#!/usr/bin/env bash

# Run: bash /nvme-data/neeleshgarg/dllm_fork/scripts/relay/benchmark/qwen_0.6b_ar_gsm8k.sh

set -euo pipefail

source /home/ngarg2/miniforge3/etc/profile.d/conda.sh
conda activate /nvme-data/neeleshgarg/envs/dllm

cd /nvme-data/neeleshgarg/dllm_fork
# Use the same local lm-eval version as the Relay benchmark.
export PYTHONPATH="/nvme-data/neeleshgarg/dllm_fork/lm-evaluation-harness${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled

checkpoint=/nvme-data/neeleshgarg/dllm_fork/.models/Qwen/Qwen3-0.6B
result_dir=/nvme-data/neeleshgarg/dllm_fork/results/qwen_0.6b_ar_gsm8k/throughput_random32/len2048/$(date +%Y%m%d_%H%M%S)
relay_results=/nvme-data/neeleshgarg/dllm_fork/results/qwen_0.6b_bd3lm_block4_k2_gsm8k/throughput_random32/dynamic/len2048
sample_selection='{"gsm8k_cot":[13,51,54,61,65,178,191,209,228,285,318,326,407,447,451,457,476,501,563,569,696,859,864,865,919,1034,1116,1149,1206,1209,1232,1309]}'

mkdir -p "${result_dir}"

# Fetch weights before lm-eval starts its timer. Reuse the local download on reruns.
python - "${checkpoint}" <<'PY'
"""Download the AR baseline; run through the accompanying benchmark shell script."""

import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=sys.argv[1],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
)
PY

# In this lm-eval version, max_gen_toks controls reserved context space.
# Zero preserves the prompt; explicit max_length sets the actual generation
# limit to 2048 tokens INCLUDING the prompt (it does not mean zero output).
# Standard HF generation enables use_cache=True. No thinking, greedy, BF16.
CUDA_VISIBLE_DEVICES=7 python -m lm_eval \
  --tasks gsm8k_cot \
  --model hf \
  --device cuda \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size 1 \
  --samples "${sample_selection}" \
  --model_args "pretrained=${checkpoint},dtype=bfloat16,max_length=2048,enable_thinking=False,attn_implementation=sdpa" \
  --gen_kwargs "do_sample=False,max_gen_toks=0,max_length=2048" \
  --output_path "${result_dir}" \
  --log_samples 2>&1 | tee "${result_dir}/eval.log"

python - "${result_dir}" "${checkpoint}" "${sample_selection}" "${relay_results}" <<'PY'
"""Summarize throughput; run through the accompanying benchmark shell script."""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


result_dir = Path(sys.argv[1])
checkpoint = sys.argv[2]
task = "gsm8k_cot"
selected_ids = json.loads(sys.argv[3])[task]
result_file = max(result_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
result = json.loads(result_file.read_text(encoding="utf-8"))
sample_file = max(
    result_dir.rglob(f"samples_{task}_*.jsonl"), key=lambda p: p.stat().st_mtime
)
responses_by_doc = {}
with sample_file.open(encoding="utf-8") as handle:
    for line in handle:
        sample = json.loads(line)
        # Each output filter can have a row; count each document only once.
        responses_by_doc.setdefault(sample["doc_id"], sample["resps"][0][0])
if set(responses_by_doc) != set(selected_ids):
    raise ValueError("Evaluated document IDs do not match the selected 32 samples")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokens = sum(
    len(tokenizer(response, add_special_tokens=False)["input_ids"])
    for response in responses_by_doc.values()
)
seconds = float(result["total_evaluation_time_seconds"])
sample_count = len(responses_by_doc)
summary = {
    "model": "Qwen/Qwen3-0.6B",
    "result_file": str(result_file.resolve()),
    "sample_file": str(sample_file.resolve()),
    "metrics": result["results"][task],
    "total_evaluation_time_seconds": seconds,
    "evaluated_samples": sample_count,
    "generated_tokens": tokens,
    "mean_generated_tokens_per_sample": tokens / sample_count,
    "samples_per_second": sample_count / seconds,
    "output_tokens_per_second": tokens / seconds,
    "output_tokens_per_second_per_gpu": tokens / seconds,
    "throughput_scope": "end-to-end lm-eval, including model loading and scoring; excluding model download",
    "token_count_scope": "decoded, stop-trimmed outputs re-tokenized, matching the Relay summary method",
    "num_gpus": 1,
    "gpu": 7,
    "batch_size_per_gpu": 1,
    "sample_doc_ids": selected_ids,
    "max_sequence_length": 2048,
    "decoding": "autoregressive greedy, KV cache enabled",
    "enable_thinking": False,
    "dtype": "bfloat16",
    "comparison_notes": [
        "HF stops on task stop strings during generation; Relay trims them afterward.",
        "Relay rounds prompts up to block size 4; its output budget can be up to 3 tokens smaller.",
        "Different checkpoints and response lengths: this compares evaluation paths, not isolated kernel speed.",
    ],
}
relay_paths = sorted(Path(sys.argv[4]).glob("*/throughput_summary.json"))
if relay_paths:
    relay_path = relay_paths[-1]
    relay = json.loads(relay_path.read_text(encoding="utf-8"))
    if set(relay["sample_doc_ids"]) == set(selected_ids):
        relay_rate = relay["output_tokens_per_second"]
        summary["relay_comparison"] = {
            "result_file": relay["result_file"],
            "output_tokens_per_second": relay_rate,
            "qwen_to_relay_throughput_ratio": (tokens / seconds) / relay_rate,
        }

summary_path = result_dir / "throughput_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Wrote {summary_path.resolve()}")
PY

python /nvme-data/neeleshgarg/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
