#!/usr/bin/env bash

# Run directly with: bash /nvme-data/neeleshgarg/dllm_fork/scripts/relay/benchmark/qwen_0.6b_bd3lm_block4_k2_gsm8k.sh
# Optional first argument: total prompt + output limit (default 2048).
# Example: append 512 to the command above for the 512-token benchmark.

set -euo pipefail

max_length=${1:-2048}
if [[ ! "${max_length}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Sequence length must be a positive integer." >&2
  exit 1
fi

source /home/ngarg2/miniforge3/etc/profile.d/conda.sh
conda activate /nvme-data/neeleshgarg/envs/dllm

cd /nvme-data/neeleshgarg/dllm_fork

checkpoint=/nvme-data/neeleshgarg/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init/bd3lm/relay_block4_k2
result_dir=/nvme-data/neeleshgarg/dllm_fork/results/qwen_0.6b_bd3lm_block4_k2_gsm8k/throughput_random32/dynamic/len${max_length}/$(date +%Y%m%d_%H%M%S)
log_file=${result_dir}/eval.log
nfe_file=${result_dir}/dynamic_nfe.jsonl
sample_selection='{"gsm8k_cot":[13,51,54,61,65,178,191,209,228,285,318,326,407,447,451,457,476,501,563,569,696,859,864,865,919,1034,1116,1149,1206,1209,1232,1309]}'

mkdir -p "${result_dir}"

CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled \
  accelerate launch \
  --num_processes 1 \
  /nvme-data/neeleshgarg/dllm_fork/dllm/pipelines/a2d/eval.py \
  --tasks gsm8k_cot \
  --model a2d_bd3lm \
  --device cuda \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size 1 \
  --samples "${sample_selection}" \
  --loophole_enabled \
  --model_args "pretrained=${checkpoint},max_new_tokens=0,max_length=${max_length},block_size=4,cfg_scale=0.0,temperature=0.0,right_shift_logits=True,relay_unmask_threshold=0.85,nfe_output_path=${nfe_file}" \
  --output_path "${result_dir}" \
  --log_samples 2>&1 | tee "${log_file}"

python - "${result_dir}" "${checkpoint}" "${sample_selection}" "${max_length}" <<'PY'
"""Summarize the completed one-GPU GSM8K throughput benchmark.

Run through /nvme-data/neeleshgarg/dllm_fork/scripts/relay/benchmark/qwen_0.6b_bd3lm_block4_k2_gsm8k.sh.
"""

import json
import statistics
import sys
from pathlib import Path

import dllm


result_dir = Path(sys.argv[1])
checkpoint = sys.argv[2]
sample_selection = json.loads(sys.argv[3])
result_file = max(
    result_dir.glob("results_*.json"),
    key=lambda path: path.stat().st_mtime,
)
result = json.loads(result_file.read_text(encoding="utf-8"))
task = "gsm8k_cot"
seconds = float(result["total_evaluation_time_seconds"])

sample_file = max(
    result_dir.glob(f"samples_{task}_*.jsonl"),
    key=lambda path: path.stat().st_mtime,
)
responses_by_doc = {}
with sample_file.open(encoding="utf-8") as handle:
    for line in handle:
        sample = json.loads(line)
        responses_by_doc.setdefault(sample["doc_id"], sample["resps"][0][0])
sample_count = len(responses_by_doc)
if set(responses_by_doc) != set(sample_selection[task]):
    raise ValueError("Evaluated document IDs do not match the selected 32 samples")

tokenizer = dllm.utils.get_tokenizer(model_name_or_path=checkpoint)
generated_tokens = sum(
    len(tokenizer(response, add_special_tokens=False)["input_ids"])
    for response in responses_by_doc.values()
)

nfe_by_prompt = {}
for nfe_path in result_dir.glob("dynamic_nfe.rank*.jsonl"):
    with nfe_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            nfe_by_prompt.setdefault(record["prompt_sha256"], int(record["nfe"]))
nfe_values = list(nfe_by_prompt.values())

summary = {
    "result_file": str(result_file.resolve()),
    "metrics": result["results"][task],
    "total_evaluation_time_seconds": seconds,
    "evaluated_samples": sample_count,
    "generated_tokens": generated_tokens,
    "mean_generated_tokens_per_sample": generated_tokens / sample_count,
    "samples_per_second": sample_count / seconds,
    "output_tokens_per_second": generated_tokens / seconds,
    "output_tokens_per_second_per_gpu": generated_tokens / seconds,
    "num_gpus": 1,
    "batch_size_per_gpu": 1,
    "sample_selection": "32 random GSM8K document IDs generated with seed 42",
    "sample_doc_ids": sample_selection[task],
    "max_sequence_length": int(sys.argv[4]),
    "throughput_scope": "end-to-end lm-eval, including model loading and scoring",
    "token_count_scope": "decoded, stop-trimmed outputs re-tokenized",
    "decoding": "dynamic confidence-threshold",
    "dynamic_unmask_threshold": 0.85,
}
if nfe_values:
    summary.update(
        {
            "nfe_samples": len(nfe_values),
            "mean_active_denoising_nfe": statistics.fmean(nfe_values),
            "median_active_denoising_nfe": statistics.median(nfe_values),
            "min_active_denoising_nfe": min(nfe_values),
            "max_active_denoising_nfe": max(nfe_values),
        }
    )

summary_path = result_dir / "throughput_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Wrote {summary_path.resolve()}")
PY

python /nvme-data/neeleshgarg/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
