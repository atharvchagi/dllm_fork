#!/usr/bin/env bash

# Run directly with: bash /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/eval/qwen_0.6b_bd3lm_block4_k2_gsm8k.sh
# Optional first argument: model epochs, 5 or 8 (default 8).

set -euo pipefail

model_epochs=${1:-8}
if [[ "${model_epochs}" != 5 && "${model_epochs}" != 8 ]]; then
  echo "Model epochs must be 5 or 8." >&2
  exit 1
fi

cd /nvme-data/neeleshgarg/repos/dllm_fork

export PYTHONPATH=/nvme-data/neeleshgarg/repos/dllm_fork
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

checkpoint=/nvme-data/neeleshgarg/repos/dllm_fork/.models/Qwen/Qwen3-0.6b-a2d-init/bd3lm/relay_block4_k2_${model_epochs}ep
result_dir=/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/${model_epochs}ep/full_eval/dynamic
log_file=${result_dir}/eval.log
nfe_file=${result_dir}/dynamic_nfe.jsonl

if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
  echo "Missing trained checkpoint: ${checkpoint}" >&2
  exit 1
fi
mkdir -p "${result_dir}"
rm -f "${result_dir}"/dynamic_nfe.rank*.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online \
  /nvme-data/neeleshgarg/envs/dllm/bin/python \
  -m accelerate.commands.launch \
  --num_processes 8 \
  /nvme-data/neeleshgarg/repos/dllm_fork/dllm/pipelines/a2d/eval.py \
  --tasks gsm8k_cot \
  --model a2d_bd3lm \
  --device cuda \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size 4 \
  --loophole_enabled \
  --model_args "pretrained=${checkpoint},max_new_tokens=0,max_length=2048,block_size=4,cfg_scale=0.0,temperature=0.0,right_shift_logits=True,relay_unmask_threshold=0.85,nfe_output_path=${nfe_file}" \
  --output_path "${result_dir}" \
  --wandb_args "project=block-relay,name=bd3lm-relay-block4-k2-${model_epochs}ep-gsm8k-dynamic-2048-eval,job_type=eval" \
  --wandb_config_args "block_size=4,relay_steps=2,max_length=2048,decoding=dynamic,relay_unmask_threshold=0.85,num_gpus=8,model_epochs=${model_epochs}" \
  --log_samples 2>&1 | tee "${log_file}"

/nvme-data/neeleshgarg/envs/dllm/bin/python \
  - "${result_dir}" "${checkpoint}" "${model_epochs}" <<'PY'
"""Summarize GSM8K accuracy and end-to-end generation throughput.

Run through /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/eval/qwen_0.6b_bd3lm_block4_k2_gsm8k.sh after lm-eval finishes.
"""

import json
import statistics
import sys
from pathlib import Path

import dllm


result_dir = Path(sys.argv[1])
checkpoint = sys.argv[2]
result_files = list(result_dir.rglob("results_*.json"))
if not result_files:
    raise FileNotFoundError(f"No lm-eval result JSON found under {result_dir}")
result_file = max(result_files, key=lambda path: path.stat().st_mtime)
result = json.loads(result_file.read_text(encoding="utf-8"))

task = "gsm8k_cot"
seconds = float(result["total_evaluation_time_seconds"])
sample_count = int(result["n-samples"][task]["effective"])

sample_files = list(result_dir.rglob(f"samples_{task}_*.jsonl"))
generated_tokens = None
if sample_files:
    sample_file = max(sample_files, key=lambda path: path.stat().st_mtime)
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=checkpoint)
    responses_by_doc = {}
    with sample_file.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            # lm-eval writes one row per output filter, so count each generated
            # response once and tokenize the unfiltered model output.
            responses_by_doc.setdefault(sample["doc_id"], sample["resps"][0][0])
    generated_tokens = sum(
        len(
            tokenizer(response, add_special_tokens=False)["input_ids"]
        )
        for response in responses_by_doc.values()
    )

summary = {
    "model_epochs": int(sys.argv[3]),
    "result_file": str(result_file.resolve()),
    "metrics": result["results"][task],
    "total_evaluation_time_seconds": seconds,
    "evaluated_samples": sample_count,
    "aggregate_samples_per_second": sample_count / seconds,
    "generated_tokens": generated_tokens,
    "aggregate_output_tokens_per_second": (
        generated_tokens / seconds if generated_tokens is not None else None
    ),
    "num_gpus": 8,
    "max_sequence_length": 2048,
    "decoding": "dynamic confidence-threshold",
    "throughput_scope": "eight-GPU end-to-end lm-eval wall time",
}
nfe_by_prompt = {}
for nfe_path in result_dir.glob("dynamic_nfe.rank*.jsonl"):
    with nfe_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            nfe_by_prompt.setdefault(record["prompt_sha256"], int(record["nfe"]))
if nfe_by_prompt:
    nfe_values = list(nfe_by_prompt.values())
    summary.update(
        {
            "dynamic_unmask_threshold": 0.85,
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

/nvme-data/neeleshgarg/envs/dllm/bin/python /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
