#!/usr/bin/env bash

# Run directly with: bash /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/eval/qwen_0.6b_bd3lm_block4_k2_gsm8k_perplexity.sh
# Optional first argument: model epochs, 5 or 8 (default 8).

set -euo pipefail

model_epochs=${1:-8}
if [[ "${model_epochs}" != 5 && "${model_epochs}" != 8 ]]; then
  echo "Model epochs must be 5 or 8." >&2
  exit 1
fi

source /home/ngarg2/miniforge3/etc/profile.d/conda.sh
conda activate /nvme-data/neeleshgarg/envs/dllm

cd /nvme-data/neeleshgarg/repos/dllm_fork

checkpoint=/nvme-data/neeleshgarg/repos/dllm_fork/.models/Qwen/Qwen3-0.6b-a2d-init/bd3lm/relay_block4_k2_${model_epochs}ep
if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
  echo "Missing trained checkpoint: ${checkpoint}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=7 accelerate launch \
  --num_processes 1 \
  /nvme-data/neeleshgarg/repos/dllm_fork/examples/a2d/bd3lm/sft.py \
  --eval_only True \
  --loss_type BPTT \
  --model_name_or_path "${checkpoint}" \
  --eval_dataset_args 'openai/gsm8k[test:1319]' \
  --num_proc 8 \
  --max_length 2048 \
  --per_device_eval_batch_size 4 \
  --block_size 4 \
  --loophole_enabled True \
  --relay_num_steps 2 \
  --relay_unmask_threshold 0.85 \
  --right_shift_logits True \
  --eval_strategy no \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --output_dir /nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/${model_epochs}ep/full_eval/perplexity

python /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
