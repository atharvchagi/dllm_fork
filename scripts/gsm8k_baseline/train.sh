#!/usr/bin/env bash
# Run directly on GPUs 3,4,5,6 (override with DLLM_GPUS):
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/train.sh 8
# Repeat with 16 and 32.

set -euo pipefail

block_size=${1:?Pass block size 8, 16, or 32}
case "${block_size}" in
  8|16|32) ;;
  *) echo "Block size must be 8, 16, or 32" >&2; exit 1 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_dir=${DLLM_CONDA_ENV:-/home/ngarg2/miniforge3/envs/dllm}
conda_hook=${DLLM_CONDA_HOOK:-/home/ngarg2/miniforge3/etc/profile.d/conda.sh}
model_dir="${repo}/.models/Qwen0.6b/a2d-init"
output_dir="${repo}/.models/gsm8k_baseline/bd3lm_block${block_size}"

if [[ ! -f "${model_dir}/config.json" || ! -f "${model_dir}/tokenizer_config.json" ]]; then
  echo "Missing local A2D init model: ${model_dir}" >&2
  echo "Download divelab/Qwen3-0.6B-a2d-init into ${model_dir} first." >&2
  exit 1
fi
if [[ ! -f "${model_dir}/model.safetensors" && ! -f "${model_dir}/model.safetensors.index.json" ]]; then
  echo "Missing model weights in ${model_dir}" >&2
  exit 1
fi

if [[ -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc"
fi
source "${conda_hook}"
conda activate "${env_dir}"
cd "${repo}"

CUDA_VISIBLE_DEVICES="${DLLM_GPUS:-3,4,5,6}" \
WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT=gsm8k-bd3lm-baseline \
  python -m accelerate.commands.launch \
    --config_file "${repo}/scripts/accelerate_configs/zero2.yaml" \
    --num_processes 4 \
    "${repo}/examples/a2d/bd3lm/sft.py" \
    --num_proc 8 \
    --loss_type CE \
    --model_name_or_path "${model_dir}" \
    --dataset_args openai/gsm8k \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --block_size "${block_size}" \
    --loophole_enabled False \
    --right_shift_logits True \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 10 \
    --seed 42 \
    --data_seed 42 \
    --run_name "qwen3-0.6b-gsm8k-bd3lm-baseline-b${block_size}" \
    --output_dir "${output_dir}"
