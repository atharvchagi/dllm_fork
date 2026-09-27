#!/usr/bin/env bash
# Run after Loophole training on dive9 physical GPUs 6,7,8,9:
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_loophole/eval.sh 8

set -euo pipefail

block_size=${1:-8}
case "${block_size}" in
  8|16|32) ;;
  *) echo "Block size must be 8, 16, or 32" >&2; exit 1 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_dir=${DLLM_CONDA_ENV:-/home/ngarg2/miniforge3/envs/dllm}
conda_hook=${DLLM_CONDA_HOOK:-/home/ngarg2/miniforge3/etc/profile.d/conda.sh}
checkpoint="${repo}/.models/gsm8k_loophole/bd3lm_block${block_size}"
result_dir="${repo}/results/gsm8k_loophole/bd3lm_block${block_size}"
# UUIDs correspond to nvidia-smi physical GPUs 6, 7, 8, and 9 on dive9.
gpu_ids=${DLLM_GPUS:-GPU-c31a66b9-cce8-bf56-a3cf-f026d1fdbb6a,GPU-ed6fa8a2-6c94-71dd-60f7-496a84b424bc,GPU-95109861-0d50-d1b0-e10e-b1b16ce19947,GPU-2cf4e296-e09d-127d-414d-54efef213402}

if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
  echo "Missing trained Loophole checkpoint: ${checkpoint}" >&2
  exit 1
fi

if [[ -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc"
fi
source "${conda_hook}"
conda activate "${env_dir}"
cd "${repo}"

export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "${result_dir}"

CUDA_VISIBLE_DEVICES="${gpu_ids}" \
WANDB_MODE="${WANDB_MODE:-online}" python -m accelerate.commands.launch \
  --num_processes 4 \
  "${repo}/dllm/pipelines/a2d/eval.py" \
  --tasks gsm8k_cot \
  --model a2d_bd3lm \
  --device cuda \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size 8 \
  --loophole_enabled \
  --model_args "pretrained=${checkpoint},max_new_tokens=256,steps=256,block_size=${block_size},cfg_scale=0.0,temperature=0.0,right_shift_logits=True" \
  --output_path "${result_dir}" \
  --wandb_args "project=gsm8k-bd3lm-loophole,name=qwen3-0.6b-gsm8k-bd3lm-loophole-b${block_size}-eval,job_type=eval" \
  --wandb_config_args "block_size=${block_size},train_max_length=1024,max_new_tokens=256,num_gpus=4,loophole_enabled=True" \
  --log_samples
