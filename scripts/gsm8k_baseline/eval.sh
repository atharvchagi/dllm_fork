#!/usr/bin/env bash
# Run after training on GPUs 3,4,5,6 (override with DLLM_GPUS):
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval.sh 8
# Repeat with 16 and 32. Evaluates the full GSM8K test split.

set -euo pipefail

block_size=${1:?Pass block size 8, 16, or 32}
case "${block_size}" in
  8|16|32) ;;
  *) echo "Block size must be 8, 16, or 32" >&2; exit 1 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_dir=${DLLM_CONDA_ENV:-/home/ngarg2/miniforge3/envs/dllm}
conda_hook=${DLLM_CONDA_HOOK:-/home/ngarg2/miniforge3/etc/profile.d/conda.sh}
checkpoint="${repo}/.models/gsm8k_baseline/bd3lm_block${block_size}"
result_dir="${repo}/results/gsm8k_baseline/bd3lm_block${block_size}"

if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
  echo "Missing trained checkpoint: ${checkpoint}" >&2
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

CUDA_VISIBLE_DEVICES="${DLLM_GPUS:-3,4,5,6}" \
WANDB_MODE="${WANDB_MODE:-online}" python -m accelerate.commands.launch \
  --num_processes 4 \
  "${repo}/dllm/pipelines/a2d/eval.py" \
  --tasks gsm8k_cot \
  --model a2d_bd3lm \
  --device cuda \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size 8 \
  --model_args "pretrained=${checkpoint},max_new_tokens=256,steps=256,block_size=${block_size},cfg_scale=0.0,temperature=0.0,right_shift_logits=True,loophole_enabled=False" \
  --output_path "${result_dir}" \
  --wandb_args "project=gsm8k-bd3lm-baseline,name=qwen3-0.6b-gsm8k-bd3lm-baseline-b${block_size}-eval,job_type=eval" \
  --wandb_config_args "block_size=${block_size},train_max_length=1024,max_new_tokens=256,num_gpus=4" \
  --log_samples
