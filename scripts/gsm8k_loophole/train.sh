#!/usr/bin/env bash
# Run directly on dive9 physical GPUs 6,7,8,9:
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_loophole/train.sh 8

set -euo pipefail

block_size=${1:-8}
case "${block_size}" in
  8|16|32) ;;
  *) echo "Block size must be 8, 16, or 32" >&2; exit 1 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_dir=${DLLM_CONDA_ENV:-/home/ngarg2/miniforge3/envs/dllm}
conda_hook=${DLLM_CONDA_HOOK:-/home/ngarg2/miniforge3/etc/profile.d/conda.sh}
model_dir="${repo}/.models/Qwen0.6b/a2d-init"
output_dir="${repo}/.models/gsm8k_loophole/bd3lm_block${block_size}"
# UUIDs correspond to nvidia-smi physical GPUs 6, 7, 8, and 9 on dive9.
gpu_ids=${DLLM_GPUS:-GPU-c31a66b9-cce8-bf56-a3cf-f026d1fdbb6a,GPU-ed6fa8a2-6c94-71dd-60f7-496a84b424bc,GPU-95109861-0d50-d1b0-e10e-b1b16ce19947,GPU-2cf4e296-e09d-127d-414d-54efef213402}
resume_args=()

if [[ -d "${output_dir}" ]]; then
  latest_checkpoint=$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -n 1)
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${output_dir}/${latest_checkpoint}")
    echo "Resuming from ${output_dir}/${latest_checkpoint}"
  fi
fi

if [[ ! -f "${model_dir}/config.json" || ! -f "${model_dir}/tokenizer_config.json" ]]; then
  echo "Missing local A2D init model: ${model_dir}" >&2
  exit 1
fi
if [[ ! -f "${model_dir}/model.safetensors" && ! -f "${model_dir}/model.safetensors.index.json" ]]; then
  echo "Missing model weights in ${model_dir}" >&2
  exit 1
fi
if [[ -f "${output_dir}/model.safetensors" ]]; then
  echo "Final Loophole checkpoint already exists: ${output_dir}" >&2
  exit 1
fi

if [[ -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc"
fi
source "${conda_hook}"
conda activate "${env_dir}"
cd "${repo}"

CUDA_VISIBLE_DEVICES="${gpu_ids}" \
WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT=gsm8k-bd3lm-loophole \
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
    --loophole_enabled True \
    --loophole_self_cond_rate 0.9 \
    --loophole_eval_self_cond_rate 1.0 \
    --right_shift_logits True \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 0.5 \
    --save_total_limit 1 \
    --save_only_model False \
    --logging_steps 10 \
    --seed 42 \
    --data_seed 42 \
    --run_name "qwen3-0.6b-gsm8k-bd3lm-loophole-b${block_size}" \
    "${resume_args[@]}" \
    --output_dir "${output_dir}"

if [[ -f "${output_dir}/model.safetensors" || -f "${output_dir}/model.safetensors.index.json" ]]; then
  final_checkpoint=$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -n 1)
  if [[ -n "${final_checkpoint}" && -f "${output_dir}/${final_checkpoint}/trainer_state.json" ]]; then
    cp "${output_dir}/${final_checkpoint}/trainer_state.json" "${output_dir}/trainer_state.json"
  fi
  find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
fi
