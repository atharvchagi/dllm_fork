#!/usr/bin/env bash
# Run after training on dive9 physical GPUs 2,3,4,5 (override with DLLM_GPUS):
# bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval.sh 8
# Evaluates the full GSM8K test split at batch size 1 per GPU.

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
run_stamp=$(date +%Y%m%d_%H%M%S)
result_dir="${repo}/results/gsm8k_baseline/bd3lm_block${block_size}/batch1_4gpu/${run_stamp}"
log_file="${result_dir}/eval.log"
# UUIDs correspond to four identical RTX 6000 Ada GPUs: physical 2, 3, 4, and 5.
gpu_ids=${DLLM_GPUS:-GPU-cb79cdce-7f85-fe4f-6d63-3a0abc3e3929,GPU-af4a1b59-58a5-af18-bdc9-0bf62e636634,GPU-90c9fbaa-d3fb-17aa-b33d-915209e44d3f,GPU-9a62c6db-c881-040b-4a9c-3f0c81475a67}

if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
  echo "Missing trained checkpoint: ${checkpoint}" >&2
  exit 1
fi

IFS=',' read -r -a selected_gpus <<< "${gpu_ids}"
active_gpu_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits)
for gpu_uuid in "${selected_gpus[@]}"; do
  if grep -Fqx -- "${gpu_uuid}" <<< "${active_gpu_uuids}"; then
    echo "Selected GPU is already running a compute process: ${gpu_uuid}" >&2
    exit 1
  fi
done

if [[ -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc"
fi
source "${conda_hook}"
conda activate "${env_dir}"
cd "${repo}"

export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-gsm8k-bd3lm-baseline}"
mkdir -p "${result_dir}"

CUDA_VISIBLE_DEVICES="${gpu_ids}" \
python -m accelerate.commands.launch \
  --num_processes 4 \
  --num_machines 1 \
  --main_process_port "${DLLM_MAIN_PROCESS_PORT:-0}" \
  --mixed_precision no \
  --dynamo_backend no \
  "${repo}/dllm/pipelines/a2d/eval.py" \
  --tasks gsm8k_cot \
  --model a2d_bd3lm \
  --device cuda \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size 1 \
  --model_args "pretrained=${checkpoint},max_new_tokens=256,steps=256,block_size=${block_size},cfg_scale=0.0,temperature=0.0,right_shift_logits=True,loophole_enabled=False" \
  --output_path "${result_dir}" \
  --log_samples 2>&1 | tee "${log_file}"

python "${repo}/scripts/gsm8k_baseline/summarize_eval.py" \
  "${result_dir}" "${checkpoint}" "${block_size}" 4 1 "${gpu_ids}"
