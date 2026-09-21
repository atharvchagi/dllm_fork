#!/usr/bin/env bash

# Run: bash /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/train/qwen_0.6b_bd3lm_block4_k2_continue5ep.sh
# Continue from the saved five-epoch weights with a fresh optimizer and schedule.

set -euo pipefail

source /home/ngarg2/miniforge3/etc/profile.d/conda.sh
conda activate /nvme-data/neeleshgarg/envs/dllm

cd /nvme-data/neeleshgarg/repos/dllm_fork

checkpoint=/nvme-data/neeleshgarg/repos/dllm_fork/.models/Qwen/Qwen3-0.6b-a2d-init/bd3lm/relay_block4_k2_5ep
output_dir=/nvme-data/neeleshgarg/repos/dllm_fork/.models/Qwen/Qwen3-0.6b-a2d-init/bd3lm/relay_block4_k2_10ep
log_file=/nvme-data/neeleshgarg/repos/dllm_fork/logs/qwen_0.6b_bd3lm_block4_k2_continue5ep_$(date +%Y%m%d_%H%M%S).log

if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
  echo "Missing trained checkpoint: ${checkpoint}" >&2
  exit 1
fi
if [[ -e "${output_dir}" ]]; then
  echo "Output already exists: ${output_dir}. Choose a new output directory for another weights-only continuation." >&2
  exit 1
fi
mkdir -p /nvme-data/neeleshgarg/repos/dllm_fork/logs

# Start a separate W&B run: Trainer steps restart at zero for weights-only continuation.
unset WANDB_RUN_ID WANDB_RESUME
CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_MODE=online WANDB_PROJECT="block-relay" WANDB_ENTITY="ngarg2-dive" \
WANDB_NOTES="Five more epochs from https://wandb.ai/ngarg2-dive/block-relay/runs/nkbuvpr6; fresh optimizer and cosine schedule." \
accelerate launch \
  --config_file /nvme-data/neeleshgarg/repos/dllm_fork/scripts/accelerate_configs/zero2.yaml \
  --num_processes 4 \
  /nvme-data/neeleshgarg/repos/dllm_fork/examples/a2d/bd3lm/sft.py \
  --num_proc 8 \
  --loss_type BPTT \
  --model_name_or_path "${checkpoint}" \
  --dataset_args Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy \
  --max_length 2048 \
  --num_train_epochs 5 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --block_size 4 \
  --loophole_enabled True \
  --relay_num_steps 2 \
  --relay_unmask_threshold 0.85 \
  --right_shift_logits True \
  --eval_dataset_args openai/gsm8k[test:1319] \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_only_model True \
  --save_total_limit 2 \
  --report_to wandb \
  --logging_steps 10 \
  --run_name bd3lm-relay-k2-b4-parallel-continue5ep \
  --output_dir "${output_dir}" 2>&1 | tee "${log_file}"
