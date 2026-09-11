#!/usr/bin/env bash

# Run standard MDLM supervised fine-tuning on four GPUs.

set -euo pipefail

cd /nvme-data/neeleshgarg/dllm_fork

WANDB_MODE=online WANDB_PROJECT="mdlm-experiments" accelerate launch \
  --config_file /nvme-data/neeleshgarg/dllm_fork/scripts/accelerate_configs/zero2.yaml \
  --num_processes 4 \
  /nvme-data/neeleshgarg/dllm_fork/examples/a2d/mdlm/sft.py \
  --num_proc 8 \
  --loss_type CE \
  --model_name_or_path /nvme-data/neeleshgarg/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init \
  --dataset_args Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy \
  --max_length 4096 \
  --num_train_epochs 5 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --right_shift_logits True \
  --eval_strategy no \
  --save_strategy epoch \
  --logging_steps 10 \
  --run_name qwen3-0.6b-mdlm-baseline-mix60k-len4096-5ep \
  --output_dir /nvme-data/neeleshgarg/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init/mdlm/baseline
