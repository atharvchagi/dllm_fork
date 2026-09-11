#!/usr/bin/env bash

# Run from /nvme-data/neeleshgarg/dllm_fork with the dllm conda environment active.

set -euo pipefail

cd /nvme-data/neeleshgarg/dllm_fork

CUDA_VISIBLE_DEVICES=4,5,6,7 \
WANDB_MODE=online WANDB_PROJECT="block-relay" accelerate launch \
  --config_file /nvme-data/neeleshgarg/dllm_fork/scripts/accelerate_configs/zero2.yaml \
  --num_processes 4 \
  /nvme-data/neeleshgarg/dllm_fork/examples/a2d/bd3lm/sft.py \
  --num_proc 8 \
  --loss_type BPTT \
  --model_name_or_path /nvme-data/neeleshgarg/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init \
  --dataset_args Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy \
  --max_length 2048 \
  --num_train_epochs 5 \
  --learning_rate 1e-4 \
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
  --logging_steps 10 \
  --run_name bd3lm-relay-k2-b4-parallel \
  --output_dir /nvme-data/neeleshgarg/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init/bd3lm/block4/relay-k2-len2048
