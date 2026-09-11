#!/usr/bin/env bash
set -euo pipefail

# Run from an allocated 8-GPU node after activating the dllm conda environment.
WANDB_MODE=online WANDB_PROJECT="bd3lm-loophole" WANDB\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /scratch/user/atharvchagi_tamu.edu/dllm_fork/scripts/accelerate_configs/zero2.yaml \
  --num_processes 8 \
  /scratch/user/atharvchagi_tamu.edu/dllm_fork/examples/a2d/bd3lm/sft.py \
  --num_proc 8 \
  --loss_type CE \
  --model_name_or_path divelab/Qwen3-0.6B-a2d-init \
  --dataset_args Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy \
  --max_length 512 \
  --num_train_epochs 5 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --block_size 4 \
  --loophole_enabled True \
  --loophole_self_cond_rate 0.9 \
  --loophole_eval_self_cond_rate 1.0 \
  --right_shift_logits True \
  --eval_strategy epoch \
  --eval_dataset 
  --save_strategy epoch \
  --logging_steps 10 \
  --run_name qwen3-0.6b-bd3lm-loophole-mix60k-4096 \
  --output_dir /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/SC-models/a2d/loopholed/Qwen3-0.6B/bd3lm/block-size-4/
