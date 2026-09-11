#!/usr/bin/env bash
#SBATCH --job-name=bd3lm-b8-loophole
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --partition=2xlong
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/user/atharvchagi_tamu.edu/dllm_fork/logs/%x-%j.out
#SBATCH --error=/scratch/user/atharvchagi_tamu.edu/dllm_fork/logs/%x-%j.err

set -euo pipefail

# Run from an allocated 8-GPU node after activating the dllm conda environment.
cd /scratch/user/atharvchagi_tamu.edu/dllm_fork
conda activate dllm

# DeepSpeed needs a CUDA toolkit (and CUDA_HOME) when it initializes its ops.
module load CUDA/12.6.0

WANDB_MODE=online WANDB_PROJECT="bd3lm-experiments" \
  /home/atharvchagi_tamu.edu/.conda/envs/dllm/bin/python \
  -m accelerate.commands.launch \
  --config_file /scratch/user/atharvchagi_tamu.edu/dllm_fork/scripts/accelerate_configs/zero2.yaml \
  --num_processes 8 \
  /scratch/user/atharvchagi_tamu.edu/dllm_fork/examples/a2d/bd3lm/sft.py \
  --num_proc 8 \
  --loss_type CE \
  --model_name_or_path divelab/Qwen3-0.6B-a2d-init \
  --dataset_args Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy \
  --max_length 4096 \
  --num_train_epochs 5 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --block_size 8 \
  --loophole_enabled True \
  --loophole_self_cond_rate 0.9 \
  --loophole_eval_self_cond_rate 1.0 \
  --right_shift_logits True \
  --eval_dataset_args openai/gsm8k[test:1319] \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 10 \
  --run_name qwen3-0.6b-bd3lm-block8-loophole-mix60k-len4096-5ep \
  --output_dir /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/SC-models/a2d/Qwen3-0.6B-a2d-init/bd3lm/block8/len4096/loophole
