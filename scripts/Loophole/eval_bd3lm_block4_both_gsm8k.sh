#!/usr/bin/env bash
# Run directly inside an allocated four-GPU node:
# bash /scratch/user/atharvchagi_tamu.edu/dllm_fork/scripts/Loophole/eval_bd3lm_block4_both_gsm8k.sh

set -eo pipefail

cd /scratch/user/atharvchagi_tamu.edu/dllm_fork
source /sw/eb/sw/Anaconda3/2025.12-2/etc/profile.d/conda.sh
conda activate /home/atharvchagi_tamu.edu/.conda/envs/dllm
module load CUDA/12.6.0
set -u

export PYTHONPATH=/scratch/user/atharvchagi_tamu.edu/dllm_fork:/scratch/user/atharvchagi_tamu.edu/dllm_fork/lm-evaluation-harness
mkdir -p /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/eval-results
mkdir -p /scratch/user/atharvchagi_tamu.edu/dllm_fork/logs

echo "Evaluating block-4 baseline GSM8K accuracy"
/home/atharvchagi_tamu.edu/.conda/envs/dllm/bin/python -m accelerate.commands.launch \
  --num_processes 4 \
  /scratch/user/atharvchagi_tamu.edu/dllm_fork/dllm/pipelines/a2d/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 0 \
  --batch_size 2 \
  --model a2d_bd3lm \
  --apply_chat_template \
  --log_samples \
  --model_args "pretrained=/scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/SC-models/a2d/Qwen3-0.6B-a2d-init/bd3lm/block4/len4096/baseline/checkpoint-final,max_new_tokens=4096,block_size=4,temperature=0.0,remasking=low_confidence,confidence_threshold=0.85,right_shift_logits=true,loophole_enabled=false" \
  --output_path /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/eval-results/bd3lm-b4-baseline-gsm8k \
  2>&1 | tee /scratch/user/atharvchagi_tamu.edu/dllm_fork/logs/eval-bd3lm-b4-baseline-gsm8k.log

echo "Evaluating block-4 Loophole GSM8K accuracy"
/home/atharvchagi_tamu.edu/.conda/envs/dllm/bin/python -m accelerate.commands.launch \
  --num_processes 4 \
  /scratch/user/atharvchagi_tamu.edu/dllm_fork/dllm/pipelines/a2d/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 0 \
  --batch_size 2 \
  --model a2d_bd3lm \
  --apply_chat_template \
  --log_samples \
  --loophole_enabled \
  --model_args "pretrained=/scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/SC-models/a2d/Qwen3-0.6B-a2d-init/bd3lm/block4/len4096/loophole/checkpoint-final,max_new_tokens=4096,block_size=4,temperature=0.0,remasking=low_confidence,confidence_threshold=0.85,right_shift_logits=true,loophole_enabled=true" \
  --output_path /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/eval-results/bd3lm-b4-loophole-gsm8k \
  2>&1 | tee /scratch/user/atharvchagi_tamu.edu/dllm_fork/logs/eval-bd3lm-b4-loophole-gsm8k.log
