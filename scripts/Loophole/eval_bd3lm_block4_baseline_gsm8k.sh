#!/usr/bin/env bash
#SBATCH --job-name=eval-bd3lm-b4-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --partition=2xlong
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/user/atharvchagi_tamu.edu/dllm_fork/logs/%x-%j.out
#SBATCH --error=/scratch/user/atharvchagi_tamu.edu/dllm_fork/logs/%x-%j.err

# Run with: sbatch /scratch/user/atharvchagi_tamu.edu/dllm_fork/scripts/Loophole/eval_bd3lm_block4_baseline_gsm8k.sh
set -euo pipefail
cd /scratch/user/atharvchagi_tamu.edu/dllm_fork
source /sw/eb/sw/Anaconda3/2025.12-2/etc/profile.d/conda.sh
conda activate /home/atharvchagi_tamu.edu/.conda/envs/dllm
module load CUDA/12.6.0
export PYTHONPATH=/scratch/user/atharvchagi_tamu.edu/dllm_fork:/scratch/user/atharvchagi_tamu.edu/dllm_fork/lm-evaluation-harness

/home/atharvchagi_tamu.edu/.conda/envs/dllm/bin/python -m accelerate.commands.launch \
  --num_processes 8 \
  /scratch/user/atharvchagi_tamu.edu/dllm_fork/dllm/pipelines/a2d/eval.py \
  --tasks gsm8k_cot --num_fewshot 0 --model a2d_bd3lm --apply_chat_template \
  --batch_size 2 --log_samples \
  --model_args "pretrained=/scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/SC-models/a2d/Qwen3-0.6B-a2d-init/bd3lm/block4/len4096/baseline/checkpoint-final,max_new_tokens=4096,steps=4096,block_size=4,temperature=0.0,remasking=low_confidence,confidence_threshold=0.85,right_shift_logits=true,loophole_enabled=false" \
  --output_path /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/eval-results/bd3lm-b4-baseline-gsm8k

/home/atharvchagi_tamu.edu/.conda/envs/dllm/bin/python \
  /scratch/user/atharvchagi_tamu.edu/dllm_fork/examples/a2d/mdlm/benchmark_perplexity.py \
  --model-spec baseline=/scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/SC-models/a2d/Qwen3-0.6B-a2d-init/bd3lm/block4/len4096/baseline/checkpoint-final,loophole=false \
  --output /scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/eval-results/bd3lm-b4-baseline-gsm8k-ppl.json \
  --dataset-path openai/gsm8k --dataset-name main --split test --text-field answer \
  --sequence-length 4096 --batch-size 2 --device cuda --no-loophole-inference-loophole
