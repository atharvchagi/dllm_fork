WANDB_MODE=online WANDB_PROJECT="4B-SFT" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
  examples/a2d/bd3lm/sft.py \
  --num_proc 8 \
  --loss_type CE \
  --model_name_or_path "/scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/models/a2d/Qwen3-4B/" \
  --dataset_args "Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy" \
  --max_length 4096 \
  --num_train_epochs 50 \
  --save_strategy "steps" \
  --save_steps 148 \
  --eval_strategy "no" \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --block_size 4 \
  --output_dir "/scratch/project/prj-02-pi-shuiwang-ji/atharvchagi/models/a2d/Qwen3-4B/bd3lm/SFT-60k-50-1epoch/"

