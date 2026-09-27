# Qwen3-0.6B BD3LM GSM8K Loophole

The [training launcher](/home/ngarg2/repos/dllm_fork/scripts/gsm8k_loophole/train.sh) starts from `/home/ngarg2/repos/dllm_fork/.models/Qwen0.6b/a2d-init` and uses the same GSM8K data, CE loss, length 1024, five epochs, learning rate `1e-4`, batch size 8 per GPU, gradient accumulation 1, four processes, seed, and ZeRO-2 settings as the [baseline launcher](/home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/train.sh). It enables the Loophole adapter with self conditioning rate 0.9. Block size defaults to 8.

On dive9 it selects physical GPUs 6, 7, 8, and 9 by UUID, avoiding CUDA index remapping. All Loophole train and eval runs use the `gsm8k-bd3lm-loophole` W&B project; baseline runs use `gsm8k-bd3lm-baseline`. Run in a separate tmux session:

```bash
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_loophole/train.sh 8
```

After training, the [evaluation launcher](/home/ngarg2/repos/dllm_fork/scripts/gsm8k_loophole/eval.sh) runs the full GSM8K test split with the baseline's 256-token, 256-step decoding settings and the Loophole adapter enabled:

```bash
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_loophole/eval.sh 8
```

Checkpoints go to `/home/ngarg2/repos/dllm_fork/.models/gsm8k_loophole/`; evaluation results go to `/home/ngarg2/repos/dllm_fork/results/gsm8k_loophole/`. Set `DLLM_GPUS` to a comma-separated list of four GPU UUIDs if the selected GPUs change.

Training keeps one full checkpoint at the halfway point, including optimizer, scheduler, and RNG state. Restarting the launcher resumes from that checkpoint automatically. After successful completion, it removes intermediate checkpoints and keeps the final model.
