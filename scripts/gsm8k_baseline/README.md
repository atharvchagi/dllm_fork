# Qwen3-0.6B BD3LM GSM8K baselines

The [training launcher](/home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/train.sh) starts from a local copy of `divelab/Qwen3-0.6B-a2d-init` and fine-tunes on the full `openai/gsm8k` train split. Each block size (8, 16, 32) gets a separate checkpoint. The loss is CE, Loophole is off, and the other training settings are identical: four GPUs, batch size 8 per GPU, gradient accumulation 1 (effective batch 32), five epochs, learning rate `1e-4`, and training sequence length 1024. Training does not score the test split.

The [evaluation launcher](/home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval.sh) scores the final checkpoint on the full GSM8K test split using `gsm8k_cot`, zero-shot chat formatting, 256 generated tokens, 256 denoising steps, and the matching block size. Results are saved under `/home/ngarg2/repos/dllm_fork/results/gsm8k_baseline/`. Training and evaluation log to the `gsm8k-bd3lm-baseline` W&B project by default.

First, download the model once on the login node, without requesting a GPU. The account currently has no `~/.zshrc`, so use the Miniforge conda hook:

```bash
source /home/ngarg2/miniforge3/etc/profile.d/conda.sh
conda activate /home/ngarg2/miniforge3/envs/dllm
hf download divelab/Qwen3-0.6B-a2d-init --local-dir /home/ngarg2/repos/dllm_fork/.models/Qwen0.6b/a2d-init
```

This creates `/home/ngarg2/repos/dllm_fork/.models/Qwen0.6b/a2d-init`. The training launcher checks for the local model before requesting work from it.

Run each block size directly on physical GPUs 3, 4, 5, and 6:

```bash
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/train.sh 8
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/train.sh 16
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/train.sh 32
```

After a training command finishes, evaluate that block size:

```bash
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval.sh 8
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval.sh 16
bash /home/ngarg2/repos/dllm_fork/scripts/gsm8k_baseline/eval.sh 32
```

Each launcher finds the repository from its own location and uses `CUDA_VISIBLE_DEVICES=3,4,5,6` by default; set `DLLM_GPUS` to use another four GPU indices. It activates `/home/ngarg2/miniforge3/envs/dllm` by default; set `DLLM_CONDA_ENV` and `DLLM_CONDA_HOOK` if needed. Download the model into that checkout's `.models/Qwen0.6b/a2d-init` directory before training; model weights are ignored by this clone's private Git exclude file. Training writes checkpoints under that checkout's `.models/gsm8k_baseline/`. Run `wandb login` in the `dllm` environment before starting if this account is not logged in; set `WANDB_MODE=disabled` if W&B logging is unwanted.
