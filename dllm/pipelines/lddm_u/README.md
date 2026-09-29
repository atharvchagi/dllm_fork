# LDDM-U evaluation

This package ports the model, diffusion process, data pipeline, and EMA loader
from the authors' [official LDDM repository](https://github.com/ahn-ml/lddm) at
commit `7d86851959cfb937be8febe1a7b5e8a2157ae287`. The upstream Apache 2.0 license
is in
[`UPSTREAM_LICENSE`](/home/ngarg2/repos/dllm_fork/dllm/pipelines/lddm_u/UPSTREAM_LICENSE).

The released checkpoints are stored at
`/home/ngarg2/repos/dllm_fork/.models/LDDM-U/lddm_u_owt_1M.ckpt` and
`/home/ngarg2/repos/dllm_fork/.models/UDLM/udlm_owt_1M.ckpt`. They stay
untracked through `/home/ngarg2/repos/dllm_fork/.git/info/exclude`.

Install the optional dependencies once:

```bash
source /home/ngarg2/miniforge3/etc/profile.d/conda.sh
conda activate /home/ngarg2/miniforge3/envs/dllm
pip install -e '/home/ngarg2/repos/dllm_fork[lddm-u]'
```

Run full held-out OpenWebText evaluations at batch size 1 on separate GPUs:

```bash
LDDM_GPU=1 bash /home/ngarg2/repos/dllm_fork/scripts/uniform_loophole/eval_lddm_u_owt.sh ppl
UDLM_GPU=2 bash /home/ngarg2/repos/dllm_fork/scripts/uniform_loophole/eval_udlm_owt.sh ppl
```

Set `LDDM_GPU` and `UDLM_GPU` to physical GPU indices. `LDDM_PPL_BATCHES`,
`LDDM_PPL_WARMUP_BATCHES`, `LDDM_PPL_MC_SAMPLES`, `LDDM_SAMPLE_BATCHES`, and
`LDDM_WARMUP_BATCHES` control evaluation size. A value of `0` for
`LDDM_PPL_BATCHES` or `UDLM_PPL_BATCHES` evaluates the full validation split.
Dataset throughput times only the synchronized NLL calls on held-out
sequences. Unconditional diffusion
sampling speed is reported separately. Set `LDDM_GENERATIVE_PPL=1` to add
GPT-2 Large generative perplexity. Results are written below
`/home/ngarg2/repos/dllm_fork/results/uniform_loophole/lddm_u_owt/` and
`/home/ngarg2/repos/dllm_fork/results/uniform_loophole/udlm_owt/`.

Both launchers log to the `uniform-loophole` W&B project under the
`owt-uniform-checkpoints` group, using separate timestamped run names.

`perplexity` is the official diffusion NLL estimate. The additional
`denoising_token_accuracy` is a random-time clean-token recovery diagnostic;
OpenWebText has no task accuracy metric.
