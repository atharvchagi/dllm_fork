These are the GSM8K results for the trained BD3LM Relay model with block size 4 and a two-step training window.

The original checkpoint is [relay_block4_k2_5ep](/nvme-data/neeleshgarg/repos/dllm_fork/.models/Qwen/Qwen3-0.6b-a2d-init/bd3lm/relay_block4_k2_5ep). Model weights and tokenizer files live directly in that directory. The continuation launcher reads those weights and saves separately to `/nvme-data/neeleshgarg/repos/dllm_fork/.models/Qwen/Qwen3-0.6b-a2d-init/bd3lm/relay_block4_k2_10ep` after five additional epochs, with a fresh optimizer and learning-rate schedule. Optimizer and scheduler state are not saved.

All existing Relay results are under [5ep](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/5ep). The [10ep directory](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/10ep) is reserved for evaluations of the continued model; it contains no evaluation results yet.

Full-dataset results are under [full_eval](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/5ep/full_eval): static generation, dynamic generation, and masked-token loss/perplexity. Static used one GPU, batch 4, and 256 **new** tokens; dynamic used eight GPUs, batch 4 per GPU, and a 2048-token **total** limit. These are not a controlled comparison of decoding policies. The perplexity is exponentiated masked-token NLL across the two Relay passes, not conventional AR sequence perplexity.

The [32-sample benchmarks](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/5ep/throughput_random32) all use dynamic decoding, GPU 7, batch 1, block size 4, threshold 0.85, greedy generation, and the same fixed random document IDs (selection seed 42). Length directories specify the total padded-prompt-plus-output limit. There are two separate 2048-token runs and one each at 1024, 512, and 256.

The AR comparison remains under [Qwen results](/nvme-data/neeleshgarg/repos/dllm_fork/results/qwen_0.6b_ar_gsm8k/throughput_random32). It uses the original Qwen3-0.6B, standard HF generation with KV caching, thinking disabled, the same samples, GPU 7, batch 1, and a 2048-token total limit.

[throughput_analysis/comparison.csv](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/throughput_analysis/comparison.csv) includes only the 32-sample Relay benchmarks from both epoch directories and the AR benchmark runs. Its `model_epochs` column distinguishes the Relay checkpoints and is blank for AR.

The separate [5ep full-evaluation summary](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/5ep/full_eval/summary.json) includes full GSM8K generation accuracy, throughput, runtime, output lengths, decoding settings, and a separate loss/perplexity evaluation section. An equivalent summary will be created under the ten-epoch full-evaluation directory when results are available. Perplexity's original metrics remain in [eval_results.json](/nvme-data/neeleshgarg/repos/dllm_fork/results/relay/qwen_0.6b_bd3lm_block4_k2_gsm8k/5ep/full_eval/perplexity/eval_results.json).

Benchmark and evaluation scripts regenerate these summaries after successful completion. To rebuild them from saved files without inference:

```bash
python /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/benchmark/summarize_throughput.py
```

Interpretation and provenance:

- Throughput counts saved, stop-trimmed output text re-tokenized, divided by end-to-end lm-eval time. This includes model loading and scoring; the AR download happened before timing. It is not GPU-only generation throughput.
- AR stops on task stop strings during generation; Relay trims those strings afterward. Output lengths and useful work can differ.
- NFE counts active denoising passes, excluding prefix/cache preparation. Missing NFE means unavailable, not zero.
- Accuracy uses flexible extraction. Strict-format matching was zero. Truncation can change extracted answers: the extra correct answer at length 512 was an extraction effect on unfinished reasoning.
- Raw lm-eval JSON can report 1319 effective samples even for these 32-sample runs. The derived summaries use the actual unique document count.
- Both original 2048-token sample files are byte-for-byte identical. Their runtimes differ; the timing change cannot be attributed confidently to removing the unused dynamic `steps` argument.
- The first 2048 run's NFE and console log were overwritten before this cleanup. Its summary was reconstructed from its own raw metrics and the verified identical outputs; NFE is intentionally absent.
- The two original 2048 run folder names use result timestamps because separate original run folders did not exist. Other benchmark folder names use launch timestamps.
- Raw result JSON, sample JSONL, NFE JSONL, model configuration, and training arguments retain their original contents, including historical paths. Only derived summary references were updated during reorganization.
- The two perplexity JSON files contain the same metrics and are both retained as trainer outputs.

Launchers are organized under [training](/nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/train), [full evaluation](/nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/eval), and [benchmarks](/nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/benchmark). The Relay benchmark accepts total sequence length as its first argument (default 2048) and model epochs as its second (5 or 10, default 10). Both full-evaluation launchers accept model epochs as their first argument, defaulting to 10. Each launcher selects the matching checkpoint and results directory and stops if the checkpoint is unavailable.

To benchmark the original five-epoch model at length 512:

```bash
bash /nvme-data/neeleshgarg/repos/dllm_fork/scripts/relay/benchmark/qwen_0.6b_bd3lm_block4_k2_gsm8k.sh 512 5
```
