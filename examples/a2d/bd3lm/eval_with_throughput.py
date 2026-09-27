"""Evaluate accuracy and measure batch-one decoding throughput.

Set THROUGHPUT_DIR to a fresh absolute directory and pass the usual A2D eval
CLI flags to this file using Accelerate in the dllm environment.
"""

import json
import os
from pathlib import Path
import runpy
import time

import torch

from dllm.core.eval import base


original_generate_until = base.BaseEvalHarness.generate_until
base.tqdm = lambda iterable, **kwargs: iterable


def keep_last_prefix_logit(model, args, kwargs):
    """Only the final prefix logit is consumed by BD3LM decoding."""
    if kwargs.get("use_cache") is True:
        kwargs["logits_to_keep"] = 1
    return args, kwargs


def generate_with_throughput(self, requests):
    """Time synchronized sampler calls, excluding inter-rank barriers."""
    if self.batch_size != 1:
        raise ValueError("This throughput entrypoint requires batch size 1")
    output_dir = Path(os.environ["THROUGHPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"generations-rank{self.rank}.jsonl"
    original_sample = self.sampler.sample
    hook = self.model.register_forward_pre_hook(keep_last_prefix_logit, with_kwargs=True)
    measured = {}

    def timed_sample(*args, **kwargs):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = original_sample(*args, **kwargs)
        torch.cuda.synchronize(self.device)
        measured["seconds"] = time.perf_counter() - started
        prompt_length = len(kwargs["inputs"][0])
        ids = result.sequences[0, prompt_length:].tolist()
        eos = self.tokenizer.eos_token_id
        eos_found = eos in ids
        count = ids.index(eos) + 1 if eos_found else len(ids)
        measured.update(output_tokens_including_eos=count, stopped_on_eos=eos_found,
                        generated_block_tokens=len(ids))
        return result

    self.sampler.sample = timed_sample
    answers = []
    started = time.perf_counter()
    try:
        with path.open("x", encoding="utf-8", buffering=1) as stream:
            for index, request in enumerate(requests):
                response = original_generate_until(self, [request])[0]
                record = dict(measured, doc_id=request.doc_id, rank=self.rank,
                              warmup=index == 0, response=response)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                answers.append(response)
                if self.rank == 0 and (index + 1) % 10 == 0:
                    print(f"THROUGHPUT_PROGRESS {index + 1}/{len(requests)} "
                          f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
    finally:
        self.sampler.sample = original_sample
        hook.remove()
    wall_seconds = time.perf_counter() - started
    if self.accelerator is not None:
        self.accelerator.wait_for_everyone()
    if self.rank == 0:
        rows = [json.loads(line) for file in sorted(output_dir.glob("generations-rank*.jsonl"))
                for line in file.read_text().splitlines()]
        unique = {row["doc_id"]: row for row in rows}
        timed = [row for row in unique.values() if not row["warmup"]]
        tokens = sum(row["output_tokens_including_eos"] for row in timed)
        seconds = sum(row["seconds"] for row in timed)
        summary = {
            "samples": len(unique), "timed_samples_excluding_first_per_rank": len(timed),
            "output_tokens_including_eos": tokens,
            "summed_sampler_seconds": seconds,
            "tokens_per_second_per_gpu": tokens / seconds,
            "samples_per_second_per_gpu": len(timed) / seconds,
            "mean_generation_seconds": seconds / len(timed),
            "generation_wall_seconds_including_warmup_and_rank_waits": wall_seconds,
            "aggregate_output_tokens_per_wall_second": sum(
                row["output_tokens_including_eos"] for row in unique.values()) / wall_seconds,
            "eos_stopped_samples": sum(row["stopped_on_eos"] for row in unique.values()),
            "world_size": self.world_size,
            "timing_scope": "Synchronized sampler calls; excludes loading, scoring, and rank waits. GPUs shared.",
        }
        (output_dir / "throughput.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    return answers


base.BaseEvalHarness.generate_until = generate_with_throughput
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "dllm/pipelines/a2d/eval.py"),
                  run_name="__main__")
