"""Run the A2D evaluation CLI while saving generations after every batch.

Set LIVE_GENERATIONS_DIR to an absolute output directory and launch this file
with Accelerate and the same flags as /nvme-data2/atharvchagi/dllm_fork/dllm/pipelines/a2d/eval.py.
"""

import json
import os
from pathlib import Path
import runpy
import time

import torch

from dllm.core.eval import base


original_generate_until = base.BaseEvalHarness.generate_until
# The outer loop below reports progress; avoid one progress bar per batch.
base.tqdm = lambda iterable, **kwargs: iterable


def keep_last_prefix_logit(model, args, kwargs):
    """BD3LM only consumes the last prefix logit; avoid allocating all of them."""
    if kwargs.get('use_cache') is True:
        kwargs['logits_to_keep'] = 1
    return args, kwargs


def generate_with_live_output(self, requests):
    """Preserve harness decoding and write each completed batch to a rank-local file."""
    output_dir = Path(os.environ['LIVE_GENERATIONS_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'generations-rank{self.rank}.jsonl'
    saved = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    replay = torch.tensor(len(saved), device=self.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(replay, op=torch.distributed.ReduceOp.MIN)
    replay_count = int(replay.item())
    if replay_count % self.batch_size:
        raise ValueError('Saved generation count must align with batch size')
    for request, row in zip(requests[:replay_count], saved[:replay_count]):
        if request.doc_id != row['doc_id'] or request.args[0] != row['prompt']:
            raise ValueError('Saved generations do not match this evaluation request')
    if not getattr(self, '_prefix_logit_hook_installed', False):
        self.model.register_forward_pre_hook(keep_last_prefix_logit, with_kwargs=True)
        self._prefix_logit_hook_installed = True
    answers = []
    started = time.monotonic()
    with path.open('w', encoding='utf-8', buffering=1) as stream:
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start:start + self.batch_size]
            if start < replay_count:
                responses = [row['response'] for row in saved[start:start + self.batch_size]]
            else:
                responses = original_generate_until(self, batch)
            for request, response in zip(batch, responses):
                stream.write(json.dumps({
                    'doc_id': request.doc_id, 'doc': request.doc,
                    'prompt': request.args[0], 'response': response,
                }, ensure_ascii=False) + '\n')
            answers.extend(responses)
            if self.rank == 0:
                print(f'LIVE_GENERATIONS {len(answers)}/{len(requests)} '
                      f'elapsed_seconds={time.monotonic() - started:.1f}', flush=True)
    return answers


base.BaseEvalHarness.generate_until = generate_with_live_output
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[3] / 'dllm/pipelines/a2d/eval.py'),
                   run_name='__main__')
