"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run through an eval entrypoint, for example:
``python /nvme-data2/atharvchagi/dllm_fork/dllm/pipelines/a2d/eval.py --help``.
"""

import dataclasses
import math
from dataclasses import dataclass

import accelerate
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

import dllm
from dllm.core.samplers import BaseSampler, BaseSamplerConfig
from dllm.utils.configs import ModelArguments


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: device and batch_size."""

    pretrained: str = ""
    device: str = "cuda"
    batch_size: int = 1

    def get_model_config(self, pretrained: str):
        """Optional: return custom model config for loading. Default None (use checkpoint config)."""
        return None


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood (and optionally loglikelihood_rolling);
    generate_until is implemented here and uses sampler + sampler_config.
    """

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        """Build a dataclass *config_cls* by copying fields from *source*, with *kwargs* overrides."""
        init = {}
        for f in dataclasses.fields(config_cls):
            if f.name in kwargs:
                init[f.name] = kwargs[f.name]
            elif hasattr(source, f.name):
                init[f.name] = getattr(source, f.name)
        return config_cls(**init)

    def __init__(
        self,
        eval_config: BaseEvalConfig | None = None,
        model_args: ModelArguments | None = None,
        sampler_config: BaseSamplerConfig | None = None,
        sampler_cls: type[BaseSampler] | None = None,
        **kwargs,
    ):
        super().__init__()
        eval_config = eval_config or BaseEvalConfig()
        # Ensure model path is in kwargs and we have a safe default for ModelArguments(__post_init__).
        model_args = model_args or ModelArguments(
            model_name_or_path=kwargs.get("pretrained")
        )
        device = kwargs.get("device", eval_config.device)

        # ── Distributed ──────────────────────────────────────────
        accelerator = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = dllm.utils.get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = dllm.utils.get_tokenizer(self.model_args)
        if sampler_config is not None:
            self.sampler_config = self._build_config(
                type(sampler_config), sampler_config, kwargs
            )
        if sampler_cls is not None:
            self.sampler = sampler_cls(model=self.model, tokenizer=self.tokenizer)

        # ── Device placement ─────────────────────────────────────
        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))
        self.enable_thinking = kwargs.get("enable_thinking", None)
        self._generation_nfe_records: list[dict[str, object]] = []
        self._generation_nfe_batches: list[dict[str, int]] = []
        self._generation_request_occurrences: dict[tuple[object, ...], int] = {}

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format chat history for input to the LM."""
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking

        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **template_kwargs,
        )

    # ── Unified generate_until scaffolding ────────────────────────────

    def _record_generation_nfe(
        self,
        batch: list[Instance],
        prompts: list[torch.Tensor],
        generation_stats: dict[str, object] | None,
    ) -> None:
        """Record sampler NFE for one lm-eval generation batch."""
        if generation_stats is None:
            return

        per_sequence_keys = {
            "active_denoising_iterations": (
                "per_sequence_active_denoising_iterations"
            ),
            "prefix_nfe": "per_sequence_prefix_nfe",
            "denoising_nfe": "per_sequence_denoising_nfe",
            "total_nfe": "per_sequence_total_nfe",
        }
        per_sequence_values = {}
        for output_key, stats_key in per_sequence_keys.items():
            values = generation_stats.get(stats_key)
            if not isinstance(values, list) or len(values) != len(batch):
                raise RuntimeError(
                    f"Sampler NFE field {stats_key!r} must contain one value "
                    f"per request (expected {len(batch)})"
                )
            per_sequence_values[output_key] = values

        for sequence_index, (instance, prompt) in enumerate(zip(batch, prompts)):
            request_key = (instance.task_name, instance.doc_id, instance.idx)
            repeat_ordinal = self._generation_request_occurrences.get(request_key, 0)
            self._generation_request_occurrences[request_key] = repeat_ordinal + 1
            expected_repeats = max(int(instance.repeats or 1), 1)

            record = {
                "task_name": instance.task_name,
                "doc_id": instance.doc_id,
                "request_index": instance.idx,
                "repeat_ordinal": repeat_ordinal,
                "is_distributed_padding": repeat_ordinal >= expected_repeats,
                "prompt_tokens": int(prompt.numel()),
                "generated_blocks": int(generation_stats["generated_blocks"]),
            }
            for output_key, values in per_sequence_values.items():
                record[output_key] = int(values[sequence_index])
            self._generation_nfe_records.append(record)

        self._generation_nfe_batches.append(
            {
                "batch_size": len(batch),
                "prefix_model_forward_calls": int(
                    generation_stats["prefix_model_forward_calls"]
                ),
                "denoising_model_forward_calls": int(
                    generation_stats["denoising_model_forward_calls"]
                ),
                "total_model_forward_calls": int(
                    generation_stats["total_model_forward_calls"]
                ),
            }
        )

    @staticmethod
    def _summarize_nfe(values: list[int]) -> dict[str, int | float]:
        """Return deterministic descriptive statistics for integer NFE values."""
        ordered = sorted(values)
        count = len(ordered)
        if count == 0:
            return {
                "count": 0,
                "sum": 0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "min": 0,
                "max": 0,
            }

        midpoint = count // 2
        if count % 2:
            median = float(ordered[midpoint])
        else:
            median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

        p95_position = 0.95 * (count - 1)
        p95_lower = math.floor(p95_position)
        p95_upper = math.ceil(p95_position)
        p95_fraction = p95_position - p95_lower
        p95 = ordered[p95_lower] + p95_fraction * (
            ordered[p95_upper] - ordered[p95_lower]
        )
        total = sum(ordered)
        return {
            "count": count,
            "sum": total,
            "mean": total / count,
            "median": median,
            "p95": p95,
            "min": ordered[0],
            "max": ordered[-1],
        }

    def get_evaluation_metadata(self) -> dict[str, object]:
        """Gather and aggregate recorded generation NFE across eval ranks."""
        local_payload = {
            "rank": self.rank,
            "records": self._generation_nfe_records,
            "batches": self._generation_nfe_batches,
        }
        if self.world_size > 1:
            gathered_payloads = [None] * self.world_size if self.rank == 0 else None
            torch.distributed.gather_object(
                obj=local_payload,
                object_gather_list=gathered_payloads,
                dst=0,
            )
            if self.rank != 0:
                return {}
        else:
            gathered_payloads = [local_payload]

        all_records = []
        all_batches = []
        per_rank_forward_calls = []
        for payload in gathered_payloads:
            records = payload["records"]
            batches = payload["batches"]
            all_records.extend(records)
            all_batches.extend(batches)
            per_rank_forward_calls.append(
                {
                    "rank": payload["rank"],
                    "batches": len(batches),
                    "prefix": sum(
                        batch["prefix_model_forward_calls"] for batch in batches
                    ),
                    "denoising": sum(
                        batch["denoising_model_forward_calls"] for batch in batches
                    ),
                    "total": sum(
                        batch["total_model_forward_calls"] for batch in batches
                    ),
                }
            )

        if not all_records:
            return {}

        padding_records = [
            record for record in all_records if record["is_distributed_padding"]
        ]
        unique_records = []
        seen_request_keys = set()
        duplicate_records = 0
        for record in all_records:
            if record["is_distributed_padding"]:
                continue
            request_key = (
                record["task_name"],
                record["doc_id"],
                record["request_index"],
                record["repeat_ordinal"],
            )
            if request_key in seen_request_keys:
                duplicate_records += 1
                continue
            seen_request_keys.add(request_key)
            unique_records.append(record)

        unique_records.sort(
            key=lambda record: (
                str(record["task_name"]),
                -1 if record["doc_id"] is None else int(record["doc_id"]),
                int(record["request_index"]),
                int(record["repeat_ordinal"]),
            )
        )

        summary_fields = (
            "active_denoising_iterations",
            "prefix_nfe",
            "denoising_nfe",
            "total_nfe",
        )
        per_sample_summary = {
            field: self._summarize_nfe(
                [int(record[field]) for record in unique_records]
            )
            for field in summary_fields
        }
        physical_forward_calls = {
            "batches": len(all_batches),
            "prefix": sum(
                batch["prefix_model_forward_calls"] for batch in all_batches
            ),
            "denoising": sum(
                batch["denoising_model_forward_calls"] for batch in all_batches
            ),
            "total": sum(
                batch["total_model_forward_calls"] for batch in all_batches
            ),
            "per_rank": sorted(
                per_rank_forward_calls, key=lambda rank_stats: rank_stats["rank"]
            ),
        }

        return {
            "generation_nfe": {
                "schema_version": 1,
                "sample_count": len(unique_records),
                "discarded_distributed_padding_requests": len(padding_records),
                "discarded_duplicate_requests": duplicate_records,
                "definitions": {
                    "active_denoising_iterations": (
                        "Dynamic decoding iterations while that sequence still had "
                        "masked tokens; CFG branches are not multiplied."
                    ),
                    "denoising_nfe": (
                        "Model evaluations during denoising. Separate conditional "
                        "and unconditional CFG calls each count as one."
                    ),
                    "prefix_nfe": (
                        "Model evaluations used to build the prefix cache, normally "
                        "one per generated block (two with CFG)."
                    ),
                    "total_nfe": "denoising_nfe + prefix_nfe.",
                    "physical_model_forward_calls_all_ranks": (
                        "Actual batched model.forward invocations summed across all "
                        "distributed workers, including padding requests."
                    ),
                },
                "per_sample_summary": per_sample_summary,
                "physical_model_forward_calls_all_ranks": physical_forward_calls,
                "samples": unique_records,
            }
        }

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        out: list[str] = []

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size), desc="Generating..."
        ):
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

            prompts = [
                torch.tensor(
                    self.tokenizer(ctx)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                for ctx in contexts
            ]

            sample_output = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=True,
                return_history=False,
            )
            generated_ids = sample_output.sequences
            self._record_generation_nfe(
                batch=batch,
                prompts=prompts,
                generation_stats=sample_output.generation_stats,
            )
            generated_answers = dllm.utils.sample_trim(
                self.tokenizer,
                generated_ids.tolist(),
                [p.tolist() for p in prompts],
            )

            for answer, gen_kwargs in zip(generated_answers, gen_kwargs_list):
                for stop_seq in gen_kwargs["until"]:
                    if stop_seq in answer:
                        answer = answer.split(stop_seq)[0]
                out.append(answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
