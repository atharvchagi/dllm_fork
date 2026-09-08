"""Train a non-Loophole Qwen3 DLM from frozen teacher trajectory targets.

Run after activating the ``dllm`` environment, for example:
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/distill_off_policy.py --objective hidden_mse --output_dir /nvme-data2/atharvchagi/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init/mdlm-opdlm-offpolicy-hidden-mse``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import transformers

import dllm
from dllm.data.offpolicy_distillation import (
    OfflineTraceCollator,
    OfflineTraceDataset,
    TEACHER_HEAD_FILENAME,
    forward_kl_loss,
    hidden_mse_loss,
    install_frozen_teacher_head,
    load_teacher_head,
)


DEFAULT_TRACE_DIR = (
    "/nvme-data/atharvchagi/qwen3_0.6b_loophole_offpolicy"
)
DEFAULT_OUTPUT = (
    "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/Qwen3-0.6B-a2d-init/"
    "mdlm-opdlm-offpolicy-hidden-mse"
)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    """Student initialization and offline objective settings."""

    model_name_or_path: str = "divelab/Qwen3-0.6B-a2d-init"
    objective: Literal["hidden_mse", "forward_kl"] = "hidden_mse"
    temperature: float = 1.0


@dataclass
class DataArguments:
    """Location of the finalized frozen teacher traces."""

    trace_dir: str = DEFAULT_TRACE_DIR
    trace_num_workers: int = 0


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    """Fixed primary-experiment optimization settings."""

    output_dir: str = DEFAULT_OUTPUT
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    tf32: bool = True
    seed: int = 42
    eval_strategy: str = "steps"
    eval_steps: int = 250
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int = 2
    logging_steps: int = 10
    prediction_loss_only: bool = True
    remove_unused_columns: bool = False


class OfflineDistillationTrainer(transformers.Trainer):
    """Trainer whose only target is a frozen hidden state or distribution."""

    def __init__(
        self,
        *,
        objective: str,
        temperature: float,
        **kwargs: Any,
    ) -> None:
        if objective not in {"hidden_mse", "forward_kl"}:
            raise ValueError(f"Unsupported objective: {objective}")
        if temperature != 1.0:
            raise ValueError("The primary experiment fixes KL temperature at 1.0")
        self.objective = objective
        self.temperature = temperature
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ):
        """Compute a pure offline target loss with no CE or new corruption."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        target_batch_indices = inputs["target_batch_indices"]
        target_positions = inputs["target_positions"]
        teacher_hidden = inputs["teacher_hidden"]
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        student_hidden = outputs.last_hidden_state[
            target_batch_indices, target_positions
        ]
        batch_size = input_ids.shape[0]
        if self.objective == "hidden_mse":
            loss = hidden_mse_loss(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                target_batch_indices=target_batch_indices,
                batch_size=batch_size,
            )
        else:
            student_logits = model.lm_head(student_hidden)
            with torch.no_grad():
                teacher_logits = model.lm_head(
                    teacher_hidden.to(
                        device=student_hidden.device,
                        dtype=student_hidden.dtype,
                    )
                )
            loss = forward_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                target_batch_indices=target_batch_indices,
                batch_size=batch_size,
                temperature=self.temperature,
            )
        if not return_outputs:
            return loss
        return loss, {"loss": loss.detach()}


def validate_student(model, teacher_weight: torch.Tensor) -> None:
    """Assert the experiment's untied, frozen-readout student contract."""
    if bool(getattr(model.config, "loophole_enabled", False)):
        raise ValueError("Offline students must not contain a Loopholing adapter")
    if bool(model.config.tie_word_embeddings):
        raise ValueError("Student readout must be untied from input embeddings")
    if model.lm_head.weight.requires_grad:
        raise ValueError("Teacher readout must be frozen")
    if model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        raise ValueError("Student readout is still tied to input embeddings")
    if not model.get_input_embeddings().weight.requires_grad:
        raise ValueError("Student input embeddings must remain trainable")
    if not torch.equal(
        model.lm_head.weight.detach().cpu().to(teacher_weight.dtype),
        teacher_weight,
    ):
        raise ValueError("Installed student readout differs from teacher artifact")


def train() -> None:
    """Load frozen traces, construct one student arm, train, and save it."""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.temperature != 1.0:
        raise ValueError("This experiment fixes temperature=1.0")
    trace_dir = Path(data_args.trace_dir)
    manifest_path = trace_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Generate and finalize offline traces before training: {manifest_path}"
        )
    dllm.utils.print_args_main(model_args, data_args, training_args)
    transformers.set_seed(training_args.seed)
    dllm.utils.disable_caching_allocator_warmup()

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    if not isinstance(config, dllm.pipelines.a2d.A2DQwen3Config):
        raise ValueError("Student initialization must be an A2D Qwen3 checkpoint")
    if bool(getattr(config, "loophole_enabled", False)):
        raise ValueError("Student initialization must be the non-Loophole A2D model")
    config.loophole_enabled = False
    model = dllm.utils.get_model(model_args=model_args, config=config)
    teacher_weight = load_teacher_head(trace_dir / TEACHER_HEAD_FILENAME)
    install_frozen_teacher_head(model, teacher_weight)
    validate_student(model, teacher_weight)
    model.config.use_cache = False

    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    train_dataset = OfflineTraceDataset(trace_dir, split="train")
    eval_dataset = OfflineTraceDataset(trace_dir, split="validation")
    training_args.dataloader_num_workers = data_args.trace_num_workers
    trainer = OfflineDistillationTrainer(
        objective=model_args.objective,
        temperature=model_args.temperature,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=OfflineTraceCollator(tokenizer.pad_token_id),
        args=training_args,
    )
    trainer.train(
        resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None)
    )
    final_dir = Path(training_args.output_dir) / "checkpoint-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    provenance = {
        "objective": model_args.objective,
        "temperature": model_args.temperature,
        "pure_distillation": True,
        "cross_entropy_weight": 0.0,
        "student_initialization": model_args.model_name_or_path,
        "trace_manifest": str(manifest_path.resolve()),
        "teacher_head": str((trace_dir / TEACHER_HEAD_FILENAME).resolve()),
        "teacher_head_frozen": True,
        "input_embeddings_trainable": True,
        "tie_word_embeddings": False,
        "model_arguments": asdict(model_args),
    }
    if trainer.is_world_process_zero():
        os.makedirs(final_dir, exist_ok=True)
        (final_dir / "distillation_config.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    train()
