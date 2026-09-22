"""Distill a two-step Loopholing teacher into an ordinary Qwen3 BD3LM.

Source ~/.zshrc and activate conda env dllm, then run:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/distill_loophole_two_pass.py --teacher_model_name_or_path /absolute/path/to/teacher

Use --load_preprocessed_data true for datasets with input_ids and labels (-100
on prompt tokens). Otherwise the standard SFT mapper prepares the dataset.
Rho is a gap in mask fraction: t is uniform on (0, 1-rho), and the initial
fraction is t+rho. Teacher and student use the same block size.
The concatenated [x_t, x_0] streams use BD3LM block attention, duplicated
position IDs, and optional AR right shifting, matching BD3LM training.
The clean stream provides earlier-block context, never same-block targets.
Integer counts are clamped to reveal and retain >=1 token per example.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import accelerate
import torch
import transformers

import dllm
from dllm.data.offpolicy_distillation import forward_kl_loss
from dllm.core.trainers.bd3lm import BD3LMTrainer, _create_bd3lm_attention_mask


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "divelab/Qwen3-0.6B-a2d-init"
    teacher_model_name_or_path: str = ""
    attn_implementation: str = "sdpa"
    rho: float = 0.25
    temperature: float = 1.0


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    load_preprocessed_data: bool = False
    max_length: int = 1024


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    block_size: int = 4
    right_shift_logits: bool = True
    output_dir: str = "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/qwen3-bd3lm-two-pass-kl"
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    eval_strategy: str = "no"
    save_steps: float = 250
    save_only_model: bool = False
    save_total_limit: int = 2
    remove_unused_columns: bool = False
    prediction_loss_only: bool = True
    report_to: str = "none"
    resume_from_checkpoint: str | None = None


def sample_transition_masks(eligible: torch.Tensor, rho: float):
    """Choose nested random masks, with >=1 revealed and >=1 retained per row."""
    if not 0 < rho < 1:
        raise ValueError("rho must lie strictly between zero and one")
    counts = eligible.sum(-1)
    if torch.any(counts < 2):
        raise ValueError("Each example needs at least two eligible response tokens")
    t = torch.rand(counts.shape, device=eligible.device) * (1 - rho)
    initial_counts = torch.ceil((t + rho) * counts).long().clamp(min=2)
    remaining_counts = torch.ceil(t * counts).long().clamp(min=1)
    remaining_counts = torch.minimum(remaining_counts, initial_counts - 1)
    # Sorting independent random scores gives a uniform ordering of valid positions.
    scores = torch.rand(eligible.shape, device=eligible.device).masked_fill(~eligible, 2)
    ranks = scores.argsort(-1).argsort(-1)
    initial = eligible & (ranks < initial_counts[:, None])
    remaining = eligible & (ranks < remaining_counts[:, None])
    return initial, remaining


class TwoPassDistillationTrainer(transformers.Trainer):
    """Online KL with a frozen recurrent teacher and a memoryless student."""

    def __init__(self, *, teacher, rho: float, temperature: float, **kwargs):
        if not 0 < rho < 1 or temperature <= 0:
            raise ValueError("Require 0 < rho < 1 and temperature > 0")
        super().__init__(**kwargs)
        self.teacher = teacher.to(self.accelerator.device).requires_grad_(False).eval()
        if self.args.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.right_shift_logits = self.args.right_shift_logits
        if self.processing_class.padding_side != "right":
            raise ValueError("BD3LM requires right padding")
        if self.right_shift_logits and self.processing_class.bos_token_id is None:
            raise ValueError("Right-shifted BD3LM requires a BOS token")
        self.rho = rho
        self.temperature = temperature
        self.mask_token_id = self.processing_class.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must define a mask token")
        # The loss already averages examples; Trainer must handle accumulation scaling.
        self.model_accepts_loss_kwargs = False

    # Match the existing BD3LM convention: shift each stream independently.
    _align_loophole_state_for_input = BD3LMTrainer._align_loophole_state_for_input
    _preprocess_inputs = BD3LMTrainer._preprocess_inputs

    def _shift_logits(self, logits):
        if self.right_shift_logits:
            return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = self._preprocess_inputs(dict(inputs))
        ids, padding = inputs["input_ids"], inputs["attention_mask"]
        eligible = (inputs["labels"] != -100) & padding.bool()
        initial, remaining = sample_transition_masks(eligible, self.rho)
        x_initial = ids.masked_fill(initial, self.mask_token_id)
        batch_size, length = ids.shape
        positions = torch.arange(length, device=ids.device).expand(batch_size, -1)
        position_ids = torch.cat([positions, positions], dim=1)
        indices = torch.arange(2 * length, device=ids.device)
        allowed = _create_bd3lm_attention_mask(
            None, None, indices[:, None], indices[None, :],
            block_size=self.args.block_size, n=length,
        )
        # Keep padding out of both clean and noisy context streams.
        valid_keys = torch.cat([padding, padding], dim=1).bool()
        allowed = allowed[None, None] & valid_keys[:, None, None, :]
        # Qwen's SDPA and eager implementations both accept an additive 4D mask.
        attention = torch.zeros(allowed.shape, dtype=next(model.parameters()).dtype, device=ids.device)
        attention.masked_fill_(~allowed, torch.finfo(attention.dtype).min)
        common = dict(attention_mask=attention, position_ids=position_ids,
                      use_cache=False, logits_to_keep=torch.arange(length, device=ids.device))
        with torch.no_grad():
            self.teacher.eval()
            first = self.teacher(
                input_ids=torch.cat([x_initial, ids], dim=1), **common,
                loophole_state=None, return_loophole_state=True, loophole_enabled=True,
            )
            if first.loophole_state is None:
                raise RuntimeError("Teacher did not return its Loopholing state")
            state = self._align_loophole_state_for_input(first.loophole_state)
            logits = self._shift_logits(first.logits)
            logits[..., self.mask_token_id] = -torch.inf
            x_t = torch.where(initial & ~remaining, logits.argmax(-1), x_initial)
            del first, logits
            second = self.teacher(
                input_ids=torch.cat([x_t, ids], dim=1), **common,
                loophole_state=state, loophole_enabled=True,
            )
            teacher_logits = self._shift_logits(second.logits)[remaining]
            del second, state
        student = model(input_ids=torch.cat([x_t, ids], dim=1), **common,
                        loophole_enabled=False)
        batch_indices = remaining.nonzero(as_tuple=True)[0]
        loss = forward_kl_loss(
            self._shift_logits(student.logits)[remaining], teacher_logits,
            batch_indices, batch_size, temperature=self.temperature,
        )
        return (loss, {"loss": loss.detach()}) if return_outputs else loss


def train():
    """Load aligned Qwen3 models and SFT data, train the student, and save it."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not model_args.teacher_model_name_or_path:
        raise ValueError("Supply --teacher_model_name_or_path with a trained Loopholing checkpoint")
    if not 0 < model_args.rho < 1 or model_args.temperature <= 0:
        raise ValueError("Require 0 < rho < 1 and temperature > 0")
    if model_args.attn_implementation not in {"sdpa", "eager"}:
        raise ValueError("This script's dense block mask requires sdpa or eager attention")
    if training_args.block_size <= 0:
        raise ValueError("block_size must be positive")
    if model_args.lora or model_args.load_in_4bit:
        raise ValueError("This script currently supports full-weight, non-quantized students")
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    student_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    teacher_config = transformers.AutoConfig.from_pretrained(model_args.teacher_model_name_or_path)
    for config in (student_config, teacher_config):
        if not isinstance(config, dllm.pipelines.a2d.A2DQwen3Config):
            raise ValueError("Both models must be A2D Qwen3 checkpoints")
    if not teacher_config.loophole_enabled:
        raise ValueError("Teacher checkpoint must already contain a trained Loopholing adapter")
    if student_config.loophole_enabled:
        raise ValueError("Student initialization must be a non-Loopholing checkpoint")
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    teacher_tokenizer = dllm.utils.get_tokenizer(model_name_or_path=model_args.teacher_model_name_or_path)
    if (tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
            or tokenizer.mask_token_id != teacher_tokenizer.mask_token_id
            or student_config.vocab_size != teacher_config.vocab_size):
        raise ValueError("Teacher and student must have identical token ID mappings and vocabulary sizes")
    student = dllm.utils.get_model(model_args=model_args, config=student_config)
    student.config.bd3lm_block_size = training_args.block_size
    student.config.right_shift_logits = training_args.right_shift_logits
    teacher = dllm.utils.get_model(
        model_name_or_path=model_args.teacher_model_name_or_path,
        config=teacher_config, dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args, load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            dataset = dataset.map(
                partial(dllm.utils.default_sft_map_fn, tokenizer=tokenizer, mask_prompt_loss=True),
                num_proc=data_args.num_proc,
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)
        # Drop metadata before collation and examples too short for a transition.
        dataset = dataset.select_columns(["input_ids", "labels"])
        dataset = dataset.filter(lambda row: sum(x != -100 for x in row["labels"]) >= 2)
    if not len(dataset["train"]):
        raise ValueError("No training examples retain two response tokens after preprocessing")
    trainer = TwoPassDistillationTrainer(
        teacher=teacher, rho=model_args.rho, temperature=model_args.temperature,
        model=student, processing_class=tokenizer, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset.get("test"),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt",
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    final_dir = Path(training_args.output_dir) / "checkpoint-final"
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        (final_dir / "distillation_config.json").write_text(
            json.dumps({"model": asdict(model_args), "data": asdict(data_args),
                        "reveal_tokens": "teacher_greedy", "loss": "forward_kl",
                        "student_head": "trainable", "block_size": training_args.block_size,
                        "right_shift_logits": training_args.right_shift_logits}, indent=2) + "\n"
        )


if __name__ == "__main__":
    train()
