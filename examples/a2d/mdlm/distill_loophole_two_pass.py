"""Distill a two-step Loopholing teacher into an ordinary Qwen3 MDLM.

Source ~/.zshrc and activate conda env dllm, then run:
    srun -p "$PARTITION" --quotatype="$QUOTATYPE" --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/distill_loophole_two_pass.py --teacher_model_name_or_path /absolute/path/to/teacher

Use --load_preprocessed_data true for datasets with input_ids and labels (-100
on prompt tokens). Otherwise the standard SFT mapper prepares the dataset.
Rho is a gap in mask fraction: t is uniform on (0, 1-rho), and the initial
fraction is t+rho. Integer counts are clamped to reveal and retain >=1 token.
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


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "divelab/Qwen3-0.6B-a2d-init"
    teacher_model_name_or_path: str = ""
    rho: float = 0.25
    temperature: float = 1.0


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    load_preprocessed_data: bool = False
    max_length: int = 512


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/qwen3-two-pass-kl"
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
        self.rho = rho
        self.temperature = temperature
        self.mask_token_id = self.processing_class.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must define a mask token")
        # The loss already averages examples; Trainer must handle accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ids, attention = inputs["input_ids"], inputs["attention_mask"]
        eligible = (inputs["labels"] != -100) & attention.bool()
        initial, remaining = sample_transition_masks(eligible, self.rho)
        x_initial = ids.masked_fill(initial, self.mask_token_id)
        with torch.no_grad():
            self.teacher.eval()
            first = self.teacher(
                input_ids=x_initial, attention_mask=attention, use_cache=False,
                loophole_state=None, return_loophole_state=True, loophole_enabled=True,
            )
            state = first.loophole_state
            if state is None:
                raise RuntimeError("Teacher did not return its Loopholing state")
            # Never reveal a mask token, even if it has the highest teacher logit.
            first.logits[..., self.mask_token_id] = -torch.inf
            predictions = first.logits.argmax(-1)
            x_t = torch.where(initial & ~remaining, predictions, x_initial)
            del first
            second = self.teacher(
                input_ids=x_t, attention_mask=attention, use_cache=False,
                loophole_state=state, loophole_enabled=True,
            )
            teacher_logits = second.logits[remaining]
            del second, state
        student = model(input_ids=x_t, attention_mask=attention, use_cache=False)
        batch_indices = remaining.nonzero(as_tuple=True)[0]
        loss = forward_kl_loss(
            student.logits[remaining], teacher_logits, batch_indices,
            ids.shape[0], temperature=self.temperature,
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
                        "student_head": "trainable"}, indent=2) + "\n"
        )


if __name__ == "__main__":
    train()
