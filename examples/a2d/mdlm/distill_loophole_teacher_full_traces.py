"""Distill a frozen Loophole A2D teacher into a non-Loophole A2D student.

Run after the Loophole teacher checkpoint is complete, for example:

    source /home/atharvchagi/.bashrc
    source /nvme-data2/atharvchagi/miniforge/etc/profile.d/conda.sh
    conda activate /nvme-data2/atharvchagi/miniforge/envs/dllm
    CUDA_VISIBLE_DEVICES=6,7 accelerate launch --mixed_precision bf16 \
        --config_file /nvme-data2/atharvchagi/dllm_fork/scripts/accelerate_configs/zero2.yaml \
        --num_processes 2 \
        /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/distill_loophole_teacher_full_traces.py

The teacher is frozen and uses its two-pass Loophole self-conditioning path;
the student is trained with forward KL on the same masked diffusion inputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import torch
import transformers

import dllm


DEFAULT_DATASET = "/nvme-data/atharvchagi/mix_60k_4096_qwen3_4b_full"
DEFAULT_STUDENT = (
    "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/"
    "Qwen3-0.6B-a2d-init/mdlm-opdlm-baseline-epoch5-len512-mask151669/checkpoint-final"
)
DEFAULT_TEACHER = (
    "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/"
    "Qwen3-0.6B-a2d-init/mdlm-opdlm-loophole-full-traces/checkpoint-final"
)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = DEFAULT_STUDENT
    teacher_model_name_or_path: str = DEFAULT_TEACHER
    teacher_self_cond_rate: float = 1.0


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = DEFAULT_DATASET
    load_preprocessed_data: bool = True
    max_length: int = 4096
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = (
        "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/"
        "Qwen3-0.6B-a2d-init/mdlm-opdlm-nonloophole-kl-full-traces"
    )
    loss_type: str = "KL"
    distillation_policy: str = "off_policy"
    kl_divergence_type: str = "forward"
    kl_weight: float = 1.0
    loophole_enabled: bool = False
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    bf16: bool = True
    group_by_length: bool = True
    eval_strategy: str = "no"
    save_strategy: str = "steps"
    save_steps: int = 1000
    logging_steps: int = 10


def _user_message(row):
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return {"role": "user", "content": message["content"]}
    return {"role": "user", "content": row["prompt"]}


def expand_and_tokenize(batch, *, tokenizer, mask_prompt_loss):
    """Expand every response variant into one full SFT sequence."""
    result = {"input_ids": [], "labels": [], "prompt_len": []}
    for i in range(len(batch["responses"])):
        row = {key: values[i] for key, values in batch.items()}
        for response in row["responses"]:
            if not isinstance(response, str) or not response.strip():
                continue
            messages = [_user_message(row), {"role": "assistant", "content": response}]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            labels = list(tokens)
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:1], tokenize=True, add_generation_prompt=True
            )
            prompt_len = min(len(prompt_tokens), len(labels))
            if mask_prompt_loss:
                labels[:prompt_len] = [-100] * prompt_len
            result["input_ids"].append(tokens)
            result["labels"].append(labels)
            result["prompt_len"].append(prompt_len)
    return result


class FrozenLoopholeTeacher(torch.nn.Module):
    """Frozen two-pass Loophole teacher returning logits for clean inputs."""

    def __init__(self, model, self_cond_rate: float = 1.0):
        super().__init__()
        self.model = model
        self.self_cond_rate = self_cond_rate
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, input_ids, attention_mask=None):
        use_self_conditioning = self.self_cond_rate >= 1.0 or (
            self.self_cond_rate > 0.0
            and bool(torch.rand((), device=input_ids.device) < self.self_cond_rate)
        )
        state = None
        if use_self_conditioning:
            with torch.no_grad():
                pseudo = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loophole_state=None,
                    return_loophole_state=True,
                    loophole_enabled=True,
                    use_cache=False,
                )
            state = getattr(pseudo, "loophole_state", None)
            if state is None:
                raise RuntimeError("Teacher did not return a Loophole state")
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loophole_state=state,
            return_loophole_state=True,
            loophole_enabled=True,
            use_cache=False,
        )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    student_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    student_config.loophole_enabled = False
    student = dllm.utils.get_model(model_args=model_args, config=student_config)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    teacher_config = transformers.AutoConfig.from_pretrained(
        model_args.teacher_model_name_or_path
    )
    teacher_config.loophole_enabled = True
    teacher_model = dllm.utils.get_model(
        model_name_or_path=model_args.teacher_model_name_or_path,
        config=teacher_config,
        dtype="bfloat16",
    )
    teacher_model.to(accelerate.PartialState().device)
    teacher = FrozenLoopholeTeacher(teacher_model, model_args.teacher_self_cond_rate)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        dataset = dataset.map(
            partial(
                expand_and_tokenize,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            ),
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=data_args.num_proc,
            desc="Expanding all SFT responses for KL distillation",
        )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    trainer = dllm.core.trainers.MDLMTrainer(
        model=student,
        ar_model=teacher,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=None,
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id,
            )
        ),
    )
    trainer.train()
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)


if __name__ == "__main__":
    train()
