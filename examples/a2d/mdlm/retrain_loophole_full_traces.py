"""Retrain an A2D Qwen3 Loophole model on every trace in a trace dataset.

Run after activating the project environment, for example:

    source /home/atharvchagi/.bashrc
    source /nvme-data2/atharvchagi/miniforge/etc/profile.d/conda.sh
    conda activate /nvme-data2/atharvchagi/miniforge/envs/dllm
    CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
        --config_file /nvme-data2/atharvchagi/dllm_fork/scripts/accelerate_configs/zero2.yaml \
        --num_processes 2 \
        /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/retrain_loophole_full_traces.py

The response list is expanded before tokenization, so all responses are used.
"""

import json
import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = (
        "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/"
        "Qwen3-0.6B-a2d-init/mdlm-opdlm-loophole-epoch5-len512-mask151669/checkpoint-final"
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = (
        "Jotsna22/mix_60k_4096_Qwen3-4B_chunk10000_ntokens4096_greedy"
    )
    max_length: int = 4096
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask prompt tokens in the training loss"},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = (
        "/nvme-data2/atharvchagi/dllm_fork/.models/a2d/"
        "Qwen3-0.6B-a2d-init/mdlm-opdlm-loophole-full-traces"
    )
    loophole_enabled: bool = True
    group_by_length: bool = True
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    save_strategy: str = "steps"
    save_steps: int = 1000
    logging_steps: int = 10
    eval_strategy: str = "no"


def _messages(row):
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None
    if isinstance(messages, list):
        user_messages = [
            m for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_messages and isinstance(user_messages[0].get("content"), str):
            return [{"role": "user", "content": user_messages[0]["content"]}]
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Trace row has neither a usable user message nor prompt")
    return [{"role": "user", "content": prompt}]


def expand_and_tokenize(batch, *, tokenizer, mask_prompt_loss):
    """Expand each row's responses and tokenize every full prompt/trace pair."""
    input_ids, labels, prompt_lens = [], [], []
    for i, row_messages in enumerate(batch["messages"]):
        row = {key: values[i] for key, values in batch.items()}
        responses = row.get("responses")
        if not isinstance(responses, list) or not responses:
            continue
        for response in responses:
            if not isinstance(response, str) or not response.strip():
                continue
            messages = _messages(row) + [{"role": "assistant", "content": response}]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            labels_row = list(tokens)
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True
            )
            prompt_len = min(len(prompt_tokens), len(labels_row))
            if mask_prompt_loss:
                labels_row[:prompt_len] = [-100] * prompt_len
            input_ids.append(tokens)
            labels.append(labels_row)
            prompt_lens.append(prompt_len)
    return {"input_ids": input_ids, "labels": labels, "prompt_len": prompt_lens}


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    if not isinstance(model_config, dllm.pipelines.a2d.A2DQwen3Config):
        raise ValueError("Full-trace Loophole retraining requires an A2D Qwen3 checkpoint")
    model_config.loophole_enabled = True
    model = dllm.utils.get_model(model_args=model_args, config=model_config)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        map_fn = partial(
            expand_and_tokenize,
            tokenizer=tokenizer,
            mask_prompt_loss=data_args.mask_prompt_loss,
        )
        dataset = dataset.map(
            map_fn,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=data_args.num_proc,
            desc="Expanding and tokenizing all trace responses",
        )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
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
