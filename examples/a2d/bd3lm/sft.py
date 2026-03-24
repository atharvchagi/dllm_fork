"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/a2d/bd3lm/sft.py

- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/a2d/bd3lm/sft.py

Slurm users
# Note: run `mkdir .logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/sft.py"

- 2 Nodes, 16 GPUs (ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    eval_dataset_args: str | None = None
    max_length: int = 512
    load_preprocessed_data: bool = False
    eval_load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    eval_mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask prompt loss for eval dataset mapping"},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = ".models/a2d/Qwen3-0.6B/bd3lm/alpaca"
    group_by_length: bool = True
    num_train_epochs: int = 20
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    # bd3lm
    block_size: int = 32


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    def _map_and_postprocess(ds, *, mask_prompt_loss: bool):
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=mask_prompt_loss,
            )
            ds = ds.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        return dllm.utils.post_process_dataset(ds, data_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        dataset = _map_and_postprocess(ds=dataset, mask_prompt_loss=data_args.mask_prompt_loss)

        eval_dataset = dataset.get("test", None)
        if data_args.eval_dataset_args:
            eval_raw = dllm.data.load_sft_dataset(
                data_args.eval_dataset_args,
                load_preprocessed_data=data_args.eval_load_preprocessed_data,
            )
            if not data_args.eval_load_preprocessed_data:
                map_fn_eval = partial(
                    dllm.utils.default_sft_map_fn,
                    tokenizer=tokenizer,
                    mask_prompt_loss=data_args.eval_mask_prompt_loss,
                )
                eval_raw = eval_raw.map(
                    map_fn_eval,
                    num_proc=data_args.num_proc,
                    desc="Mapping eval dataset to SFT format",
                )
            eval_raw = dllm.utils.post_process_dataset(eval_raw, data_args)
            eval_dataset = eval_raw.get("test", eval_raw.get("validation", None))

    eval_strategy = getattr(training_args, "eval_strategy", "no")
    eval_strategy = getattr(eval_strategy, "value", str(eval_strategy)).lower()
    if eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            "Evaluation is enabled but no eval split was found. "
            "Provide --eval_dataset_args with a dataset containing test/validation, "
            "or set --eval_strategy no."
        )

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                block_size=training_args.block_size,
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
