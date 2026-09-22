"""Train a BD3LM model with SFT data.

Run ``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/sft.py --help``
after activating the ``dllm`` conda environment.
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import torch
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B"
    teacher_model_name_or_path: str | None = None
    teacher_dtype: str = "bfloat16"
    teacher_load_in_4bit: bool = False
    teacher_attn_implementation: str | None = None


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
    student_config = None
    if training_args.loophole_enabled:
        student_config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path
        )
        if not isinstance(student_config, dllm.pipelines.a2d.A2DQwen3Config):
            raise ValueError(
                "--loophole_enabled currently requires an A2D Qwen3 checkpoint"
            )
        # This adds a zero-initialized Loophole adapter when the initialization
        # checkpoint is the ordinary qwen-a2d model.
        student_config.loophole_enabled = True
    model = dllm.utils.get_model(model_args=model_args, config=student_config)
    teacher_model = None
    if training_args.loss_type == "KL":
        if model_args.teacher_model_name_or_path is None:
            raise ValueError(
                "--teacher_model_name_or_path is required when --loss_type KL "
                "for BD3LM."
            )
        teacher_model_name_or_path = dllm.utils.resolve_with_base_env(
            model_args.teacher_model_name_or_path,
            "BASE_MODELS_DIR",
        )

        teacher_config = transformers.AutoConfig.from_pretrained(
            teacher_model_name_or_path
        )
        if training_args.teacher_loophole_enabled:
            if not isinstance(
                teacher_config, dllm.pipelines.a2d.A2DQwen3Config
            ):
                raise ValueError(
                    "--teacher_loophole_enabled requires an A2D Qwen3 teacher"
                )
            if not getattr(teacher_config, "loophole_enabled", False):
                raise ValueError(
                    "--teacher_loophole_enabled requires a checkpoint whose "
                    "config.loophole_enabled is true"
                )

        teacher_kwargs = {
            "dtype": getattr(torch, model_args.teacher_dtype),
            "attn_implementation": model_args.teacher_attn_implementation,
            "config": teacher_config,
        }
        if torch.cuda.is_available():
            teacher_kwargs["device_map"] = {
                "": accelerate.PartialState().local_process_index
            }
        if model_args.teacher_load_in_4bit:
            if not transformers.utils.is_bitsandbytes_available():
                raise RuntimeError(
                    "--teacher_load_in_4bit requires bitsandbytes"
                )
            teacher_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, model_args.teacher_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        teacher_cls = (
            dllm.pipelines.a2d.A2DQwen3LMHeadModel
            if training_args.teacher_loophole_enabled
            else transformers.AutoModelForCausalLM
        )
        teacher_model = teacher_cls.from_pretrained(
            teacher_model_name_or_path,
            **teacher_kwargs,
        )
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        if model.config.vocab_size != teacher_model.config.vocab_size:
            raise ValueError(
                "Teacher and student vocabularies must match for KL distillation: "
                f"student={model.config.vocab_size}, "
                f"teacher={teacher_model.config.vocab_size}"
            )
    elif training_args.loss_type == "CE+KL":
        raise ValueError(
            "BD3LM does not support --loss_type CE+KL. Use --loss_type CE or "
            "--loss_type KL."
        )

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
        dataset = _map_and_postprocess(
            ds=dataset,
            mask_prompt_loss=data_args.mask_prompt_loss,
        )

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
        ar_model=teacher_model,
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
