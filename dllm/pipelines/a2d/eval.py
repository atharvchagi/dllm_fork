"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/a2d/eval.py \
    --tasks gsm8k_cot \
    --model a2d_mdlm \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1,max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0"

For BD3LM: use --model a2d_bd3lm and pretrained=dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1
"""

import argparse
import logging

from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.registry import register_model
from lm_eval.utils import simple_parse_args_string

from dllm.core.eval import (
    BD3LMEvalHarness as BD3LMEvalHarnessBase,
    MDLMEvalHarness,
)


@register_model("a2d_mdlm")
class A2DMDLMEvalHarness(MDLMEvalHarness):
    """A2D MDLM eval: thin subclass of core MDLMEvalHarness."""

    pass


@register_model("a2d_bd3lm")
class A2DBD3LMEvalHarness(BD3LMEvalHarnessBase):
    """A2D BD3LM eval: thin subclass of core BD3LMEvalHarness."""

    pass


def _inject_enable_thinking(args: argparse.Namespace) -> argparse.Namespace:
    """Merge top-level enable_thinking flag into model_args for lm-eval models."""
    if args.enable_thinking is None:
        return args

    if isinstance(args.model_args, dict):
        model_args = dict(args.model_args)
    else:
        model_args = simple_parse_args_string(args.model_args)

    if (
        "enable_thinking" in model_args
        and model_args["enable_thinking"] != args.enable_thinking
    ):
        logging.warning(
            "Overriding model_args enable_thinking=%s with CLI flag enable_thinking=%s",
            model_args["enable_thinking"],
            args.enable_thinking,
        )

    model_args["enable_thinking"] = args.enable_thinking
    args.model_args = model_args
    return args


if __name__ == "__main__":
    parser = setup_parser()
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=None,
        help="Override chat-template thinking behavior for compatible models.",
    )
    args = parse_eval_args(parser)
    cli_evaluate(_inject_enable_thinking(args))
