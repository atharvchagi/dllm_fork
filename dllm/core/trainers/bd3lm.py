"""
References:

Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models:
https://arxiv.org/abs/2503.09573
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.utils.collators import CollatorWrapper

from .mdlm import MDLMConfig, MDLMTrainer


@dataclass
class AppendEOSBlockWrapper(CollatorWrapper):
    block_size: int = 32

    def before(self, features):
        for ex in features:
            ids = ex["input_ids"]
            labs = ex["labels"]

            assert isinstance(ids, list) and isinstance(labs, list)

            L = len(ids)
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                ex["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
                ex["labels"] = labs + [self.tokenizer.eos_token_id] * pad_len
        return features


def _create_bd3lm_attention_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


@dataclass
class BD3LMConfig(MDLMConfig):
    block_size: int = 32


class BD3LMTrainer(MDLMTrainer):

    def __init__(
        self,
        args: BD3LMConfig,
        *pargs,
        **kwargs,
    ):
        if args.loss_type == "CE+KL":
            raise ValueError(
                "BD3LMTrainer does not support loss_type='CE+KL'. Use 'CE' or 'KL'."
            )
        super().__init__(args=args, *pargs, **kwargs)
        self.block_size = args.block_size
        # BD3LM intentionally supports CE and KL only.
        self.loss_type_dict = {
            "CE": self.compute_CE_loss,
            "KL": self.compute_KL_loss,
        }

    def _prepare_diffusion_batch(
        self,
        inputs: dict[str, torch.Tensor | Any],
    ) -> dict[str, Any]:
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        return {
            "inputs": inputs,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "b": b,
            "l": l,
            "maskable_mask": maskable_mask,
            "masked_mask": masked_mask,
            "noised_input_ids": noised_input_ids,
            "t": t,
        }

    def _make_student_attention_mask(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        l: int,
        device: torch.device,
    ):
        # [TODO]: others like flash attention 2
        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2, device=device)[:, None],
                kv_idx=torch.arange(l * 2, device=device)[None, :],
                block_size=self.block_size,
                n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            return attention_mask.to(device)

        if (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            from torch.nn.attention.flex_attention import create_block_mask

            return create_block_mask(
                partial(_create_bd3lm_attention_mask, block_size=self.block_size, n=l),
                B=None,
                H=None,
                Q_LEN=l * 2,
                KV_LEN=l * 2,
            )

        raise NotImplementedError

    def _forward_student(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        input_ids: torch.Tensor,
        noised_input_ids: torch.Tensor,
    ):
        b, l = input_ids.shape
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)
        attention_mask = self._make_student_attention_mask(
            model=model,
            l=l,
            device=input_ids.device,
        )
        base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits[:, :l]  # [b, l, v]
        return outputs, logits

    def _compute_student_token_nll(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        masked_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),  # [b, v, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        return token_nll * loss_weights * masked_mask.to(token_nll.dtype)

    def _normalize_ce_loss(
        self,
        token_nll: torch.Tensor,
        maskable_mask: torch.Tensor,
        b: int,
    ) -> torch.Tensor:
        # Keep BD3LM CE behavior unchanged for backward compatibility.
        if self.loss_norm_type == "token":
            token_nll = token_nll / maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / (
                maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError("Invalid loss_norm_type.")
        return token_nll

    def _normalize_kl_loss(
        self,
        weighted_kl: torch.Tensor,
        masked_mask: torch.Tensor,
        b: int,
    ) -> torch.Tensor:
        if self.loss_norm_type == "token":
            weighted_kl = weighted_kl / masked_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            weighted_kl = weighted_kl / (
                masked_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            weighted_kl = weighted_kl / b
        else:
            raise ValueError("Invalid loss_norm_type.")
        return weighted_kl

    def _forward_teacher_logits(
        self,
        input_ids: torch.Tensor,
        noised_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.ar_model is None:
            raise ValueError("KL loss requires an autoregressive model. Pass ar_model to __init__.")

        with torch.no_grad():
            if self.distillation_policy == "off_policy":
                teacher_input_ids = input_ids
            elif self.distillation_policy == "on_policy":
                teacher_input_ids = noised_input_ids
            else:
                raise ValueError(
                    f"Unknown distillation policy: {self.distillation_policy}. "
                    "Available: ['off_policy', 'on_policy']"
                )

            try:
                teacher_device = next(self.ar_model.parameters()).device
            except StopIteration:
                teacher_device = teacher_input_ids.device

            if teacher_input_ids.device != teacher_device:
                teacher_input_ids = teacher_input_ids.to(teacher_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(teacher_device)

            ar_outputs = self.ar_model(
                input_ids=teacher_input_ids,
                attention_mask=attention_mask,
            )
            ar_outputs = self._postprocess_outputs(ar_outputs)
            teacher_logits = getattr(ar_outputs, "logits", None)
            if teacher_logits is None:
                raise ValueError(
                    "Teacher model output has no logits. Use a causal-LM teacher "
                    "(e.g., transformers.AutoModelForCausalLM) for KL distillation. "
                    f"Got output type: {type(ar_outputs).__name__}."
                )
            return teacher_logits

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        if self.selected_loss_type == "CE+KL":
            raise ValueError(
                "BD3LMTrainer does not support loss_type='CE+KL'. Use 'CE' or 'KL'."
            )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            **kwargs,
        )

    def compute_CE_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        batch = self._prepare_diffusion_batch(inputs)
        outputs, logits = self._forward_student(
            model=model,
            input_ids=batch["input_ids"],
            noised_input_ids=batch["noised_input_ids"],
        )

        loss_weights = self._compute_loss_weights(
            t=batch["t"],
            inputs=batch["inputs"],
            masked_mask=batch["masked_mask"],
        )

        assert (
            batch["input_ids"][batch["maskable_mask"]]
            == batch["labels"][batch["maskable_mask"]]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = self._compute_student_token_nll(
            logits=logits,
            input_ids=batch["input_ids"],
            loss_weights=loss_weights,
            masked_mask=batch["masked_mask"],
        )

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=batch["maskable_mask"].to(dtype=logits.dtype).detach(),
        )

        token_nll = self._normalize_ce_loss(
            token_nll=token_nll,
            maskable_mask=batch["maskable_mask"],
            b=batch["b"],
        )
        loss = token_nll.sum()
        return (loss, outputs) if return_outputs else loss

    def compute_KL_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        batch = self._prepare_diffusion_batch(inputs)
        outputs, student_logits = self._forward_student(
            model=model,
            input_ids=batch["input_ids"],
            noised_input_ids=batch["noised_input_ids"],
        )
        teacher_logits = self._forward_teacher_logits(
            input_ids=batch["input_ids"],
            noised_input_ids=batch["noised_input_ids"],
            attention_mask=batch["attention_mask"],
        )

        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                "Teacher and student logits must have the same shape. "
                f"Got teacher={tuple(teacher_logits.shape)} "
                f"and student={tuple(student_logits.shape)}."
            )

        kl_per_position = self.compute_kl_divergence(
            ar_logits=teacher_logits,
            dlm_logits=student_logits,
            maskable_mask=None,
            reduction="none",
        )

        loss_weights = self._compute_loss_weights(
            t=batch["t"],
            inputs=batch["inputs"],
            masked_mask=batch["masked_mask"],
        )
        weighted_kl = (
            kl_per_position
            * loss_weights
            * batch["masked_mask"].to(kl_per_position.dtype)
        )

        with torch.no_grad():
            token_nll = self._compute_student_token_nll(
                logits=student_logits,
                input_ids=batch["input_ids"],
                loss_weights=loss_weights,
                masked_mask=batch["masked_mask"],
            )
            self.meter.update(
                split="train" if model.training else "eval",
                value=token_nll.detach(),
                weight=batch["masked_mask"].to(dtype=student_logits.dtype).detach(),
            )

        weighted_kl = self._normalize_kl_loss(
            weighted_kl=weighted_kl,
            masked_mask=batch["masked_mask"],
            b=batch["b"],
        )
        loss = weighted_kl.sum()
        return (loss, outputs) if return_outputs else loss
