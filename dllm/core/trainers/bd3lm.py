"""
References:

Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models:
https://arxiv.org/abs/2503.09573

Learned Relay Representations for Forward-Thinking Discrete Diffusion Models:
https://arxiv.org/abs/2605.22967

Run training through a BD3LM entrypoint such as
``python /nvme-data/neeleshgarg/dllm_fork/examples/a2d/bd3lm/sft.py --help``.
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


class RelayStreamingBatch:
    """Keep partial token and detached relay states across optimizer batches."""

    def __init__(self) -> None:
        self.storage: dict[str, torch.Tensor] | None = None
        self.ready_to_evict: torch.Tensor | None = None

    def reset(self) -> None:
        self.storage = None
        self.ready_to_evict = None

    def matches(self, batch_size: int, sequence_length: int) -> bool:
        if self.storage is None:
            return False
        return self.storage["x_clean"].shape == (batch_size, sequence_length)

    @torch.no_grad()
    def evict_and_fill(
        self,
        batch: dict[str, torch.Tensor],
        mask_token_id: int,
        hidden_size: int,
    ) -> None:
        """Replace completed slots and fully mask newly inserted responses."""
        x_clean = batch["x_clean"]
        labels = batch["labels"]
        batch_size, sequence_length = x_clean.shape

        if self.storage is None:
            self.storage = {
                "x_clean": torch.empty_like(x_clean),
                "x_input": torch.empty_like(x_clean),
                "labels": torch.empty_like(labels),
                "h_s": torch.zeros(
                    batch_size,
                    sequence_length,
                    hidden_size,
                    dtype=torch.float32,
                    device=x_clean.device,
                ),
            }
            self.ready_to_evict = torch.ones(
                batch_size, dtype=torch.bool, device=x_clean.device
            )

        slots = self.ready_to_evict.nonzero(as_tuple=True)[0]
        count = min(slots.numel(), batch_size)
        if count == 0:
            return

        slots = slots[:count]
        x_clean = x_clean[:count]
        labels = labels[:count]
        maskable = labels != -100
        self.storage["x_clean"][slots] = x_clean
        self.storage["labels"][slots] = labels
        self.storage["x_input"][slots] = torch.where(
            maskable,
            torch.full_like(x_clean, mask_token_id),
            x_clean,
        )
        self.storage["h_s"][slots] = 0
        self.ready_to_evict[slots] = False

    @torch.no_grad()
    def persist(
        self,
        x_input: torch.Tensor,
        h_s: torch.Tensor,
        mask_token_id: int,
    ) -> None:
        """Detach and retain the state reached after a K=2 training window."""
        if self.storage is None:
            raise RuntimeError("RelayStreamingBatch must be filled before persisting")
        self.storage["x_input"] = x_input.detach().clone()
        self.storage["h_s"] = h_s.detach().float().clone()
        maskable = self.storage["labels"] != -100
        still_masked = (self.storage["x_input"] == mask_token_id) & maskable
        self.ready_to_evict = ~still_masked.any(dim=-1)

    def tensors(self) -> dict[str, torch.Tensor]:
        if self.storage is None:
            raise RuntimeError("RelayStreamingBatch has not been initialized")
        return self.storage


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
    relay_num_steps: int = 2
    relay_unmask_threshold: float = 0.85


class BD3LMTrainer(MDLMTrainer):

    _supports_right_shift_loopholing = True
    _loophole_loss_types = ("CE", "BPTT")

    def __init__(
        self,
        args: BD3LMConfig,
        *pargs,
        **kwargs,
    ):
        self._relay_metrics: dict[str, dict[str, list[torch.Tensor]]] = {
            "train": {},
            "eval": {},
        }
        if args.loss_type == "BPTT" and not args.loophole_enabled:
            raise ValueError("BD3LM loss_type='BPTT' requires loophole_enabled=True")
        if args.loss_type == "BPTT" and args.relay_num_steps != 2:
            raise ValueError("BD3LM currently implements relay_num_steps=2 only")
        if not 0.0 <= args.relay_unmask_threshold <= 1.0:
            raise ValueError("relay_unmask_threshold must be in [0, 1]")
        if args.loss_type == "CE+KL":
            raise ValueError(
                "BD3LMTrainer does not support loss_type='CE+KL'. Use 'CE' or 'KL'."
            )
        super().__init__(args=args, *pargs, **kwargs)
        self.block_size = args.block_size
        self.relay_num_steps = args.relay_num_steps
        self.relay_unmask_threshold = args.relay_unmask_threshold
        self.relay_train_buffer = RelayStreamingBatch()
        self.relay_eval_buffer = RelayStreamingBatch()
        self.loss_type_dict = {
            "CE": self.compute_CE_loss,
            "KL": self.compute_KL_loss,
            "BPTT": self.compute_BPTT_loss,
        }

    @torch.no_grad()
    def _record_relay_metrics(
        self,
        split: str,
        values: dict[str, torch.Tensor],
    ) -> None:
        for name, value in values.items():
            self._relay_metrics[split].setdefault(name, []).append(
                value.detach().float()
            )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Attach averaged Relay diagnostics to ordinary Trainer log events."""
        split = None
        if "eval_loss" in logs:
            split = "eval"
        elif "loss" in logs:
            split = "train"

        if split is not None:
            prefix = "eval_relay" if split == "eval" else "relay"
            for name, values in self._relay_metrics[split].items():
                value = torch.stack(values).mean().to(self.accelerator.device)
                value = self.accelerator.reduce(value, reduction="mean")
                logs[f"{prefix}/{name}"] = value.item()
            self._relay_metrics[split].clear()

        super().log(logs, start_time=start_time)

    def _align_loophole_state_for_input(
        self,
        loophole_state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply A2D's AR shift independently to the ``x_t`` and ``x_0`` streams."""
        if not self.right_shift_logits:
            return loophole_state
        if loophole_state.shape[1] % 2 != 0:
            raise ValueError(
                "BD3LM Loophole state must contain equal x_t and x_0 streams"
            )

        stream_length = loophole_state.shape[1] // 2
        shifted_streams = []
        for stream in loophole_state.split(stream_length, dim=1):
            shifted_streams.append(
                torch.cat([torch.zeros_like(stream[:, :1]), stream[:, :-1]], dim=1)
            )
        return torch.cat(shifted_streams, dim=1)

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
        
        #print(shape(input_ids), shape(labels), shape(attention_mask), shape(maskable_mask), shape(masked_mask), shape(noised_input_ids), shape(t))

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

        outputs = self._forward_with_loopholing(
            model=model,
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits[:, :l]  # [b, l, v]
        return outputs, logits

    def _forward_relay_step(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        input_ids: torch.Tensor,
        noised_input_ids: torch.Tensor,
        loophole_state: torch.Tensor | None,
        masked_mask: torch.Tensor,
    ):
        """Run one differentiable BD3LM rollout step with an optional relay state."""
        b, l = input_ids.shape
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)
        attention_mask = self._make_student_attention_mask(
            model=model,
            l=l,
            device=input_ids.device,
        )
        base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)
        loophole_mask = torch.cat(
            [masked_mask, torch.zeros_like(masked_mask)], dim=1
        )
        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
            logits_to_keep=torch.arange(l, device=input_ids.device),
            loophole_state=loophole_state,
            loophole_mask=loophole_mask,
            return_loophole_state=True,
            loophole_enabled=True,
        )
        outputs = self._postprocess_outputs(outputs)
        return outputs, outputs.logits[:, :l]

    def _prepare_relay_state(
        self,
        noisy_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Align the noisy-stream carry and zero-pad the clean BD3LM stream."""
        if self.right_shift_logits:
            noisy_hidden_state = torch.cat(
                [noisy_hidden_state[:, :1], noisy_hidden_state[:, :-1]], dim=1
            )
        return torch.cat(
            [noisy_hidden_state, torch.zeros_like(noisy_hidden_state)], dim=1
        )

    @torch.no_grad()
    def _select_relay_unmask_positions(
        self,
        logits: torch.Tensor,
        masked_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select confident positions, with one fallback reveal per active block."""
        # Converting the full [B, L, V] tensor to fp32 at once can require
        # several additional GiB for long sequences. Chunking over positions
        # computes the identical confidence values with a much smaller peak.
        confidence = torch.cat(
            [
                chunk.float().softmax(dim=-1).amax(dim=-1)
                for chunk in logits.split(64, dim=1)
            ],
            dim=1,
        )
        scores = confidence.masked_fill(~masked_mask, float("-inf"))
        selected = (scores >= self.relay_unmask_threshold) & masked_mask

        b, l = scores.shape
        block_size = self.block_size
        num_blocks = (l + block_size - 1) // block_size
        padded_length = num_blocks * block_size
        if padded_length != l:
            scores = F.pad(scores, (0, padded_length - l), value=float("-inf"))
        block_scores = scores.view(b, num_blocks, block_size)
        active_blocks = torch.isfinite(block_scores).any(dim=-1)
        best_in_block = block_scores.argmax(dim=-1)
        block_offsets = (
            torch.arange(num_blocks, device=logits.device) * block_size
        ).unsqueeze(0)
        best_positions = best_in_block + block_offsets
        batch_indices = torch.arange(b, device=logits.device).unsqueeze(1).expand_as(
            best_positions
        )
        selected[
            batch_indices[active_blocks], best_positions[active_blocks]
        ] = True
        return selected & masked_mask

    @staticmethod
    def _mean_masked_ce(
        logits: torch.Tensor,
        targets: torch.Tensor,
        masked_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
        ) * masked_mask.to(logits.dtype)
        loss = token_nll.sum() / masked_mask.sum().clamp_min(1)
        return loss, token_nll

    def compute_BPTT_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """Compute the two-step Relay objective with BPTT through the latent carry."""
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        batch_size, sequence_length = input_ids.shape
        relay_buffer = (
            self.relay_train_buffer if model.training else self.relay_eval_buffer
        )
        if not model.training or not relay_buffer.matches(
            batch_size, sequence_length
        ):
            relay_buffer.reset()
        hidden_size = self.accelerator.unwrap_model(model).config.hidden_size
        relay_buffer.evict_and_fill(
            batch={"x_clean": input_ids, "labels": labels},
            mask_token_id=self.processing_class.mask_token_id,
            hidden_size=hidden_size,
        )
        buffered = relay_buffer.tensors()
        input_ids = buffered["x_clean"]
        labels = buffered["labels"]
        noised_input_ids = buffered["x_input"]
        maskable_mask = labels != -100
        masked_mask = (
            noised_input_ids == self.processing_class.mask_token_id
        ) & maskable_mask
        relay_state_1 = self._prepare_relay_state(buffered["h_s"])

        outputs_1, logits_1 = self._forward_relay_step(
            model=model,
            input_ids=input_ids,
            noised_input_ids=noised_input_ids,
            loophole_state=relay_state_1,
            masked_mask=masked_mask,
        )
        loss_1, token_nll_1 = self._mean_masked_ce(
            logits=logits_1,
            targets=input_ids,
            masked_mask=masked_mask,
        )

        reveal_mask = self._select_relay_unmask_positions(
            logits=logits_1,
            masked_mask=masked_mask,
        )
        noised_input_ids_2 = torch.where(
            reveal_mask,
            input_ids,
            noised_input_ids,
        )
        masked_mask_2 = masked_mask & ~reveal_mask
        relay_state = getattr(outputs_1, "loophole_state", None)
        if relay_state is None:
            raise RuntimeError("The first Relay step did not return loophole_state")
        relay_state = self._prepare_relay_state(relay_state[:, :sequence_length])

        outputs_2, logits_2 = self._forward_relay_step(
            model=model,
            input_ids=input_ids,
            noised_input_ids=noised_input_ids_2,
            loophole_state=relay_state,
            masked_mask=masked_mask_2,
        )
        loss_2, token_nll_2 = self._mean_masked_ce(
            logits=logits_2,
            targets=input_ids,
            masked_mask=masked_mask_2,
        )
        loss = loss_1 + loss_2

        with torch.no_grad():
            reveal_mask_2 = self._select_relay_unmask_positions(
                logits=logits_2,
                masked_mask=masked_mask_2,
            )
            next_noised_input_ids = torch.where(
                reveal_mask_2, input_ids, noised_input_ids_2
            )
            next_noisy_state = outputs_2.loophole_state[:, :sequence_length]
            relay_buffer.persist(
                x_input=next_noised_input_ids,
                h_s=next_noisy_state,
                mask_token_id=self.processing_class.mask_token_id,
            )

            maskable_count = maskable_mask.sum().clamp_min(1)
            backbone = getattr(self.accelerator.unwrap_model(model), "model", None)
            relay_norm = getattr(backbone, "loophole_norm", None)
            relay_metrics = {
                "loss_1": loss_1,
                "loss_2": loss_2,
                "mask_ratio_step_1": masked_mask.sum() / maskable_count,
                "revealed_fraction_step_1": reveal_mask.sum()
                / masked_mask.sum().clamp_min(1),
                "completed_slot_fraction": relay_buffer.ready_to_evict.float().mean(),
            }
            if relay_norm is not None:
                relay_metrics["adapter_weight_norm"] = (
                    relay_norm.weight.float().norm()
                )
            self._record_relay_metrics(
                split="train" if model.training else "eval",
                values=relay_metrics,
            )

        split = "train" if model.training else "eval"
        self.meter.update(
            split=split,
            token_nll=token_nll_1.detach(),
            token_acc=(logits_1.argmax(dim=-1) == input_ids)
            .to(logits_1.dtype)
            .detach(),
            weight=masked_mask.to(logits_1.dtype).detach(),
        )
        self.meter.update(
            split=split,
            token_nll=token_nll_2.detach(),
            token_acc=(logits_2.argmax(dim=-1) == input_ids)
            .to(logits_2.dtype)
            .detach(),
            weight=masked_mask_2.to(logits_2.dtype).detach(),
        )
        return (loss, outputs_2) if return_outputs else loss

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
        token_acc = (logits.argmax(dim=-1) == batch["input_ids"]).to(logits.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            token_nll=token_nll.detach(),
            token_acc=token_acc.detach(),
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
            token_acc = (student_logits.argmax(dim=-1) == batch["input_ids"]).to(
                student_logits.dtype
            )
            self.meter.update(
                split="train" if model.training else "eval",
                token_nll=token_nll.detach(),
                token_acc=token_acc.detach(),
                weight=batch["masked_mask"].to(dtype=student_logits.dtype).detach(),
            )

        weighted_kl = self._normalize_kl_loss(
            weighted_kl=weighted_kl,
            masked_mask=batch["masked_mask"],
            b=batch["b"],
        )
        loss = weighted_kl.sum()
        return (loss, outputs) if return_outputs else loss
