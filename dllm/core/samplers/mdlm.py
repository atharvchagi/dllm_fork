"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

Run sampling through an MDLM entrypoint such as
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/mdlm/sample.py --help``.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    loophole_enabled: bool = False
    return_step_entropy: bool = False
    confidence_threshold: float | None = None


@dataclass
class MDLMSampler(BaseSampler):
    @staticmethod
    def _masked_prediction_metrics(
        logits: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute per-sequence entropy and confidence on active mask positions."""
        batch_size = logits.shape[0]
        entropy = torch.full(
            (batch_size,),
            torch.nan,
            dtype=torch.float32,
            device=logits.device,
        )
        top1_probability = torch.full_like(entropy, torch.nan)
        remaining_masks = active_mask.sum(dim=-1)

        for row in range(batch_size):
            selected_logits = logits[row, active_mask[row]].float()
            if selected_logits.numel() == 0:
                continue
            probabilities = F.softmax(selected_logits, dim=-1)
            token_entropy = torch.special.entr(probabilities).sum(dim=-1)
            entropy[row] = token_entropy.mean()
            top1_probability[row] = probabilities.amax(dim=-1).mean()

        return {
            "mean_entropy_nats": entropy.detach().cpu(),
            "mean_top1_probability": top1_probability.detach().cpu(),
            "remaining_masks": remaining_masks.detach().cpu(),
        }

    @staticmethod
    def _select_transfer_index(
        confidence: torch.Tensor,
        candidate_mask: torch.Tensor,
        minimum_transfer_tokens: torch.Tensor,
        confidence_threshold: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select scheduled tokens, plus every threshold-qualified token."""
        batch_size = confidence.shape[0]
        transfer_index = torch.zeros_like(candidate_mask)
        threshold_accepted = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=confidence.device,
        )
        below_threshold_fallback = torch.zeros_like(threshold_accepted)

        for row in range(batch_size):
            masked_count = int(candidate_mask[row].sum().item())
            minimum_count = min(
                int(minimum_transfer_tokens[row].item()),
                masked_count,
            )
            if masked_count == 0:
                continue

            if confidence_threshold is not None:
                high_confidence = candidate_mask[row] & (
                    confidence[row] >= confidence_threshold
                )
                transfer_index[row] = high_confidence
                accepted_count = int(high_confidence.sum().item())
                threshold_accepted[row] = accepted_count
                fallback_count = max(0, minimum_count - accepted_count)
                if fallback_count > 0:
                    fallback_confidence = torch.where(
                        candidate_mask[row] & ~high_confidence,
                        confidence[row],
                        -torch.inf,
                    )
                    _, selected = torch.topk(
                        fallback_confidence,
                        k=fallback_count,
                    )
                    transfer_index[row, selected] = True
                    below_threshold_fallback[row] = fallback_count
                continue

            if minimum_count > 0:
                candidate_confidence = torch.where(
                    candidate_mask[row],
                    confidence[row],
                    -torch.inf,
                )
                _, selected = torch.topk(candidate_confidence, k=minimum_count)
                transfer_index[row, selected] = True

        return transfer_index, threshold_accepted, below_threshold_fallback

    @staticmethod
    def _selected_confidence_metrics(
        confidence: torch.Tensor,
        transfer_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sequence mean and minimum committed-token confidence."""
        batch_size = confidence.shape[0]
        mean_confidence = torch.full(
            (batch_size,),
            torch.nan,
            dtype=torch.float32,
            device=confidence.device,
        )
        minimum_confidence = torch.full_like(mean_confidence, torch.nan)
        for row in range(batch_size):
            selected = confidence[row, transfer_index[row]].float()
            if selected.numel() == 0:
                continue
            mean_confidence[row] = selected.mean()
            minimum_confidence[row] = selected.min()
        return mean_confidence, minimum_confidence

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loophole_state: torch.Tensor | None,
        loophole_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward the canvas and advance its position-aligned recurrent state."""
        if not loophole_enabled:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.logits, None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loophole_state=loophole_state,
            return_loophole_state=True,
            loophole_enabled=True,
        )
        next_state = getattr(outputs, "loophole_state", None)
        if next_state is None:
            raise RuntimeError(
                "The Loopholing sampler forward did not return loophole_state"
            )
        return outputs.logits, next_state

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        return_history = kwargs.get("return_history", config.return_history)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        loophole_enabled = kwargs.get(
            "loophole_enabled", config.loophole_enabled
        )
        return_step_entropy = kwargs.get(
            "return_step_entropy", config.return_step_entropy
        )
        confidence_threshold = kwargs.get(
            "confidence_threshold", config.confidence_threshold
        )
        distillation_step_callback = kwargs.get("distillation_step_callback", None)
        if return_step_entropy and not return_dict:
            raise ValueError("return_step_entropy=True requires return_dict=True")
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        if loophole_enabled and right_shift_logits:
            raise ValueError(
                "Loopholing does not yet support right_shift_logits=True because "
                "the recurrent state would require an explicit positional shift."
            )
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be between 0 and 1")
            if remasking != "low_confidence":
                raise ValueError(
                    "confidence_threshold requires remasking='low_confidence'"
                )
        if distillation_step_callback is not None:
            if not callable(distillation_step_callback):
                raise TypeError("distillation_step_callback must be callable")
            if not loophole_enabled:
                raise ValueError(
                    "distillation_step_callback requires loophole_enabled=True"
                )
            if cfg_scale > 0.0:
                raise ValueError(
                    "distillation_step_callback does not support classifier-free guidance"
                )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict and return_history else None
        step_metrics = [] if return_step_entropy else None
        loophole_state = None
        global_step = 0

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map
                candidate_mask = torch.zeros_like(mask_index)
                for j in range(B):
                    start = prompt_lens[j] + b * block_size
                    end = min(
                        start + block_size,
                        prompt_lens[j] + max_new_tokens,
                        T,
                    )
                    if start < end:
                        candidate_mask[j, start:end] = mask_index[j, start:end]
                if not candidate_mask.any():
                    break

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits, loophole_state = self._model_forward(
                        input_ids=x_,
                        attention_mask=attention_mask.repeat(2, 1),
                        loophole_state=loophole_state,
                        loophole_enabled=loophole_enabled,
                    )
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits, loophole_state = self._model_forward(
                        input_ids=x,
                        attention_mask=attention_mask,
                        loophole_state=loophole_state,
                        loophole_enabled=loophole_enabled,
                    )

                if distillation_step_callback is not None:
                    distillation_step_callback(
                        input_ids=x,
                        attention_mask=attention_mask,
                        candidate_mask=candidate_mask,
                        logits=logits,
                        loophole_state=loophole_state,
                        global_step=global_step + 1,
                        block=b + 1,
                        step_in_block=i + 1,
                    )

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if step_metrics is not None:
                    metrics = self._masked_prediction_metrics(logits, candidate_mask)
                    metrics.update(
                        {
                            "step": global_step + 1,
                            "block": b + 1,
                            "step_in_block": i + 1,
                        }
                    )
                else:
                    metrics = None

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    confidence_logits = (
                        logits.float()
                        if confidence_threshold is not None
                        else logits
                    )
                    p = F.softmax(confidence_logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    candidate_mask, x0_p, -np.inf
                )  # consider current-block masked positions only

                # The schedule is a completion floor. Threshold mode may accept
                # more positions, allowing a block to finish before its step budget.
                transfer_index, threshold_accepted, fallback_tokens = (
                    self._select_transfer_index(
                        confidence=confidence,
                        candidate_mask=candidate_mask,
                        minimum_transfer_tokens=num_transfer_tokens[:, i],
                        confidence_threshold=confidence_threshold,
                    )
                )
                if metrics is not None:
                    transferred_tokens = transfer_index.sum(dim=-1)
                    mean_selected, minimum_selected = (
                        self._selected_confidence_metrics(
                            confidence,
                            transfer_index,
                        )
                    )
                    metrics.update(
                        {
                            "scheduled_minimum_tokens": num_transfer_tokens[
                                :, i
                            ].detach().cpu(),
                            "transferred_tokens": transferred_tokens.detach().cpu(),
                            "remaining_masks_after": (
                                candidate_mask.sum(dim=-1) - transferred_tokens
                            ).detach().cpu(),
                            "mean_transferred_confidence": mean_selected.detach().cpu(),
                            "minimum_transferred_confidence": minimum_selected.detach().cpu(),
                            "threshold_accepted_tokens": threshold_accepted.detach().cpu(),
                            "below_threshold_fallback_tokens": fallback_tokens.detach().cpu(),
                        }
                    )
                    step_metrics.append(metrics)

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())
                global_step += 1

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(
                sequences=x,
                histories=histories,
                step_metrics=step_metrics,
            )

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        return_history = kwargs.get("return_history", config.return_history)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        loophole_enabled = kwargs.get(
            "loophole_enabled", config.loophole_enabled
        )
        confidence_threshold = kwargs.get(
            "confidence_threshold", config.confidence_threshold
        )
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        if loophole_enabled and right_shift_logits:
            raise ValueError(
                "Loopholing does not yet support right_shift_logits=True because "
                "the recurrent state would require an explicit positional shift."
            )
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be between 0 and 1")
            if remasking != "low_confidence":
                raise ValueError(
                    "confidence_threshold requires remasking='low_confidence'"
                )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict and return_history else None
        loophole_state = None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id
                candidate_mask = torch.zeros_like(mask_index_full)
                for j in range(B):
                    end_j = start + widths[j]
                    if start < end_j:
                        candidate_mask[j, start:end_j] = mask_index_full[
                            j, start:end_j
                        ]
                if not candidate_mask.any():
                    break

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits, loophole_state = self._model_forward(
                        input_ids=x_,
                        attention_mask=attention_mask.repeat(2, 1),
                        loophole_state=loophole_state,
                        loophole_enabled=loophole_enabled,
                    )
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits, loophole_state = self._model_forward(
                        input_ids=x,
                        attention_mask=attention_mask,
                        loophole_state=loophole_state,
                        loophole_enabled=loophole_enabled,
                    )

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    confidence_logits = (
                        logits.float()
                        if confidence_threshold is not None
                        else logits
                    )
                    p = F.softmax(confidence_logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(candidate_mask, x0_p, -np.inf)

                transfer_index, _, _ = self._select_transfer_index(
                    confidence=confidence,
                    candidate_mask=candidate_mask,
                    minimum_transfer_tokens=num_transfer_tokens[:, s],
                    confidence_threshold=confidence_threshold,
                )

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
