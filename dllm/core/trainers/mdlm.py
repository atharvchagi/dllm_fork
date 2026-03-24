"""
References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

from typing import Any, Union, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.configs import TrainingArguments
from dllm.utils.data import prepend_bos
from .utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback

@dataclass
class MDLMConfig(TrainingArguments):
   time_epsilon: float = 1e-3
   loss_weight_type: str = "scheduler"  # "scheduler", "uniform"
   loss_norm_type: str = "token"  # "batch", "sequence", "token"
   right_shift_logits: bool = False
   loss_type: str = "CE"  # "CE", "KL", "CE+KL"
   distillation_policy: str = "off_policy"  # "off_policy", "on_policy"
   kl_divergence_type: str = "forward"  # "forward", "reverse"
   kl_weight: float = 1.0  # Weight for KL term in CE+KL loss (loss = CE + kl_weight * KL)

class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        args: MDLMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        ar_model: transformers.PreTrainedModel | nn.Module | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits
        self.selected_loss_type = args.loss_type
        self.distillation_policy = args.distillation_policy
        self.kl_divergence_type = args.kl_divergence_type
        self.kl_weight = args.kl_weight
        
        # Optional AR model for KL divergence computation
        self.ar_model = ar_model
        if self.selected_loss_type in ["KL", "CE+KL"] and self.ar_model is None:
            raise ValueError(f"{self.selected_loss_type} loss requires an autoregressive model. Pass ar_model to __init__.")
        
        # Dictionary of available loss functions
        self.loss_type_dict = {
            "CE": self.compute_CE_loss,
            "KL": self.compute_KL_loss,
            "CE+KL": self.compute_CE_KL_loss,
        }

        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)

            # If labels exist and EVERY sequence already starts with -100,
            # we treat them as is and skip prepending BOS.
            if labels is not None:
                # shape: [bsz, seq_len]
                if torch.all(labels[:, 0] == -100):
                    return inputs

            # Otherwise, prepend BOS (and corresponding labels / attention_mask).
            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Use the configured objective (and subclass overrides) during eval.
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Dispatch to the appropriate loss function based on configuration.

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        loss_fn = self.loss_type_dict.get(self.selected_loss_type)
        if loss_fn is None:
            raise ValueError(
                f"Unknown loss type: {self.selected_loss_type}. "
                f"Available: {list(self.loss_type_dict.keys())}"
            )
        return loss_fn(model, inputs, return_outputs=return_outputs, **kwargs)

    def compute_CE_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the masked diffusion language modeling loss using cross-entropy.

        Applies stochastic masking to input tokens based on a diffusion timestep,
        then computes the weighted cross-entropy loss for predicting the original tokens.

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        # Each example draws a random timestep t ∈ [ε, 1), where ε avoids degenerate values near 0.
        # The scheduler defines the masking rate α(t); we convert it to a masking probability p_mask = 1 - α(t).
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        # === 2. Apply stochastic masking ===
        # Tokens are masked independently according to p_mask(t).
        # Positions with label = -100 are excluded (ignored in loss).
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        # Replace masked tokens with the special [MASK] token.
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model ===
        # The model predicts clean tokens given noised inputs.
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        
        
        
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        # === 5. Compute weighted cross-entropy ===
        # Sanity check: ensure input_ids and labels match at valid positions
        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)  # [b, l]

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        # === 6. Normalize loss ===
        if self.loss_norm_type == "token":
            token_nll /= masked_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= masked_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()
        print(f"CE Loss: {loss}")

        # === 7. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
    
   

    def compute_kl_divergence(
        self,
        ar_logits: torch.Tensor,
        dlm_logits: torch.Tensor,
        maskable_mask: torch.Tensor = None,
        reduction: str = "none"
    ) -> torch.Tensor:
        """
        Computes KL divergence between teacher and student logits.
        Supports both forward KL(teacher || student) and reverse KL(student || teacher).

        Args:
            ar_logits: Logits from the Teacher (AR model) [Batch, Seq, Vocab]
            dlm_logits: Logits from the Student (DLM model) [Batch, Seq, Vocab]
            maskable_mask: Boolean mask [Batch, Seq] where True indicates positions to calculate loss.
                           (Optional, mostly used if reduction is not 'none')
            reduction: 'none', 'batchmean', 'sum', or 'mean'.

        Returns:
            KL divergence tensor. If reduction is 'none', shape is [Batch, Seq].
        """
        if self.kl_divergence_type == "forward":
            # Forward KL: KL(teacher || student) = sum_i P(i) * log(P(i) / Q(i))
            # Student should cover all modes of teacher (mode-seeking)
            student_log_probs = F.log_softmax(dlm_logits, dim=-1)
            teacher_probs = F.softmax(ar_logits, dim=-1)
            # F.kl_div expects (input=log_probs, target=probs)
            kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        elif self.kl_divergence_type == "reverse":
            # Reverse KL: KL(student || teacher) = sum_i Q(i) * log(Q(i) / P(i))
            # Student should match the mode of teacher (mode-covering)
            teacher_log_probs = F.log_softmax(ar_logits, dim=-1)
            student_probs = F.softmax(dlm_logits, dim=-1)
            # F.kl_div expects (input=log_probs, target=probs)
            kl_div = F.kl_div(teacher_log_probs, student_probs, reduction='none')
        else:
            raise ValueError(f"Unknown KL divergence type: {self.kl_divergence_type}")

        # Sum over the vocabulary dimension to get KL per token
        # Result shape: [Batch, Seq]
        
        # print(f"KL div shape: {kl_div.shape}")
        # print(f"KL div: {kl_div}")
        
        kl_per_token = kl_div.sum(dim=-1)
        
        # print(f"KL per token: {kl_per_token}")
        # print(kl_per_token.shape)

        # Zero out masked positions if mask provided (safety step)
        if maskable_mask is not None:
            kl_per_token = kl_per_token * maskable_mask.to(kl_per_token.dtype)
        
        # print(f"KL per token after masking: {kl_per_token}")
        # print(kl_per_token.shape)

        if reduction == "none":
            return kl_per_token
        elif reduction == "sum":
            return kl_per_token.sum()
        elif reduction == "mean":
            if maskable_mask is not None:
                return kl_per_token.sum() / maskable_mask.sum().clamp_min(1)
            return kl_per_token.mean()

        return kl_per_token

    def compute_KL_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute KL divergence between autoregressive model and DLM.

        Implements: KL(p_T || p_θ) = Σ_v p_T(v|x_0^{<i}) * log(p_T(v|x_0^{<i}) / p_θ(v|x_t, i))

        Args:
            model: The DLM model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        assert self.processing_class.padding_side == "right"
        
        #print(f"Inputs: {inputs}")
        
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        
        #print(f"input_ids shape: {input_ids.shape}")
        
        
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]
        
        #print(p_mask)

        # === 2. Apply stochastic masking ===
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )
        
        #print(masked_mask)
        #
        #print(f"input_ids shape: {input_ids.shape}")
        #print(f"noised_input_ids shape: {noised_input_ids.shape}")
        #print(f"input_ids: {input_ids}")
        #print(f"noised_input_ids: {noised_input_ids}")
        #
        #print(f"Decoded Input IDs: {[self.processing_class.decode(ids) for ids in input_ids]}")
        #print(f"Decoded Noised Input IDs: {[self.processing_class.decode(ids) for ids in noised_input_ids]}")

        # === 3. Get autoregressive model logits (teacher) ===
        # === 3. Get autoregressive model logits (teacher) ===
        with torch.no_grad():
            if self.distillation_policy == "off_policy":
                ar_outputs = self.ar_model(input_ids=input_ids, attention_mask=attention_mask)
            elif self.distillation_policy == "on_policy":
                ar_outputs = self.ar_model(input_ids=noised_input_ids, attention_mask=attention_mask)
        
            # Let the global postprocess handle the shift so it matches the student logic
            ar_outputs = self._postprocess_outputs(ar_outputs)
            ar_logits = ar_outputs.logits
            
        
        #print(f"AR logits shape: {ar_logits.shape}")
        #print(f"AR logits: {ar_logits}")
          
        # === 4. Forward pass through DLM (student) ===
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        dlm_logits = outputs.logits  # [b, l, V]
       
        #print(f"DLM logits shape: {dlm_logits.shape}")
        #print(f"DLM logits: {dlm_logits}")

        # === 5. Compute KL divergence ===
        # Compute KL for all positions first
        kl_per_position = self.compute_kl_divergence(
            ar_logits=ar_logits,
            dlm_logits=dlm_logits,
            maskable_mask= None,  
            reduction="none" 
        )

        # === 6. Apply loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )
        
        weighted_kl = kl_per_position * loss_weights * masked_mask.to(kl_per_position.dtype)
        
        #print(f"KL per position: {kl_per_position}")
        #print(f"Weighted KL per position with masks: {weighted_kl}")

        # === 6b. Compute actual NLL for perplexity metric ===
        # For proper perplexity, we need the student's NLL on ground truth tokens
        # (KL divergence itself doesn't translate to perplexity)
        with torch.no_grad():
            token_nll = F.cross_entropy(
                dlm_logits.transpose(1, 2),  # [b, V, l]
                input_ids,  # [b, l]
                reduction="none",  # [b, l]
            )
            token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)  # [b, l]
            
            # Update metrics with NLL (not KL) for meaningful perplexity
            self.meter.update(
                split="train" if model.training else "eval",
                value=token_nll.detach(),
                weight=masked_mask.to(dtype=dlm_logits.dtype).detach(),
            )

        # === 7. Normalize loss ===
        if self.loss_norm_type == "token":
            weighted_kl /= masked_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            weighted_kl /= masked_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            weighted_kl /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = weighted_kl.sum()
        print(f"KL Loss: {loss}")

        # === 8. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss

    
    def compute_CE_KL_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute combined CE + KL loss for joint training.
        
        Loss = CE + kl_weight * KL
        where CE is cross-entropy on ground truth and KL is distillation from teacher.
        """
        
        assert self.processing_class.padding_side == "right"
        
        #print(f"Inputs: {inputs}")
        
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        
        #print(f"input_ids shape: {input_ids.shape}")
        
        
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]
        
        #print(p_mask)

        # === 2. Apply stochastic masking ===
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )
        
        #print(masked_mask)
        #
        #print(f"input_ids shape: {input_ids.shape}")
        #print(f"noised_input_ids shape: {noised_input_ids.shape}")
        #print(f"input_ids: {input_ids}")
        #print(f"noised_input_ids: {noised_input_ids}")
        #
        #print(f"Decoded Input IDs: {[self.processing_class.decode(ids) for ids in input_ids]}")
        #print(f"Decoded Noised Input IDs: {[self.processing_class.decode(ids) for ids in noised_input_ids]}")

    
        # === 3. Get autoregressive model logits (teacher) ===
        with torch.no_grad():
            if self.distillation_policy == "off_policy":
                ar_outputs = self.ar_model(input_ids=input_ids, attention_mask=attention_mask)
            elif self.distillation_policy == "on_policy":
                ar_outputs = self.ar_model(input_ids=noised_input_ids, attention_mask=attention_mask)
        
            # Let the global postprocess handle the shift so it matches the student logic
            ar_outputs = self._postprocess_outputs(ar_outputs)
            ar_logits = ar_outputs.logits
            
        
        #print(f"AR logits shape: {ar_logits.shape}")
        #print(f"AR logits: {ar_logits}")
          
        # === 4. Forward pass through DLM (student) ===
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        dlm_logits = outputs.logits  # [b, l, V]
       
        #print(f"DLM logits shape: {dlm_logits.shape}")
        #print(f"DLM logits: {dlm_logits}")

        # === 5. Compute KL divergence ===
        # Compute KL for all positions first
        kl_per_position = self.compute_kl_divergence(
            ar_logits=ar_logits,
            dlm_logits=dlm_logits,
            maskable_mask= None,  
            reduction="none" 
        )

        # === 6. Apply loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )
        
        weighted_kl = kl_per_position * loss_weights * masked_mask.to(kl_per_position.dtype)
        
        #print(f"KL per position: {kl_per_position}")
        #print(f"Weighted KL per position with masks: {weighted_kl}")

        # === 6b. Compute actual NLL for perplexity metric ===
        # For proper perplexity, we need the student's NLL on ground truth tokens
        # (KL divergence itself doesn't translate to perplexity)
        
        token_nll = F.cross_entropy(
            dlm_logits.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)  # [b, l]
        
        # Update metrics with NLL (not KL) for meaningful perplexity
        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=masked_mask.to(dtype=dlm_logits.dtype).detach(),
        )

        # === 7. Normalize both CE and KL losses ===
        if self.loss_norm_type == "token":
            weighted_kl /= masked_mask.sum().clamp_min(1)
            token_nll /= masked_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            weighted_kl /= masked_mask.sum(-1, keepdim=True).clamp_min(1) * b
            token_nll /= masked_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            weighted_kl /= b
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        
        ce_loss = token_nll.sum()
        kl_loss = weighted_kl.sum()
        loss = ce_loss + self.kl_weight * kl_loss
        print(f"CE+KL Loss: {loss:.4f} (CE: {ce_loss:.4f}, KL: {kl_loss:.4f}, weight: {self.kl_weight})")

        # === 8. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
        
        
        
        
    
    # def training_step(
    #     self,
    #     model: nn.Module,
    #     inputs: dict[str, Union[torch.Tensor, Any]],
    #     num_items_in_batch: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     # Prepare buffers for context parallelism

    #     cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        

    #     # Context manager is no-op if CP isn't enabled
    #     with cp_context():
    #         model.eval() 
            
    #         """
    #         Edited for AR analysis: set the model to eval so backward graph not made.
    #         """
            
            
    #         if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #             self.optimizer.train()

    #         # inputs = self._prepare_inputs(inputs)
    #         # if is_sagemaker_mp_enabled():
    #         #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         #     return loss_mb.reduce_mean().detach().to(self.args.device)

    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    #         # del inputs
    #         # if (
    #         #     self.args.torch_empty_cache_steps is not None
    #         #     and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #         # ):
    #         #     if is_torch_xpu_available():
    #         #         torch.xpu.empty_cache()
    #         #     elif is_torch_mlu_available():
    #         #         torch.mlu.empty_cache()
    #         #     elif is_torch_musa_available():
    #         #         torch.musa.empty_cache()
    #         #     elif is_torch_npu_available():
    #         #         torch.npu.empty_cache()
    #         #     elif is_torch_mps_available():
    #         #         torch.mps.empty_cache()
    #         #     elif is_torch_hpu_available():
    #         #         logger.warning(
    #         #             "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
    #         #         )
    #         #     else:
    #         #         torch.cuda.empty_cache()

    #         # # kwargs = {}

    #         # For LOMO optimizers you need to explicitly use the learning rate
    #         # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         #     kwargs["learning_rate"] = self._get_learning_rate()

    #         # if self.args.n_gpu > 1:
    #         #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         # if self.use_apex:
    #         #     from apex import amp

    #         #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #         #         scaled_loss.backward()
    #         # else:
    #         #     # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
    #         #     if (
    #         #         not self.model_accepts_loss_kwargs or num_items_in_batch is None
    #         #     ) and self.compute_loss_func is None:
    #         #         # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
    #         #         loss = loss / self.current_gradient_accumulation_steps

    #         #     # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
    #         #     # https://github.com/huggingface/transformers/pull/35808
    #         #     if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    #         #         kwargs["scale_wrt_gas"] = False

    #             #self.accelerator.backward(loss, **kwargs)
            
                
    #             """
    #             Edited for AR analysis: do not want to backprop
    #             """

    #         return loss.detach()
    