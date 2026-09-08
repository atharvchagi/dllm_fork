"""A2D Qwen3 model definitions.

Run a local construction smoke test with
``python -m dllm.pipelines.a2d.models.qwen3.modeling_qwen3``.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor

@dataclass
class A2DQwen3BaseModelOutputWithPast(BaseModelOutputWithPast):
    """Qwen3 backbone output with the recurrent Loopholing state."""

    loophole_state: Optional[torch.FloatTensor] = None


@dataclass
class A2DQwen3CausalLMOutputWithPast(CausalLMOutputWithPast):
    """Qwen3 language-model output with the recurrent Loopholing state."""

    loophole_state: Optional[torch.FloatTensor] = None


class LoopholeLayerNorm(nn.LayerNorm):
    """LayerNorm whose affine parameters must always initialize to zero."""


class A2DQwen3Config(transformers.Qwen3Config):
    model_type = "a2d-qwen3"  # <- NEW model_type

    def __init__(self, loophole_enabled: bool = False, **kwargs):
        super().__init__(**kwargs)
        # This is an architecture capability flag and is serialized with checkpoints.
        # Individual trainer/sampler calls still opt into the recurrent path explicitly.
        self.loophole_enabled = loophole_enabled


class A2DQwen3Model(transformers.Qwen3Model):

    def __init__(self, config: A2DQwen3Config):
        super().__init__(config)
        self.loophole_norm = (
            LoopholeLayerNorm(config.hidden_size)
            if config.loophole_enabled
            else None
        )
        self.reset_loophole_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, LoopholeLayerNorm):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
            return
        super()._init_weights(module)

    def reset_loophole_parameters(self) -> None:
        """Zero the Loopholing adapter so a converted model starts unchanged."""
        if self.loophole_norm is None:
            return
        nn.init.zeros_(self.loophole_norm.weight)
        nn.init.zeros_(self.loophole_norm.bias)

    def _inject_loophole_state(
        self,
        inputs_embeds: torch.Tensor,
        loophole_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.loophole_norm is None:
            raise ValueError(
                "Loopholing was requested, but this checkpoint does not contain the "
                "Loopholing adapter. Convert or load it with loophole_enabled=True first."
            )

        if loophole_state is None:
            loophole_state = torch.zeros_like(inputs_embeds)
        else:
            if loophole_state.shape != inputs_embeds.shape:
                raise ValueError(
                    "loophole_state must have shape "
                    f"{tuple(inputs_embeds.shape)}, got {tuple(loophole_state.shape)}"
                )
            if loophole_state.device != inputs_embeds.device:
                raise ValueError(
                    "loophole_state and token embeddings must be on the same device, "
                    f"got {loophole_state.device} and {inputs_embeds.device}"
                )
            if not loophole_state.is_floating_point():
                raise TypeError(
                    "loophole_state must be floating point, "
                    f"got dtype {loophole_state.dtype}"
                )
            loophole_state = loophole_state.to(dtype=inputs_embeds.dtype)

        return inputs_embeds + self.loophole_norm(loophole_state)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loophole_state: Optional[torch.FloatTensor] = None,
        return_loophole_state: bool = False,
        loophole_enabled: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> A2DQwen3BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # An ordinary forward bypasses the adapter even when a checkpoint contains it.
        # Requesting a state input/output opts in unless explicitly overridden.
        use_loophole = (
            loophole_state is not None or return_loophole_state
            if loophole_enabled is None
            else loophole_enabled
        )
        if loophole_state is not None and not use_loophole:
            raise ValueError(
                "loophole_state was provided while loophole_enabled=False"
            )
        if use_loophole:
            inputs_embeds = self._inject_loophole_state(
                inputs_embeds=inputs_embeds,
                loophole_state=loophole_state,
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        """
        # -------------------------------------------------------------
        # ORIGINAL CODE (causal mask)
        # -------------------------------------------------------------
        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        # -------------------------------------------------------------
        # ORIGINAL CODE (causal mask)
        # -------------------------------------------------------------
        """
        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # 1) If no mask is provided → treat all tokens as valid (no padding)
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], 
                    device=inputs_embeds.device, 
                    dtype=torch.long
                )

            # 2) If mask is not already a 4D attention mask → convert it
            if not (
                isinstance(attention_mask, BlockMask)
                or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
            ):
                attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

            # 3) Build causal mask mapping used by the attention layers
            causal_mask_mapping = {"full_attention": attention_mask}

            # Sliding-window layers share the same non-causal mask
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = attention_mask
        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return A2DQwen3BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            loophole_state=hidden_states if return_loophole_state else None,
        )


class A2DQwen3LMHeadModel(transformers.Qwen3ForCausalLM):
    config: A2DQwen3Config

    def __init__(self, config):
        transformers.Qwen3PreTrainedModel.__init__(self, config)
        self.model = A2DQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        # The outer post_init() reinitializes LayerNorm scales to one. Reset the
        # adapter afterwards so converting a baseline checkpoint is functionally exact.
        self.model.reset_loophole_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        # from_pretrained() initializes parameters missing from an older checkpoint
        # after construction, so the dedicated adapter needs an explicit rule here.
        if isinstance(module, LoopholeLayerNorm):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
            return
        super()._init_weights(module)

    @transformers.utils.can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        loophole_state: Optional[torch.FloatTensor] = None,
        return_loophole_state: bool = False,
        loophole_enabled: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> A2DQwen3CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            loophole_state=loophole_state,
            return_loophole_state=return_loophole_state,
            loophole_enabled=loophole_enabled,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return A2DQwen3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loophole_state=outputs.loophole_state,
        )


transformers.AutoConfig.register("a2d-qwen3", A2DQwen3Config)
transformers.AutoModel.register(A2DQwen3Config, A2DQwen3LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen3Config, A2DQwen3LMHeadModel)


if __name__ == "__main__":
    import dllm
    import torch
    from transformers import AutoModel

    # Load a config from a local path (either a directory containing config.json, or the file itself)
    config_path = dllm.utils.resolve_with_base_env(
        "Qwen/Qwen3-0.6B-Base", "BASE_MODELS_DIR"
    )
    config = A2DQwen3Config.from_pretrained(config_path)
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    if hasattr(config, "architectures"):
        delattr(config, "architectures")

    torch.set_default_device("cuda")
    model = A2DQwen3LMHeadModel(config)
    model.save_pretrained("models-tmp/a2d-qwen3")
    auto_model = AutoModel.from_pretrained("models-tmp/a2d-qwen3")
