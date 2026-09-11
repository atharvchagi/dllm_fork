"""Unit tests for Qwen3 Loopholing training and sampling.

Run with ``pytest /scratch/user/atharvchagi_tamu.edu/dllm_fork/scripts/tests/test_loopholing.py -v``
after activating the ``dllm`` conda environment.
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from dllm.core.eval.mdlm import MDLMEvalHarness
from dllm.core.samplers.bd3lm import BD3LMSampler, BD3LMSamplerConfig
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.trainers.bd3lm import BD3LMTrainer
from dllm.core.trainers.mdlm import MDLMTrainer
from dllm.data.offpolicy_distillation import (
    OfflineTraceCollator,
    OfflineTraceDataset,
    TRACE_FORMAT_VERSION,
    ThresholdTraceCollector,
    TraceSnapshot,
    forward_kl_loss,
    hidden_mse_loss,
    install_frozen_teacher_head,
    save_trace_tensor_shard,
)
from dllm.pipelines.a2d.models.qwen3.modeling_qwen3 import (
    A2DQwen3Config,
    A2DQwen3LMHeadModel,
)
from dllm.utils.models import _ensure_mask_token


def test_existing_checkpoint_mask_token_is_preserved():
    tokenizer = SimpleNamespace(
        mask_token="<|MASK|>",
        add_special_tokens=Mock(),
    )

    _ensure_mask_token(tokenizer, "<|mask|>")

    tokenizer.add_special_tokens.assert_not_called()
    assert tokenizer.mask_token == "<|MASK|>"


def test_default_mask_token_is_added_when_checkpoint_has_none():
    tokenizer = SimpleNamespace(
        mask_token=None,
        add_special_tokens=Mock(),
    )

    _ensure_mask_token(tokenizer, "<|mask|>")

    tokenizer.add_special_tokens.assert_called_once_with({"mask_token": "<|mask|>"})


def _tiny_qwen3_config(loophole_enabled: bool = True) -> A2DQwen3Config:
    return A2DQwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        loophole_enabled=loophole_enabled,
    )


def test_qwen3_zero_initialized_loophole_is_baseline_equivalent():
    torch.manual_seed(0)
    model = A2DQwen3LMHeadModel(_tiny_qwen3_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        baseline = model(input_ids=input_ids)
        looped = model(
            input_ids=input_ids,
            loophole_enabled=True,
            return_loophole_state=True,
        )

    assert torch.equal(baseline.logits, looped.logits)
    assert looped.loophole_state.shape == (1, 3, 16)
    assert looped.loophole_state.dtype == baseline.logits.dtype
    assert torch.count_nonzero(model.model.loophole_norm.weight) == 0
    assert torch.count_nonzero(model.model.loophole_norm.bias) == 0


def test_qwen3_loophole_state_changes_logits_after_adapter_learns():
    torch.manual_seed(0)
    model = A2DQwen3LMHeadModel(_tiny_qwen3_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    state = torch.randn(1, 3, 16)

    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits
        model.model.loophole_norm.weight.fill_(1.0)
        looped = model(
            input_ids=input_ids,
            loophole_state=state,
            loophole_enabled=True,
            return_loophole_state=True,
        ).logits

    assert not torch.equal(baseline, looped)


def test_qwen3_disabled_forward_bypasses_learned_adapter():
    torch.manual_seed(0)
    model = A2DQwen3LMHeadModel(_tiny_qwen3_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits
        model.model.loophole_norm.bias.fill_(1.0)
        disabled = model(
            input_ids=input_ids,
            loophole_enabled=False,
        ).logits
        enabled = model(
            input_ids=input_ids,
            loophole_enabled=True,
            return_loophole_state=True,
        ).logits

    assert torch.equal(baseline, disabled)
    assert not torch.equal(baseline, enabled)


def test_qwen3_rejects_misaligned_loophole_state():
    model = A2DQwen3LMHeadModel(_tiny_qwen3_config()).eval()

    with pytest.raises(ValueError, match="loophole_state must have shape"):
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            loophole_state=torch.zeros(1, 2, 16),
            loophole_enabled=True,
            return_loophole_state=True,
        )


def test_qwen3_loophole_config_and_parameters_round_trip(tmp_path):
    model = A2DQwen3LMHeadModel(_tiny_qwen3_config())
    with torch.no_grad():
        model.model.loophole_norm.weight.fill_(0.25)
        model.model.loophole_norm.bias.fill_(-0.5)
    model.save_pretrained(tmp_path)

    restored = A2DQwen3LMHeadModel.from_pretrained(tmp_path)

    assert restored.config.loophole_enabled is True
    assert torch.equal(
        restored.model.loophole_norm.weight,
        model.model.loophole_norm.weight,
    )
    assert torch.equal(
        restored.model.loophole_norm.bias,
        model.model.loophole_norm.bias,
    )


def test_qwen3_baseline_checkpoint_can_be_upgraded_with_zero_adapter(tmp_path):
    baseline = A2DQwen3LMHeadModel(_tiny_qwen3_config(loophole_enabled=False))
    baseline.save_pretrained(tmp_path)
    upgraded_config = A2DQwen3Config.from_pretrained(tmp_path)
    upgraded_config.loophole_enabled = True

    upgraded = A2DQwen3LMHeadModel.from_pretrained(
        tmp_path,
        config=upgraded_config,
    )

    assert upgraded.model.loophole_norm is not None
    assert torch.count_nonzero(upgraded.model.loophole_norm.weight) == 0
    assert torch.count_nonzero(upgraded.model.loophole_norm.bias) == 0


class _RecordingLoopholeLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        loophole_state=None,
        return_loophole_state=False,
        loophole_enabled=None,
        position_ids=None,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": (
                    None
                    if attention_mask is None
                    else attention_mask.detach().clone()
                ),
                "grad_enabled": torch.is_grad_enabled(),
                "loophole_state": loophole_state,
                "loophole_enabled": loophole_enabled,
                "position_ids": (
                    None if position_ids is None else position_ids.detach().clone()
                ),
            }
        )
        hidden = self.embedding(input_ids)
        if loophole_state is not None:
            hidden = hidden + loophole_state
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            loophole_state=hidden if return_loophole_state else None,
        )


def _trainer_for_loophole_rate(train_rate: float, eval_rate: float = 1.0):
    trainer = MDLMTrainer.__new__(MDLMTrainer)
    trainer.loophole_enabled = True
    trainer.loophole_self_cond_rate = train_rate
    trainer.loophole_eval_self_cond_rate = eval_rate
    return trainer


def test_training_rate_one_uses_detached_pseudo_state():
    trainer = _trainer_for_loophole_rate(train_rate=1.0)
    model = _RecordingLoopholeLM().train()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    outputs = trainer._forward_with_loopholing(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    outputs.logits.sum().backward()

    assert len(model.calls) == 2
    assert model.calls[0]["grad_enabled"] is False
    assert model.calls[1]["grad_enabled"] is True
    assert model.calls[1]["loophole_state"].requires_grad is False
    assert torch.equal(model.calls[0]["input_ids"], model.calls[1]["input_ids"])
    assert torch.equal(
        model.calls[0]["attention_mask"], model.calls[1]["attention_mask"]
    )
    assert model.embedding.weight.grad is not None


def test_training_rate_zero_uses_one_differentiable_forward():
    trainer = _trainer_for_loophole_rate(train_rate=0.0)
    model = _RecordingLoopholeLM().train()

    trainer._forward_with_loopholing(
        model=model,
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=None,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["grad_enabled"] is True
    assert model.calls[0]["loophole_state"] is None


def test_evaluation_uses_evaluation_self_conditioning_rate():
    trainer = _trainer_for_loophole_rate(train_rate=0.0, eval_rate=1.0)
    model = _RecordingLoopholeLM().eval()

    with torch.no_grad():
        trainer._forward_with_loopholing(
            model=model,
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=None,
        )

    assert len(model.calls) == 2
    assert all(call["grad_enabled"] is False for call in model.calls)


def test_bd3lm_training_self_conditions_the_concatenated_forward():
    trainer = BD3LMTrainer.__new__(BD3LMTrainer)
    trainer.loophole_enabled = True
    trainer.loophole_self_cond_rate = 1.0
    trainer.loophole_eval_self_cond_rate = 1.0
    trainer.right_shift_logits = False
    trainer._make_student_attention_mask = Mock(
        return_value=torch.ones(1, 1, 4, 4, dtype=torch.bool)
    )
    model = _RecordingLoopholeLM().train()
    input_ids = torch.tensor([[1, 2]])
    noised_input_ids = torch.tensor([[7, 2]])

    outputs, logits = trainer._forward_student(
        model=model,
        input_ids=input_ids,
        noised_input_ids=noised_input_ids,
    )
    logits.sum().backward()

    expected_concat = torch.tensor([[7, 2, 1, 2]])
    expected_positions = torch.tensor([[0, 1, 0, 1]])
    assert outputs.logits.shape[:2] == expected_concat.shape
    assert logits.shape[:2] == input_ids.shape
    assert len(model.calls) == 2
    assert model.calls[0]["grad_enabled"] is False
    assert model.calls[1]["grad_enabled"] is True
    assert model.calls[1]["loophole_state"].requires_grad is False
    assert torch.equal(model.calls[0]["input_ids"], expected_concat)
    assert torch.equal(model.calls[1]["input_ids"], expected_concat)
    assert torch.equal(model.calls[0]["position_ids"], expected_positions)
    assert torch.equal(model.calls[1]["position_ids"], expected_positions)
    assert model.embedding.weight.grad is not None


def test_bd3lm_training_aligns_loophole_state_for_a2d_right_shift():
    trainer = BD3LMTrainer.__new__(BD3LMTrainer)
    trainer.loophole_enabled = True
    trainer.loophole_self_cond_rate = 1.0
    trainer.loophole_eval_self_cond_rate = 1.0
    trainer.right_shift_logits = True
    trainer._make_student_attention_mask = Mock(
        return_value=torch.ones(1, 1, 4, 4, dtype=torch.bool)
    )
    model = _RecordingLoopholeLM().train()
    input_ids = torch.tensor([[1, 2]])
    noised_input_ids = torch.tensor([[7, 2]])

    trainer._forward_student(
        model=model,
        input_ids=input_ids,
        noised_input_ids=noised_input_ids,
    )

    concat_input_ids = torch.tensor([[7, 2, 1, 2]])
    with torch.no_grad():
        raw_state = model.embedding(concat_input_ids)
    expected_state = torch.cat(
        [
            torch.zeros_like(raw_state[:, :1]),
            raw_state[:, :1],
            torch.zeros_like(raw_state[:, 2:3]),
            raw_state[:, 2:3],
        ],
        dim=1,
    )
    assert torch.equal(model.calls[1]["loophole_state"], expected_state)


def test_eval_harness_uses_two_pass_loophole_logits():
    harness = MDLMEvalHarness.__new__(MDLMEvalHarness)
    harness.model = _RecordingLoopholeLM().eval()
    harness.loophole_enabled = True
    batch = torch.tensor([[1, 2, 3]])

    logits = harness._get_logits(
        batch=batch,
        prompt_index=torch.tensor([True, False, False]),
    )

    assert logits.shape[:2] == batch.shape
    assert len(harness.model.calls) == 2
    assert harness.model.calls[1]["loophole_state"].requires_grad is False


class _SamplerLoopholeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    def forward(
        self,
        input_ids,
        attention_mask=None,
        loophole_state=None,
        return_loophole_state=False,
        loophole_enabled=None,
    ):
        self.calls.append(loophole_state)
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device)
        logits[..., 3] = 10.0
        next_state = torch.full(
            (batch, length, 4),
            fill_value=float(len(self.calls)),
            device=input_ids.device,
        )
        return SimpleNamespace(
            logits=logits,
            loophole_state=next_state if return_loophole_state else None,
        )


class _BD3LMSamplerLoopholeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []
        self.block_forward_count = 0

    @property
    def device(self):
        return self.anchor.device

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        loophole_state=None,
        return_loophole_state=False,
        loophole_enabled=None,
    ):
        is_prefix = use_cache is True
        if not is_prefix:
            self.block_forward_count += 1
        self.calls.append(
            {
                "is_prefix": is_prefix,
                "loophole_state": loophole_state,
                "return_loophole_state": return_loophole_state,
                "loophole_enabled": loophole_enabled,
            }
        )

        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device)
        logits[..., 3] = 10.0
        state_value = -1.0 if is_prefix else float(self.block_forward_count)
        next_state = torch.full(
            (batch, length, 4),
            fill_value=state_value,
            device=input_ids.device,
        )
        return SimpleNamespace(
            logits=logits,
            past_key_values=(torch.zeros(1, device=input_ids.device),),
            loophole_state=next_state if return_loophole_state else None,
        )


def test_sampler_carries_state_between_steps_and_resets_between_requests():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    config = MDLMSamplerConfig(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        loophole_enabled=True,
    )

    sampler.sample(inputs=[[1]], config=config)
    first_request_call_count = len(model.calls)
    assert first_request_call_count == 2
    assert model.calls[0] is None
    assert torch.equal(model.calls[1], torch.ones_like(model.calls[1]))

    sampler.sample(inputs=[[1]], config=config)
    assert model.calls[first_request_call_count] is None


def test_bd3lm_sampler_carries_state_within_blocks_and_resets_between_blocks():
    model = _BD3LMSamplerLoopholeLM()
    tokenizer = SimpleNamespace(
        mask_token_id=7,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=6,
    )
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    config = BD3LMSamplerConfig(
        max_new_tokens=4,
        block_size=2,
        steps=4,
        loophole_enabled=True,
    )

    sampler.sample(inputs=[[1]], config=config)

    block_calls = [call for call in model.calls if not call["is_prefix"]]
    assert len(block_calls) == 4
    assert block_calls[0]["loophole_state"] is None
    assert torch.equal(
        block_calls[1]["loophole_state"],
        torch.ones_like(block_calls[1]["loophole_state"]),
    )
    assert block_calls[2]["loophole_state"] is None
    assert torch.equal(
        block_calls[3]["loophole_state"],
        torch.full_like(block_calls[3]["loophole_state"], 3.0),
    )
    assert all(call["return_loophole_state"] for call in block_calls)
    assert all(call["loophole_enabled"] is True for call in block_calls)
    prefix_calls = [call for call in model.calls if call["is_prefix"]]
    assert all(call["loophole_enabled"] is True for call in prefix_calls)


def test_bd3lm_sampler_keeps_cfg_loophole_states_separate():
    model = _BD3LMSamplerLoopholeLM()
    tokenizer = SimpleNamespace(
        mask_token_id=7,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=6,
    )
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    config = BD3LMSamplerConfig(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        cfg_scale=1.0,
        loophole_enabled=True,
    )

    sampler.sample(inputs=[[1]], config=config)

    block_calls = [call for call in model.calls if not call["is_prefix"]]
    assert len(block_calls) == 4
    assert block_calls[0]["loophole_state"] is None
    assert block_calls[1]["loophole_state"] is None
    assert torch.equal(
        block_calls[2]["loophole_state"],
        torch.ones_like(block_calls[2]["loophole_state"]),
    )
    assert torch.equal(
        block_calls[3]["loophole_state"],
        torch.full_like(block_calls[3]["loophole_state"], 2.0),
    )


def test_bd3lm_sampler_aligns_loophole_state_for_a2d_right_shift():
    model = _BD3LMSamplerLoopholeLM()
    tokenizer = SimpleNamespace(
        mask_token_id=7,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=6,
    )
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)

    sampler.sample(
        inputs=[[1]],
        config=BD3LMSamplerConfig(
            max_new_tokens=2,
            block_size=2,
            steps=2,
            loophole_enabled=True,
            right_shift_logits=True,
        ),
    )

    prefix_calls = [call for call in model.calls if call["is_prefix"]]
    block_calls = [call for call in model.calls if not call["is_prefix"]]
    assert len(prefix_calls) == 1
    assert prefix_calls[0]["return_loophole_state"] is True
    assert len(block_calls) == 2
    assert block_calls[0]["loophole_state"] is None
    expected_state = torch.tensor([[[-1.0] * 4, [1.0] * 4]])
    assert torch.equal(block_calls[1]["loophole_state"], expected_state)


def test_sampler_returns_entropy_for_each_masked_generation_step():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    config = MDLMSamplerConfig(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        return_dict=True,
        return_step_entropy=True,
    )

    output = sampler.sample(inputs=[[1]], config=config)

    assert len(output.step_metrics) == 2
    assert [metric["step"] for metric in output.step_metrics] == [1, 2]
    assert [metric["remaining_masks"].item() for metric in output.step_metrics] == [
        2,
        1,
    ]
    assert [metric["transferred_tokens"].item() for metric in output.step_metrics] == [
        1,
        1,
    ]
    for metric in output.step_metrics:
        assert torch.isfinite(metric["mean_entropy_nats"]).all()
        assert torch.all(metric["mean_top1_probability"] > 0)
        assert torch.all(metric["mean_top1_probability"] <= 1)


def test_masked_prediction_entropy_matches_uniform_distribution():
    logits = torch.zeros(1, 2, 8)
    active_mask = torch.tensor([[True, False]])

    metrics = MDLMSampler._masked_prediction_metrics(logits, active_mask)

    assert metrics["mean_entropy_nats"].item() == pytest.approx(math.log(8))
    assert metrics["mean_top1_probability"].item() == pytest.approx(1 / 8)
    assert metrics["remaining_masks"].item() == 1


def test_confidence_threshold_transfers_all_qualified_tokens_early():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    config = MDLMSamplerConfig(
        max_new_tokens=4,
        block_size=4,
        steps=4,
        confidence_threshold=0.85,
        return_dict=True,
        return_step_entropy=True,
    )

    output = sampler.sample(inputs=[[1]], config=config)

    assert len(model.calls) == 1
    assert len(output.step_metrics) == 1
    metrics = output.step_metrics[0]
    assert metrics["transferred_tokens"].item() == 4
    assert metrics["threshold_accepted_tokens"].item() == 4
    assert metrics["below_threshold_fallback_tokens"].item() == 0
    assert torch.equal(output.sequences, torch.tensor([[1, 3, 3, 3, 3]]))


def test_confidence_threshold_uses_schedule_as_completion_floor():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    config = MDLMSamplerConfig(
        max_new_tokens=2,
        block_size=2,
        steps=2,
        confidence_threshold=0.99999,
        return_dict=True,
        return_step_entropy=True,
    )

    output = sampler.sample(inputs=[[1]], config=config)

    assert len(model.calls) == 2
    assert [
        metric["below_threshold_fallback_tokens"].item()
        for metric in output.step_metrics
    ] == [1, 1]
    assert tokenizer.mask_token_id not in output.sequences


def test_sampler_exposes_pre_transfer_state_to_distillation_callback():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    callback_rows = []

    sampler.sample(
        inputs=[[1]],
        config=MDLMSamplerConfig(
            max_new_tokens=2,
            block_size=2,
            steps=2,
            loophole_enabled=True,
            confidence_threshold=0.99999,
            return_dict=True,
            return_history=False,
        ),
        distillation_step_callback=lambda **row: callback_rows.append(row),
    )

    assert len(callback_rows) == 2
    assert [row["global_step"] for row in callback_rows] == [1, 2]
    assert [row["candidate_mask"].sum().item() for row in callback_rows] == [2, 1]
    assert callback_rows[0]["loophole_state"].shape == (1, 3, 4)


def test_confidence_threshold_rejects_invalid_configuration():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="between 0 and 1"):
        sampler.sample(
            inputs=[[1]],
            config=MDLMSamplerConfig(confidence_threshold=1.01),
        )

    with pytest.raises(ValueError, match="requires remasking"):
        sampler.sample(
            inputs=[[1]],
            config=MDLMSamplerConfig(
                confidence_threshold=0.85,
                remasking="random",
            ),
        )


def test_infill_carries_state_between_steps():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    config = MDLMSamplerConfig(
        block_size=2,
        steps=2,
        loophole_enabled=True,
    )

    sampler.infill(inputs=[[7, 7]], config=config)

    assert len(model.calls) == 2
    assert model.calls[0] is None
    assert torch.equal(model.calls[1], torch.ones_like(model.calls[1]))


def test_infill_confidence_threshold_can_finish_a_block_early():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    config = MDLMSamplerConfig(
        block_size=4,
        steps=4,
        confidence_threshold=0.85,
    )

    output = sampler.infill(inputs=[[7, 7, 7, 7]], config=config)

    assert len(model.calls) == 1
    assert torch.equal(output, torch.tensor([[3, 3, 3, 3]]))


def test_sampler_rejects_loopholing_with_right_shift():
    model = _SamplerLoopholeLM()
    tokenizer = SimpleNamespace(mask_token_id=7, bos_token_id=1, eos_token_id=0)
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="right_shift_logits"):
        sampler.sample(
            inputs=[[1]],
            config=MDLMSamplerConfig(
                max_new_tokens=1,
                steps=1,
                loophole_enabled=True,
                right_shift_logits=True,
            ),
        )


def _distillation_snapshot(value: float, *, decode_step: int = 1) -> TraceSnapshot:
    return TraceSnapshot(
        input_ids=torch.tensor([1, 9, 9, 2], dtype=torch.int32),
        attention_mask=torch.ones(4, dtype=torch.uint8),
        target_positions=torch.tensor([1, 2], dtype=torch.int32),
        teacher_hidden=torch.full((2, 3), value, dtype=torch.bfloat16),
        decode_step=decode_step,
        requested_fraction=0.5,
        actual_fraction=0.5,
        prompt_length=1,
        initial_response_tokens=3,
    )


def test_offpolicy_trace_dataset_and_collator_round_trip(tmp_path):
    shard = tmp_path / "tensors-00000.safetensors"
    save_trace_tensor_shard(
        shard,
        [
            (0, 0, _distillation_snapshot(1.0)),
            (1, 1, _distillation_snapshot(2.0, decode_step=2)),
        ],
        hidden_size=3,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": TRACE_FORMAT_VERSION,
                "tensor_shards": [shard.name],
            }
        ),
        encoding="utf-8",
    )

    train = OfflineTraceDataset(tmp_path, split="train")
    validation = OfflineTraceDataset(tmp_path, split="validation")
    assert len(train) == 1
    assert len(validation) == 1
    assert torch.equal(train[0]["input_ids"], torch.tensor([1, 9, 9, 2]))
    assert torch.equal(
        validation[0]["teacher_hidden"],
        torch.full((2, 3), 2.0, dtype=torch.bfloat16),
    )

    batch = OfflineTraceCollator(pad_token_id=0)([train[0], validation[0]])
    assert batch["input_ids"].shape == (2, 4)
    assert batch["teacher_hidden"].shape == (4, 3)
    assert torch.equal(batch["target_batch_indices"], torch.tensor([0, 0, 1, 1]))


def test_offpolicy_collector_captures_requested_levels_in_order():
    collector = ThresholdTraceCollector(
        prompt_lengths=[2],
        initial_response_tokens=4,
        target_fractions=(0.75, 0.25),
    )
    input_ids = torch.tensor([[1, 2, 9, 9, 9, 9]])
    attention_mask = torch.ones_like(input_ids)
    state = torch.randn(1, 6, 3)

    collector(
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_mask=torch.tensor([[False, False, True, True, True, False]]),
        loophole_state=state,
        global_step=2,
    )
    collector(
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_mask=torch.tensor([[False, False, True, False, False, False]]),
        loophole_state=state,
        global_step=4,
    )

    snapshots = collector.snapshots[0]
    assert [snapshot.requested_fraction for snapshot in snapshots] == [0.75, 0.25]
    assert [snapshot.decode_step for snapshot in snapshots] == [2, 4]
    assert [snapshot.actual_fraction for snapshot in snapshots] == [0.75, 0.25]
    assert snapshots[0].teacher_hidden.shape == (3, 3)


def test_offpolicy_losses_have_expected_fixed_points():
    batch_indices = torch.tensor([0, 0, 1])
    teacher_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
    assert hidden_mse_loss(
        teacher_hidden,
        teacher_hidden,
        batch_indices,
        batch_size=2,
    ).item() == 0.0

    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, -0.5], [2.0, 2.0, 2.0]])
    identical_kl = forward_kl_loss(logits, logits, batch_indices, batch_size=2)
    changed_kl = forward_kl_loss(
        logits.flip(-1),
        logits,
        batch_indices,
        batch_size=2,
    )
    assert abs(identical_kl.item()) < 1e-6
    assert changed_kl.item() > 0.0


class _TiedDistillationToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embedding = nn.Embedding(5, 3)
        self.lm_head = nn.Linear(3, 5, bias=False)
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding


def test_offpolicy_teacher_head_is_untied_and_frozen():
    model = _TiedDistillationToyModel()
    teacher_weight = torch.randn(5, 3)

    install_frozen_teacher_head(model, teacher_weight)

    assert model.config.tie_word_embeddings is False
    assert model.embedding.weight.requires_grad is True
    assert model.lm_head.weight.requires_grad is False
    assert model.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert torch.equal(model.lm_head.weight, teacher_weight)
