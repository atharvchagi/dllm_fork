"""Utilities for offline Loophole trajectory distillation.

Run the focused tests with
``pytest /nvme-data2/atharvchagi/dllm_fork/scripts/tests/test_loopholing.py -v``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset


TRACE_FORMAT_VERSION = 1
DEFAULT_TRACE_FRACTIONS = (0.875, 0.625, 0.375, 0.125)
TEACHER_HEAD_FILENAME = "teacher_lm_head.safetensors"


@dataclass
class TraceSnapshot:
    """One teacher forward selected from a threshold-decoding trajectory."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_positions: torch.Tensor
    teacher_hidden: torch.Tensor
    decode_step: int
    requested_fraction: float
    actual_fraction: float
    prompt_length: int
    initial_response_tokens: int


class ThresholdTraceCollector:
    """Collect distinct forwards as each sample crosses target mask fractions."""

    def __init__(
        self,
        prompt_lengths: Sequence[int],
        initial_response_tokens: int,
        target_fractions: Sequence[float] = DEFAULT_TRACE_FRACTIONS,
    ) -> None:
        fractions = tuple(float(value) for value in target_fractions)
        if not fractions:
            raise ValueError("At least one target fraction is required")
        if any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("Target fractions must be strictly between zero and one")
        if any(left <= right for left, right in zip(fractions, fractions[1:])):
            raise ValueError("Target fractions must be strictly decreasing")
        if initial_response_tokens <= 0:
            raise ValueError("initial_response_tokens must be positive")

        self.prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self.initial_response_tokens = int(initial_response_tokens)
        self.target_fractions = fractions
        self.next_fraction_index = [0 for _ in self.prompt_lengths]
        self.snapshots: list[list[TraceSnapshot]] = [
            [] for _ in self.prompt_lengths
        ]

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        loophole_state: torch.Tensor | None,
        global_step: int,
        **_: Any,
    ) -> None:
        """Capture at most one requested level per sample and model forward."""
        if loophole_state is None:
            raise RuntimeError("Trace collection requires a returned Loopholing state")
        batch_size = input_ids.shape[0]
        if batch_size != len(self.prompt_lengths):
            raise ValueError(
                f"Collector expected batch {len(self.prompt_lengths)}, got {batch_size}"
            )
        if loophole_state.shape[:2] != input_ids.shape:
            raise ValueError("Loopholing state must align with the token canvas")

        for row, prompt_length in enumerate(self.prompt_lengths):
            target_index = self.next_fraction_index[row]
            if target_index >= len(self.target_fractions):
                continue
            positions = candidate_mask[row].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            actual_fraction = positions.numel() / self.initial_response_tokens
            requested_fraction = self.target_fractions[target_index]
            if actual_fraction > requested_fraction:
                continue

            valid_end = min(
                prompt_length + self.initial_response_tokens,
                input_ids.shape[1],
            )
            positions = positions[positions < valid_end]
            if positions.numel() == 0:
                continue
            snapshot = TraceSnapshot(
                input_ids=input_ids[row, :valid_end].detach().to("cpu", torch.int32),
                attention_mask=attention_mask[row, :valid_end]
                .detach()
                .to("cpu", torch.uint8),
                target_positions=positions.detach().to("cpu", torch.int32),
                teacher_hidden=loophole_state[row, positions]
                .detach()
                .to("cpu", torch.bfloat16),
                decode_step=int(global_step),
                requested_fraction=requested_fraction,
                actual_fraction=float(actual_fraction),
                prompt_length=prompt_length,
                initial_response_tokens=self.initial_response_tokens,
            )
            self.snapshots[row].append(snapshot)
            self.next_fraction_index[row] += 1


def trim_snapshot_targets(
    snapshot: TraceSnapshot,
    completion_length: int,
) -> TraceSnapshot | None:
    """Remove target positions at or beyond the first generated stop token."""
    response_end = snapshot.prompt_length + int(completion_length)
    keep = snapshot.target_positions < response_end
    if not torch.any(keep):
        return None
    return TraceSnapshot(
        input_ids=snapshot.input_ids,
        attention_mask=snapshot.attention_mask,
        target_positions=snapshot.target_positions[keep],
        teacher_hidden=snapshot.teacher_hidden[keep],
        decode_step=snapshot.decode_step,
        requested_fraction=snapshot.requested_fraction,
        actual_fraction=snapshot.actual_fraction,
        prompt_length=snapshot.prompt_length,
        initial_response_tokens=snapshot.initial_response_tokens,
    )


def _offsets(lengths: Iterable[int]) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int64)


def save_trace_tensor_shard(
    path: str | Path,
    snapshots: Sequence[tuple[int, int, TraceSnapshot]],
    hidden_size: int,
) -> None:
    """Save flattened variable-length snapshots to one safetensors shard."""
    path = Path(path)
    if not snapshots:
        raise ValueError("Cannot save an empty trace tensor shard")
    for _, _, snapshot in snapshots:
        if snapshot.teacher_hidden.ndim != 2:
            raise ValueError("teacher_hidden must be rank two")
        if snapshot.teacher_hidden.shape[1] != hidden_size:
            raise ValueError("teacher_hidden does not match hidden_size")
        if len(snapshot.target_positions) != len(snapshot.teacher_hidden):
            raise ValueError("Each target position needs one hidden target")

    input_lengths = [len(item[2].input_ids) for item in snapshots]
    target_lengths = [len(item[2].target_positions) for item in snapshots]
    tensors = {
        "input_ids": torch.cat([item[2].input_ids for item in snapshots]),
        "attention_mask": torch.cat(
            [item[2].attention_mask for item in snapshots]
        ),
        "input_offsets": _offsets(input_lengths),
        "target_positions": torch.cat(
            [item[2].target_positions for item in snapshots]
        ),
        "target_offsets": _offsets(target_lengths),
        "teacher_hidden": torch.cat(
            [item[2].teacher_hidden for item in snapshots]
        ).contiguous(),
        "example_index": torch.tensor(
            [item[0] for item in snapshots], dtype=torch.int64
        ),
        "split_code": torch.tensor(
            [item[1] for item in snapshots], dtype=torch.int8
        ),
        "decode_step": torch.tensor(
            [item[2].decode_step for item in snapshots], dtype=torch.int32
        ),
        "requested_fraction": torch.tensor(
            [item[2].requested_fraction for item in snapshots], dtype=torch.float32
        ),
        "actual_fraction": torch.tensor(
            [item[2].actual_fraction for item in snapshots], dtype=torch.float32
        ),
        "prompt_length": torch.tensor(
            [item[2].prompt_length for item in snapshots], dtype=torch.int32
        ),
        "initial_response_tokens": torch.tensor(
            [item[2].initial_response_tokens for item in snapshots],
            dtype=torch.int32,
        ),
    }
    save_file(
        tensors,
        path,
        metadata={
            "format_version": str(TRACE_FORMAT_VERSION),
            "hidden_size": str(hidden_size),
            "split_codes": json.dumps({"train": 0, "validation": 1}),
        },
    )


def save_teacher_head(path: str | Path, weight: torch.Tensor) -> None:
    """Save the frozen teacher vocabulary readout as a standalone artifact."""
    if weight.ndim != 2:
        raise ValueError("Teacher head weight must be rank two")
    save_file(
        {"lm_head.weight": weight.detach().to("cpu", torch.bfloat16).contiguous()},
        Path(path),
        metadata={"format_version": str(TRACE_FORMAT_VERSION)},
    )


def load_teacher_head(path: str | Path) -> torch.Tensor:
    """Load a standalone teacher vocabulary readout on CPU."""
    tensors = load_file(Path(path), device="cpu")
    if "lm_head.weight" not in tensors:
        raise KeyError("Teacher-head artifact has no 'lm_head.weight' tensor")
    return tensors["lm_head.weight"]


def install_frozen_teacher_head(
    model: nn.Module,
    teacher_weight: torch.Tensor,
) -> nn.Linear:
    """Untie a Qwen readout, copy the teacher head, and freeze it."""
    if not hasattr(model, "lm_head") or not hasattr(model, "get_input_embeddings"):
        raise TypeError("Model must expose lm_head and get_input_embeddings()")
    embeddings = model.get_input_embeddings()
    if teacher_weight.shape != model.lm_head.weight.shape:
        raise ValueError(
            "Teacher head shape does not match student head: "
            f"{tuple(teacher_weight.shape)} != {tuple(model.lm_head.weight.shape)}"
        )
    new_head = nn.Linear(
        teacher_weight.shape[1],
        teacher_weight.shape[0],
        bias=False,
        device=embeddings.weight.device,
        dtype=embeddings.weight.dtype,
    )
    with torch.no_grad():
        new_head.weight.copy_(
            teacher_weight.to(
                device=new_head.weight.device,
                dtype=new_head.weight.dtype,
            )
        )
    new_head.weight.requires_grad_(False)
    model.lm_head = new_head
    model.config.tie_word_embeddings = False
    if model.lm_head.weight.data_ptr() == embeddings.weight.data_ptr():
        raise RuntimeError("Teacher readout remained tied to student embeddings")
    return new_head


class OfflineTraceDataset(Dataset):
    """Memory-mapped view over finalized offline trace tensor shards."""

    SPLIT_CODES = {"train": 0, "validation": 1}

    def __init__(self, trace_dir: str | Path, split: str) -> None:
        self.trace_dir = Path(trace_dir)
        if split not in self.SPLIT_CODES:
            raise ValueError(f"Unsupported split {split!r}")
        manifest_path = self.trace_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Finalized trace manifest does not exist: {manifest_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != TRACE_FORMAT_VERSION:
            raise ValueError("Unsupported trace manifest format version")

        shard_names = self.manifest.get("tensor_shards", [])
        if not shard_names:
            raise ValueError("Trace manifest contains no tensor shards")
        self.shard_paths = [self.trace_dir / name for name in shard_names]
        self._offsets_by_shard: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._indices: list[tuple[int, int]] = []
        split_code = self.SPLIT_CODES[split]
        for shard_index, shard_path in enumerate(self.shard_paths):
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing trace shard: {shard_path}")
            handle = safe_open(shard_path, framework="pt", device="cpu")
            input_offsets = handle.get_tensor("input_offsets")
            target_offsets = handle.get_tensor("target_offsets")
            split_codes = handle.get_tensor("split_code")
            if len(input_offsets) != len(split_codes) + 1:
                raise ValueError(f"Invalid input offsets in {shard_path}")
            if len(target_offsets) != len(split_codes) + 1:
                raise ValueError(f"Invalid target offsets in {shard_path}")
            self._offsets_by_shard.append((input_offsets, target_offsets))
            self._indices.extend(
                (shard_index, local_index)
                for local_index in (split_codes == split_code)
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
        self._handles: dict[int, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __len__(self) -> int:
        return len(self._indices)

    def _handle(self, shard_index: int):
        if shard_index not in self._handles:
            self._handles[shard_index] = safe_open(
                self.shard_paths[shard_index], framework="pt", device="cpu"
            )
        return self._handles[shard_index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, local_index = self._indices[index]
        input_offsets, target_offsets = self._offsets_by_shard[shard_index]
        input_start = int(input_offsets[local_index])
        input_end = int(input_offsets[local_index + 1])
        target_start = int(target_offsets[local_index])
        target_end = int(target_offsets[local_index + 1])
        handle = self._handle(shard_index)
        return {
            "input_ids": handle.get_slice("input_ids")[input_start:input_end].long(),
            "attention_mask": handle.get_slice("attention_mask")[
                input_start:input_end
            ].long(),
            "target_positions": handle.get_slice("target_positions")[
                target_start:target_end
            ].long(),
            "teacher_hidden": handle.get_slice("teacher_hidden")[
                target_start:target_end
            ],
        }


@dataclass
class OfflineTraceCollator:
    """Pad canvases while keeping hidden targets in a compact flat tensor."""

    pad_token_id: int

    def __call__(self, rows: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        target_batches = []
        target_positions = []
        teacher_hidden = []
        target_counts = []
        for batch_index, row in enumerate(rows):
            length = len(row["input_ids"])
            count = len(row["target_positions"])
            if count <= 0:
                raise ValueError("Every trace snapshot must contain a target")
            if int(row["target_positions"].max()) >= length:
                raise ValueError("A target position lies outside its token canvas")
            input_ids[batch_index, :length] = row["input_ids"]
            attention_mask[batch_index, :length] = row["attention_mask"]
            target_batches.append(
                torch.full((count,), batch_index, dtype=torch.long)
            )
            target_positions.append(row["target_positions"])
            teacher_hidden.append(row["teacher_hidden"])
            target_counts.append(count)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_batch_indices": torch.cat(target_batches),
            "target_positions": torch.cat(target_positions),
            "teacher_hidden": torch.cat(teacher_hidden),
            "target_counts": torch.tensor(target_counts, dtype=torch.long),
        }


def mean_per_snapshot(
    per_target_loss: torch.Tensor,
    target_batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Average targets within each snapshot, then average snapshots equally."""
    sums = torch.zeros(
        batch_size, device=per_target_loss.device, dtype=per_target_loss.dtype
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, target_batch_indices, per_target_loss)
    counts.scatter_add_(0, target_batch_indices, torch.ones_like(per_target_loss))
    if torch.any(counts == 0):
        raise ValueError("Every snapshot must contribute at least one target")
    return (sums / counts).mean()


def hidden_mse_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    target_batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Compute FP32 hidden MSE with equal weight for each snapshot."""
    per_target = (student_hidden.float() - teacher_hidden.float()).square().mean(-1)
    return mean_per_snapshot(per_target, target_batch_indices, batch_size)


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_batch_indices: torch.Tensor,
    batch_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute exact forward KL(teacher || student) in FP32."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_log_probs = torch.log_softmax(teacher_logits.float() / temperature, -1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = torch.log_softmax(student_logits.float() / temperature, -1)
    per_target = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1)
    per_target = per_target * temperature**2
    return mean_per_snapshot(per_target, target_batch_indices, batch_size)
