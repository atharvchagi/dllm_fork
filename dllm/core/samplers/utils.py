"""Shared diffusion-sampling helpers.

Run these helpers through an absolute pipeline entrypoint such as
``python /nvme-data2/atharvchagi/dllm_fork/examples/a2d/bd3lm/sample.py --help``.
"""

import torch

from dllm.core.schedulers import BaseAlphaScheduler


def select_transfer_index(
    confidence: torch.Tensor,
    candidate_mask: torch.Tensor,
    minimum_transfer_tokens: torch.Tensor,
    confidence_threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select threshold-qualified tokens with a per-sequence progress floor."""
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


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
