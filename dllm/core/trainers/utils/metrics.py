"""
Token-level metrics for evaluation.

- NLLMetric: token-level mean negative log-likelihood.
- PPLMetric: exp(mean NLL) = perplexity.
- AccuracyMetric: token-level mean accuracy.

All metrics use sync_on_compute=True so compute() aggregates over all ranks.
"""

import torch
import torchmetrics


class _KwargMeanMetric(torchmetrics.aggregation.MeanMetric):
    """MeanMetric that reads value/weight from keyword arguments."""

    def __init__(self, value_key: str, **kwargs):
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)
        self.value_key = value_key

    def update(self, *args, **kwargs):
        # Backward compatibility: older code paths may pass value=... directly.
        if self.value_key in kwargs:
            value = kwargs[self.value_key]
        else:
            value = kwargs.get("value", None)
        if value is None:
            raise ValueError(
                f"Missing metric value for key '{self.value_key}' in kwargs."
            )
        weight = kwargs.get("weight", None)
        return super().update(value, weight=weight)


class NLLMetric(_KwargMeanMetric):
    """Token-level mean NLL. Weights should be the mask of predicted (e.g. masked) tokens."""

    def __init__(self, **kwargs):
        super().__init__(value_key="token_nll", **kwargs)


class AccuracyMetric(_KwargMeanMetric):
    """Token-level mean accuracy. Weights should mask valid evaluated tokens."""

    def __init__(self, **kwargs):
        super().__init__(value_key="token_acc", **kwargs)


class PPLMetric(NLLMetric):
    """Token-level perplexity = exp(mean NLL)."""

    def compute(self) -> torch.Tensor:
        mean_nll = super().compute()
        return torch.exp(mean_nll)
