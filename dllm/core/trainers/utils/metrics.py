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
    """MeanMetric supporting trainer-specific keywords and positional updates."""

    def __init__(self, value_key: str, **kwargs):
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)
        self.value_key = value_key

    def update(self, *args, **kwargs):
        # Trainer callbacks use the metric-specific key, while direct callers and
        # older code use the ordinary MeanMetric positional/value conventions.
        if self.value_key in kwargs:
            value = kwargs[self.value_key]
        elif "value" in kwargs:
            value = kwargs["value"]
        elif args:
            value = args[0]
        else:
            value = None
        if value is None:
            raise ValueError(
                f"Missing metric value for key '{self.value_key}'."
            )

        positional_weight = args[1] if len(args) > 1 else None
        if len(args) > 2:
            raise TypeError("Metric update accepts at most value and weight")
        if positional_weight is not None and "weight" in kwargs:
            raise TypeError("Metric weight was provided both positionally and by keyword")
        weight = kwargs.get("weight", positional_weight)
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
