"""Off-policy and staleness signals computed from per-token tensors logged during training.

Everything here is pure numpy so it can be unit-tested on synthetic arrays. The trainer subclasses hand in
``log_ratio = log pi_current - log pi_behaviour`` for every trained completion token, where the behaviour policy
is the vLLM sampler that generated the token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor, without importing torch here
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def offpolicy_stats(log_ratio, is_clip_min: float = 0.5, is_clip_max: float = 5.0) -> dict[str, float]:
    """Summary of one step's per-token log-ratios.

    ``is_outside_frac`` is the fraction of tokens whose ratio exp(log_ratio) falls outside
    [is_clip_min, is_clip_max], i.e. the tokens masked out of the loss by TRL's masked importance sampling.
    Non-finite entries (e.g. tokens vLLM returned no logprob for) are ignored.
    """
    lr = _to_numpy(log_ratio)
    lr = lr[np.isfinite(lr)]
    if lr.size == 0:
        return {}
    abs_lr = np.abs(lr)
    ratio = np.exp(np.clip(lr, -50.0, 50.0))
    return {
        "logratio_mean": float(lr.mean()),
        "logratio_std": float(lr.std()),
        "logratio_abs_mean": float(abs_lr.mean()),
        "logratio_abs_p95": float(np.quantile(abs_lr, 0.95)),
        "logratio_abs_max": float(abs_lr.max()),
        "is_outside_frac": float(((ratio < is_clip_min) | (ratio > is_clip_max)).mean()),
        "logratio_tokens": int(lr.size),
    }


@dataclass
class LogRatioAccumulator:
    """Collects per-token log-ratios across the micro-batches of one optimizer step."""

    chunks: list = field(default_factory=list)

    def add(self, log_ratio) -> None:
        self.chunks.append(_to_numpy(log_ratio))

    def pop_stats(self, **kwargs) -> dict[str, float]:
        if not self.chunks:
            return {}
        stats = offpolicy_stats(np.concatenate(self.chunks), **kwargs)
        self.chunks = []
        return stats


class EMA:
    """Exponential moving average that ignores None/NaN updates; ``value`` is None until the first real update."""

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float | None) -> float | None:
        if x is None or not np.isfinite(x):
            return self.value
        self.value = x if self.value is None else self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


def ema_series(values: list[float | None], alpha: float) -> list[float | None]:
    """EMA of a whole series, one output per input (handy for analysis and tests)."""
    ema = EMA(alpha)
    return [ema.update(v) for v in values]
