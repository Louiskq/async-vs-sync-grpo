import math

import numpy as np
import pytest

from src.signals import EMA, LogRatioAccumulator, ema_series, offpolicy_stats


def test_offpolicy_stats_on_policy_is_zero():
    s = offpolicy_stats(np.zeros(1000))
    assert s["logratio_mean"] == 0.0
    assert s["logratio_abs_p95"] == 0.0
    assert s["is_outside_frac"] == 0.0
    assert s["logratio_tokens"] == 1000


def test_offpolicy_stats_known_distribution():
    rng = np.random.default_rng(0)
    lr = rng.normal(0.0, 0.1, size=200_000)
    s = offpolicy_stats(lr)
    assert s["logratio_std"] == pytest.approx(0.1, rel=0.02)
    # p95 of |N(0, sigma)| is 1.96 sigma
    assert s["logratio_abs_p95"] == pytest.approx(0.196, rel=0.02)
    assert s["logratio_abs_mean"] == pytest.approx(0.1 * math.sqrt(2 / math.pi), rel=0.02)


def test_offpolicy_stats_outside_fraction_uses_ratio_bounds():
    # ratios: e^-1 = 0.37 (< 0.5, out), 1.0 (in), e^1 = 2.7 (in), e^2 = 7.4 (> 5, out)
    s = offpolicy_stats([-1.0, 0.0, 1.0, 2.0], is_clip_min=0.5, is_clip_max=5.0)
    assert s["is_outside_frac"] == 0.5
    assert s["logratio_abs_max"] == 2.0


def test_offpolicy_stats_ignores_non_finite_and_empty():
    s = offpolicy_stats([0.1, float("nan"), float("-inf"), -0.1])
    assert s["logratio_tokens"] == 2
    assert offpolicy_stats([]) == {}
    assert offpolicy_stats([float("nan")]) == {}


def test_offpolicy_stats_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    s = offpolicy_stats(torch.tensor([[0.5, -0.5]], dtype=torch.bfloat16))
    assert s["logratio_abs_max"] == 0.5


def test_accumulator_concatenates_microbatches_then_resets():
    acc = LogRatioAccumulator()
    acc.add(np.zeros(3))
    acc.add(np.array([1.0]))
    s = acc.pop_stats()
    assert s["logratio_tokens"] == 4
    assert s["logratio_abs_max"] == 1.0
    assert acc.pop_stats() == {}


def test_ema():
    e = EMA(0.5)
    assert e.update(None) is None
    assert e.update(2.0) == 2.0  # first value initialises
    assert e.update(4.0) == 3.0
    assert e.update(float("nan")) == 3.0  # NaN ignored
    assert ema_series([1.0, None, 3.0], 0.5) == [1.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        EMA(0.0)
