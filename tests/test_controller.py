import multiprocessing as mp
import queue
import types

import numpy as np
import pytest

from src.controller import (
    ControllerConfig,
    RuleController,
    apply_knob,
    get_knob,
    set_learning_rate,
    set_max_staleness,
)


def rows(signal_values, reward=0.5, start=1):
    return [{"step": start + i, "logratio_abs_p95": s, "reward": reward} for i, s in enumerate(signal_values)]


def run(ctrl, history):
    return [d for d in (ctrl.observe(r) for r in history) if d is not None]


def cfg(**kw):
    base = dict(high=0.5, low=0.2, every_k=10, warmup_steps=10, cooldown_steps=20, ema_alpha=1.0)
    return ControllerConfig(**{**base, **kw})


def test_config_validation_and_defaults():
    with pytest.raises(ValueError):
        ControllerConfig(high=0.1, low=0.2)
    with pytest.raises(ValueError):
        ControllerConfig(high=0.2, low=0.1, knob="batch_size")
    c = ControllerConfig(high=0.2, low=0.1)
    assert (c.min_value, c.max_value, c.tighten, c.loosen) == (1, 8, -1, 1)
    lr = ControllerConfig(high=0.2, low=0.1, knob="learning_rate")
    assert lr.multiplicative and lr.tighten == 0.7 and lr.loosen == 1.2


def test_decides_only_every_k_after_warmup():
    ctrl = RuleController(cfg(), initial_value=4)
    decisions = run(ctrl, rows([0.3] * 35))
    assert [d.step for d in decisions] == [10, 20, 30]
    assert all(not d.changed and d.reason.startswith("hold: drift") for d in decisions)


def test_tightens_on_high_drift_then_respects_cooldown_and_min():
    ctrl = RuleController(cfg(cooldown_steps=20), initial_value=2)
    decisions = run(ctrl, rows([0.9] * 60))
    # step 10: 2 -> 1; step 20: cooldown; step 30: at min; step 40, 50: at min
    assert [(d.step, d.old_value, d.new_value) for d in decisions][:2] == [(10, 2, 1), (20, 1, 1)]
    assert "cooldown" in decisions[1].reason
    assert "already at min" in decisions[2].reason
    assert ctrl.value == 1


def test_loosens_on_low_drift_up_to_max():
    ctrl = RuleController(cfg(cooldown_steps=10, max_value=6), initial_value=4)
    decisions = run(ctrl, rows([0.05] * 50))
    assert [d.new_value for d in decisions if d.changed] == [5, 6]
    assert "already at max" in decisions[-1].reason
    assert isinstance(ctrl.value, int)


def test_does_not_loosen_when_reward_falling():
    ctrl = RuleController(cfg(), initial_value=4)
    history = [{"step": s, "logratio_abs_p95": 0.05, "reward": 1.0 - 0.02 * s} for s in range(1, 11)]
    (decision,) = run(ctrl, history)
    assert not decision.changed and "reward falling" in decision.reason


def test_ema_smooths_a_single_spike():
    # One spike of 5.0 at the decision step; the raw value would trip high=1.0, the EMA (0.1*5 + 0.9*0.3) does not.
    ctrl = RuleController(cfg(ema_alpha=0.1, high=1.0), initial_value=4)
    (decision,) = run(ctrl, rows([0.3] * 9 + [5.0]))
    assert decision.drift == pytest.approx(0.77)
    assert not decision.changed


def test_missing_signal_holds():
    ctrl = RuleController(cfg(), initial_value=4)
    (decision,) = run(ctrl, [{"step": s, "reward": 0.5} for s in range(1, 11)])
    assert not decision.changed and "no logratio_abs_p95" in decision.reason


def test_learning_rate_knob_is_multiplicative_and_clamped():
    ctrl = RuleController(cfg(knob="learning_rate", cooldown_steps=10, min_value=5e-7), initial_value=1e-6)
    decisions = run(ctrl, rows([0.9] * 30))
    assert decisions[0].new_value == pytest.approx(7e-7)
    assert decisions[1].new_value == pytest.approx(5e-7)  # 4.9e-7 clamped


def test_decision_dict_is_json_ready():
    d = run(RuleController(cfg(), 4), rows([0.9] * 10))[0].to_dict()
    assert d["changed"] is True and d["old_value"] == 4 and d["new_value"] == 3


# -- knob setters -------------------------------------------------------------------------------------------------


def test_set_learning_rate_updates_optimizer_and_constant_scheduler():
    torch = pytest.importorskip("torch")
    from transformers import get_constant_schedule

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-6)
    sched = get_constant_schedule(opt)
    trainer = types.SimpleNamespace(optimizer=opt, lr_scheduler=sched, args=None)
    set_learning_rate(trainer, 3e-6)
    p.grad = torch.zeros(1)
    opt.step()
    sched.step()  # the scheduler must not reset lr back to the old base
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-6)
    assert get_knob(trainer, "learning_rate") == pytest.approx(3e-6)


def _sample(version, n_tokens=4):
    from trl.trainer.async_grpo_trainer import RolloutSample

    return RolloutSample(
        prompt=[],
        completion=[],
        input_ids=np.arange(n_tokens),
        completion_mask=np.ones(n_tokens),
        old_log_probs=np.zeros(n_tokens),
        sampling_mask=[],
        sampler_ids=np.zeros((n_tokens, 0)),
        sampler_log_probs=np.zeros((n_tokens, 0)),
        advantage=1.0,
        model_version=version,
        group_id=0,
        metrics={},
    )


def test_set_max_staleness_changes_real_trl_drop_filter():
    """Mutating max_staleness on a live TRL RolloutQueueDataset changes which samples get dropped."""
    pytest.importorskip("trl")
    from collections import defaultdict

    from trl import AsyncGRPOConfig
    from trl.trainer.async_grpo_trainer import RolloutQueueDataset

    q = queue.Queue()
    current = {"v": 5}
    metrics = defaultdict(list)
    ds = RolloutQueueDataset(
        rollout_queue=q,
        model_version_fn=lambda: current["v"],
        check_health_fn=lambda _: None,
        stale_after_s=1.0,
        metrics=metrics,
        dropped_groups=set(),
        token_allowance=mp.Value("q", 0),
        max_staleness=4,
        poll_interval_s=0.01,
    )
    trainer = types.SimpleNamespace(
        args=AsyncGRPOConfig(output_dir="/tmp/unused", report_to="none", max_staleness=4), _rollout_dataset=ds
    )
    it = iter(ds)

    q.put(_sample(version=2))  # staleness 3 <= 4: kept
    assert next(it)["advantage"] == 1.0

    apply_knob(trainer, "max_staleness", 1)
    assert trainer.args.max_staleness == 1 and ds.max_staleness == 1
    q.put(_sample(version=2))  # staleness 3 > 1: dropped
    q.put(_sample(version=4))  # staleness 1: kept
    assert next(it) is not None
    assert metrics["sample/dropped_stale_count"] == [1.0]
    assert metrics["sample/staleness_max"] == [3.0, 1.0]


def test_set_max_staleness_without_dataset_and_rejects_negative():
    trainer = types.SimpleNamespace(args=types.SimpleNamespace(max_staleness=4))
    set_max_staleness(trainer, 2)
    assert get_knob(trainer, "max_staleness") == 2
    with pytest.raises(ValueError):
        set_max_staleness(trainer, -1)
