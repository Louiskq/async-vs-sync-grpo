"""Within-run hyperparameter control from async off-policy signals.

``RuleController`` follows the SPEC: every ``every_k`` steps it looks at ``drift = EMA(signal)`` and

* tightens the knob when ``drift > high`` (staleness - 1, or LR x 0.7),
* loosens it when ``drift < low`` and the reward EMA is not falling (staleness + 1, or LR x 1.2),
* otherwise holds.

Hysteresis comes from the gap between ``low`` and ``high`` plus a cooldown after every change. Thresholds must be
set from pilot data (see ``analysis/thresholds.py``), never from the runs being reported.

The controller sees only logged metric rows, never the trainer, so it is testable on synthetic histories. The
``set_*`` functions at the bottom are the only code that touches TRL internals; ``docs/notes.md`` explains why
they are sufficient for AsyncGRPOTrainer in TRL 1.18.2.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.signals import ema_series

KNOBS = ("max_staleness", "learning_rate")


@dataclass
class ControllerConfig:
    high: float
    low: float
    knob: str = "max_staleness"
    signal: str = "logratio_abs_p95"
    every_k: int = 10
    warmup_steps: int = 10
    cooldown_steps: int = 20
    ema_alpha: float = 0.2
    min_value: float | None = None
    max_value: float | None = None
    tighten: float | None = None  # additive for max_staleness, multiplicative for learning_rate
    loosen: float | None = None
    reward_key: str = "reward"
    reward_drop_tol: float = 0.02

    def __post_init__(self):
        if self.knob not in KNOBS:
            raise ValueError(f"knob must be one of {KNOBS}, got {self.knob!r}")
        if not self.low < self.high:
            raise ValueError(f"need low < high for hysteresis, got low={self.low}, high={self.high}")
        if self.knob == "max_staleness":
            defaults = {"min_value": 1, "max_value": 8, "tighten": -1, "loosen": 1}
        else:
            defaults = {"min_value": 1e-7, "max_value": 1e-5, "tighten": 0.7, "loosen": 1.2}
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)

    @property
    def multiplicative(self) -> bool:
        return self.knob == "learning_rate"


@dataclass
class Decision:
    step: int
    knob: str
    old_value: float
    new_value: float
    reason: str
    drift: float | None
    reward_ema: float | None

    @property
    def changed(self) -> bool:
        return self.new_value != self.old_value

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "changed": self.changed}


class Controller(Protocol):
    """Shared interface for the rule-based and (stretch) LLM controllers."""

    value: float

    def decide(self, history: list[dict[str, Any]]) -> tuple[float, str]: ...


class RuleController:
    def __init__(self, config: ControllerConfig, initial_value: float):
        self.config = config
        self.value = initial_value
        self.last_change_step = -math.inf
        self.history: list[dict[str, Any]] = []

    # -- pure decision ---------------------------------------------------------------------------------------------
    def decide(self, history: list[dict[str, Any]]) -> tuple[float, str]:
        """New knob value and the reason, from the metric rows seen so far (last row = current step)."""
        c = self.config
        step = history[-1]["step"]
        drift = ema_series([row.get(c.signal) for row in history], c.ema_alpha)[-1]
        if drift is None:
            return self.value, f"hold: no {c.signal} yet"
        if step - self.last_change_step < c.cooldown_steps:
            return self.value, f"hold: cooldown ({step - self.last_change_step:.0f} < {c.cooldown_steps} steps)"
        if drift > c.high:
            new = self._move(c.tighten)
            if new == self.value:
                return self.value, f"hold: drift {drift:.4g} > high {c.high:.4g} but already at min"
            return new, f"tighten: drift {drift:.4g} > high {c.high:.4g}"
        if drift < c.low:
            rewards = ema_series([row.get(c.reward_key) for row in history], c.ema_alpha)
            # Compare with the reward EMA one decision window ago (or the oldest row, on the first decision).
            now, before = rewards[-1], rewards[max(0, len(rewards) - 1 - c.every_k)]
            if now is not None and before is not None and now < before - c.reward_drop_tol:
                return (
                    self.value,
                    f"hold: drift {drift:.4g} < low {c.low:.4g} but reward falling ({before:.3f} -> {now:.3f})",
                )
            new = self._move(c.loosen)
            if new == self.value:
                return self.value, f"hold: drift {drift:.4g} < low {c.low:.4g} but already at max"
            return new, f"loosen: drift {drift:.4g} < low {c.low:.4g}"
        return self.value, f"hold: drift {drift:.4g} in [{c.low:.4g}, {c.high:.4g}]"

    def _move(self, delta: float) -> float:
        c = self.config
        new = self.value * delta if c.multiplicative else self.value + delta
        new = min(max(new, c.min_value), c.max_value)
        return int(round(new)) if c.knob == "max_staleness" else new

    # -- stateful driver -------------------------------------------------------------------------------------------
    def observe(self, row: dict[str, Any]) -> Decision | None:
        """Feed one step's metric row; returns a Decision on decision steps (every_k, after warmup), else None."""
        self.history.append(row)
        step = row["step"]
        c = self.config
        if step < c.warmup_steps or step % c.every_k != 0:
            return None
        new_value, reason = self.decide(self.history)
        drift = ema_series([r.get(c.signal) for r in self.history], c.ema_alpha)[-1]
        reward = ema_series([r.get(c.reward_key) for r in self.history], c.ema_alpha)[-1]
        decision = Decision(step, c.knob, self.value, new_value, reason, drift, reward)
        if decision.changed:
            self.value = new_value
            self.last_change_step = step
        return decision


# -- applying a knob to a live AsyncGRPOTrainer (TRL 1.18.2) -------------------------------------------------------
def get_knob(trainer, knob: str) -> float:
    if knob == "max_staleness":
        return trainer.args.max_staleness
    if knob == "learning_rate":
        return trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer is not None else trainer.args.learning_rate
    raise ValueError(knob)


def set_max_staleness(trainer, value: int) -> None:
    """Change ``max_staleness`` mid-run.

    Two places read it, both in the trainer process: ``RolloutQueueDataset.max_staleness`` (the drop filter, read
    per sample) and ``args.max_staleness`` (read every step by ``_log_step_metrics`` to size the generation
    lookahead it publishes to the rollout process through the shared ``token_allowance`` value). The rollout child
    never reads the config value itself, so nothing needs to cross the process boundary.
    """
    value = int(value)
    if value < 0:
        raise ValueError(f"max_staleness must be >= 0, got {value}")
    trainer.args.max_staleness = value
    dataset = getattr(trainer, "_rollout_dataset", None)
    if dataset is not None:
        dataset.max_staleness = value


def set_learning_rate(trainer, value: float) -> None:
    """Change the LR mid-run. The constant LambdaLR recomputes lr from base_lrs each step, so update both."""
    for group in trainer.optimizer.param_groups:
        group["lr"] = value
    scheduler = getattr(trainer, "lr_scheduler", None)
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [value] * len(scheduler.base_lrs)


def apply_knob(trainer, knob: str, value: float) -> None:
    {"max_staleness": set_max_staleness, "learning_rate": set_learning_rate}[knob](trainer, value)
