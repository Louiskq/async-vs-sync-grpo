import types

import numpy as np
import pytest

from src.controller import ControllerConfig, RuleController
from src.logging_utils import JSONLWriter, StepStats, UnifiedMetricsCallback, _parse_bus_id, read_jsonl

ASYNC_LOGS = {
    "reward": 0.5,
    "completions/mean_length": 200.0,
    "rollout/generated_tokens_per_s": 5000.0,
    "batch/trained_tokens_per_step": 60000.0,
    "perf/rollout_wait_s": 1.5,
    "rollout/backpressure_s": 0.2,
    "sample/staleness_mean": 1.2,
    "sample/staleness_max": 3.0,
    "batch/samples_per_step": 250.0,
    "learning_rate": 1e-6,
}


def make_cb(tmp_path, kind="async", controller=None):
    stats = StepStats()
    cb = UnifiedMetricsCallback(kind, str(tmp_path), stats, {"name": "t"}, 0.5, 5.0, controller=controller)
    trainer = types.SimpleNamespace(args=types.SimpleNamespace(max_staleness=4), _rollout_dataset=None)
    cb.bind(trainer)
    return cb, stats, trainer


def step(cb, n, logs):
    state = types.SimpleNamespace(global_step=n)
    cb.on_step_end(None, state, None)
    cb.on_log(None, state, None, logs=dict(logs))


def rows(tmp_path):
    return [r for r in read_jsonl(tmp_path / "metrics.jsonl") if "event" not in r]


def test_async_row_schema(tmp_path):
    cb, stats, trainer = make_cb(tmp_path)
    trainer._metrics = {"train": {"sample/staleness_mean": [0.0, 1.0, 1.0, 3.0]}}
    cb.on_train_begin(None, None, None)
    stats.logratio.add(np.array([0.0, 0.1, -0.1, 2.0]))
    step(cb, 1, ASYNC_LOGS)
    (r,) = rows(tmp_path)
    assert r["staleness_hist"] == {"0": 1, "1": 2, "3": 1}
    assert r["step"] == 1 and r["wall_s"] >= 0
    assert r["reward"] == 0.5 and r["staleness_max"] == 3.0 and r["max_staleness"] == 4
    assert r["logratio_abs_max"] == 2.0 and r["is_outside_frac"] == 0.25
    assert r["trl"]["reward"] == 0.5
    events = [e["event"] for e in read_jsonl(tmp_path / "metrics.jsonl") if "event" in e]
    assert events == ["start"]


def test_sync_row_uses_trainer_timers(tmp_path):
    cb, stats, _ = make_cb(tmp_path, kind="sync")
    cb.on_train_begin(None, None, None)
    stats.gen_s, stats.completion_tokens, stats.samples = 2.0, 1000.0, 256
    step(cb, 1, {"reward": 0.4, "completions/mean_length": 150.0})
    (r,) = rows(tmp_path)
    assert r["gen_tokens_per_s"] == 500.0 and r["samples_per_step"] == 256 and r["trainer_wait_s"] == 2.0
    assert r["staleness_max"] == 0.0
    assert stats.gen_s == 0.0  # popped


def test_checkpoint_time_is_excluded_from_wall_clock(tmp_path, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("src.logging_utils.time.time", lambda: clock["t"])
    cb, _, _ = make_cb(tmp_path)
    cb.on_train_begin(None, None, None)
    clock["t"] += 10
    step(cb, 1, ASYNC_LOGS)
    clock["t"] += 30  # checkpoint write
    cb.on_save(None, None, None)
    clock["t"] += 10
    step(cb, 2, ASYNC_LOGS)
    r1, r2 = rows(tmp_path)
    assert r1["wall_s"] == 10 and r2["wall_s"] == 20 and r2["step_s"] == 10


def test_controller_decision_is_applied_and_logged(tmp_path):
    ctrl = RuleController(
        ControllerConfig(high=0.5, low=0.1, every_k=2, warmup_steps=2, cooldown_steps=2, ema_alpha=1.0), 4
    )
    cb, stats, trainer = make_cb(tmp_path, controller=ctrl)
    cb.on_train_begin(None, None, None)
    for n in (1, 2):
        stats.logratio.add(np.full(10, 1.0))  # p95 |log-ratio| = 1.0 > high
        step(cb, n, ASYNC_LOGS)
    assert trainer.args.max_staleness == 3
    r1, r2 = rows(tmp_path)
    assert r2["max_staleness"] == 4  # value in effect during the step; the change applies from the next step
    assert r2["decision"]["new_value"] == 3 and r2["decision"]["changed"]
    decisions = [e for e in read_jsonl(tmp_path / "metrics.jsonl") if e.get("event") == "decision"]
    assert len(decisions) == 1 and decisions[0]["reason"].startswith("tighten")


def test_summary_log_is_an_event(tmp_path):
    cb, _, _ = make_cb(tmp_path)
    cb.on_train_begin(None, None, None)
    cb.on_log(None, types.SimpleNamespace(global_step=5), None, logs={"train_runtime": 12.0})
    assert rows(tmp_path) == []


def test_jsonl_writer_makes_nan_null(tmp_path):
    w = JSONLWriter(tmp_path / "x.jsonl")
    w.write({"a": float("nan"), "b": np.float32(1.5), "c": [float("inf")]})
    assert read_jsonl(tmp_path / "x.jsonl") == [{"a": None, "b": 1.5, "c": [None]}]


def test_parse_bus_id():
    assert _parse_bus_id("00000009:01:00.0") == (9, 1, 0)
    assert _parse_bus_id("0000:1B:00.0") == (0, 27, 0)
    with pytest.raises(ValueError):
        _parse_bus_id("garbage")


def test_gpu_sampler_roles_from_launcher_bus_ids(monkeypatch):
    """Roles come from the bus ids slurm/_run.sh exports, independent of torch device properties."""
    import src.logging_utils as lu

    class Pci:
        def __init__(self, bus):
            self.busId = bus

    fake = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 4,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetPciInfo=lambda h: Pci(f"00000009:0{h}:00.0".encode()),
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
    monkeypatch.setattr(lu, "trainer_bus_id", lambda: None)
    monkeypatch.setenv("AGRPO_VLLM_BUS_ID", "0009:02:00.0")
    monkeypatch.setenv("AGRPO_TRAINER_BUS_ID", "00000009:03:00.0")
    s = lu.GPUSampler()
    assert s.roles == {0: "gpu0", 1: "gpu1", 2: "vllm", 3: "trainer"}
