"""Unified per-step metrics -> JSONL, for both the sync and async trainers.

One row per optimizer step (``logging_steps=1``) in ``<output_dir>/metrics.jsonl``. Each row has the SPEC's
unified fields at the top level (same names for both trainers, ``None`` where a field does not apply) and the raw
TRL logs under ``"trl"``. Events (start, controller decisions, end) go to the same file with an ``"event"`` key.

Wall-clock (``wall_s``) starts at ``on_train_begin``, i.e. after model load, vLLM server readiness and (async) the
cold weight sync, and excludes time spent writing checkpoints. Startup time is recorded once in the start event.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from src.signals import LogRatioAccumulator


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy/torch scalars -> python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            x = x.item()
        except (ValueError, TypeError):
            return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class JSONLWriter:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)  # line-buffered: rows survive a killed job
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(_clean(row), allow_nan=False)
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        self._f.close()


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# -- GPU utilisation -----------------------------------------------------------------------------------------------
def _parse_bus_id(bus_id: str) -> tuple[int, int, int]:
    """NVML/nvidia-smi bus id "00000009:01:00.0" -> (domain, bus, device); domain width differs between tools."""
    domain, bus, rest = bus_id.strip().lower().split(":")
    return int(domain, 16), int(bus, 16), int(rest.split(".")[0], 16)


def trainer_bus_id() -> tuple[int, int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        return p.pci_domain_id, p.pci_bus_id, p.pci_device_id  # absent on some torch builds -> None below
    except Exception:
        return None


class GPUSampler:
    """Samples NVML utilisation and memory for every visible GPU in a daemon thread (default every 1 s).

    Devices are labelled by role: the one the trainer process runs on is ``trainer``; with exactly two devices
    the other is ``vllm``; otherwise ``gpu<i>``. ``pop()`` returns the means since the previous ``pop()``.
    Disabled (``pop()`` returns ``{}``) when NVML is unavailable, e.g. on a laptop.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.roles: dict[int, str] = {}
        self._handles = []
        self._sums: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        except Exception:
            self._nvml = None
            return
        # Roles from the bus ids slurm/_run.sh exports; else detect the trainer's own device through torch.
        env = {k: os.environ.get(f"AGRPO_{k.upper()}_BUS_ID") for k in ("vllm", "trainer")}
        known = {_parse_bus_id(v): k for k, v in env.items() if v}
        mine = trainer_bus_id() if "trainer" not in known.values() else None
        for i, h in enumerate(self._handles):
            bus = pynvml.nvmlDeviceGetPciInfo(h).busId
            bus = _parse_bus_id(bus.decode() if isinstance(bus, bytes) else bus)
            if bus in known:
                self.roles[i] = known[bus]
            elif mine is not None and bus == mine:
                self.roles[i] = "trainer"
        if len(self._handles) == 2 and len(self.roles) == 1:
            other = 1 - next(iter(self.roles))
            self.roles[other] = "vllm"
        for i in range(len(self._handles)):
            self.roles.setdefault(i, f"gpu{i}")

    @property
    def enabled(self) -> bool:
        return self._nvml is not None and bool(self._handles)

    def start(self) -> None:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            sample = {}
            for i, h in enumerate(self._handles):
                role = self.roles[i]
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(h)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(h)
                except Exception:
                    continue
                sample[f"gpu_util_{role}"] = float(util.gpu)
                sample[f"gpu_mem_gb_{role}"] = mem.used / 1e9
            with self._lock:
                for k, v in sample.items():
                    s = self._sums.setdefault(k, [0.0, 0.0])
                    s[0] += v
                    s[1] += 1

    def pop(self) -> dict[str, float]:
        with self._lock:
            out = {k: s / n for k, (s, n) in self._sums.items() if n}
            self._sums = {}
        return out

    def stop(self) -> None:
        self._stop.set()


# -- the callback --------------------------------------------------------------------------------------------------
class StepStats:
    """Per-step counters the trainer subclasses fill in (see train_sync.py / train_async.py)."""

    def __init__(self):
        self.logratio = LogRatioAccumulator()
        self.gen_s = 0.0  # sync: wall time inside vLLM generation (trainer idle)
        self.completion_tokens = 0.0  # sync: completion tokens generated (= trained) this step
        self.samples = 0  # sync: completions trained this step

    def pop_sync(self) -> dict[str, float]:
        out = {"gen_s": self.gen_s, "completion_tokens": self.completion_tokens, "samples": self.samples}
        self.gen_s, self.completion_tokens, self.samples = 0.0, 0.0, 0
        return out


def _get(logs: dict, key: str):
    v = logs.get(key)
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


class UnifiedMetricsCallback(TrainerCallback):
    def __init__(
        self,
        kind: str,
        output_dir: str,
        stats: StepStats,
        run_info: dict[str, Any],
        is_clip_min: float,
        is_clip_max: float,
        controller=None,
        launch_time: float | None = None,
    ):
        if kind not in ("sync", "async"):
            raise ValueError(kind)
        self.kind = kind
        self.writer = JSONLWriter(Path(output_dir) / "metrics.jsonl")
        self.stats = stats
        self.run_info = run_info
        self.is_clip = {"is_clip_min": is_clip_min, "is_clip_max": is_clip_max}
        self.controller = controller
        self.launch_time = launch_time
        self.trainer = None
        self.gpu = GPUSampler()
        self.t0: float | None = None
        self.excluded_s = 0.0
        self._last_wall = 0.0
        self._last_log_time: float | None = None
        self._offpolicy: dict[str, float] = {}
        self._pending_hist: dict[str, int] | None = None

    def bind(self, trainer) -> UnifiedMetricsCallback:
        self.trainer = trainer
        return self

    def wall_s(self, now: float | None = None) -> float:
        return (now or time.time()) - self.t0 - self.excluded_s

    # Registered after TRL's own callbacks, so on_train_begin runs after the async cold weight sync / worker start.
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.gpu.start()
        startup_s = self.t0 - self.launch_time if self.launch_time else None
        self.writer.write(
            {"event": "start", "time": self.t0, "startup_s": startup_s, "gpu_roles": self.gpu.roles, **self.run_info}
        )

    def on_step_end(self, args, state, control, **kwargs):
        self._offpolicy = self.stats.logratio.pop_stats(**self.is_clip)
        self._pending_hist = None
        metrics = getattr(self.trainer, "_metrics", None)
        if self.kind == "async" and metrics is not None:
            values = metrics["train"].get("sample/staleness_mean") or []
            hist: dict[str, int] = {}
            for v in values:
                hist[str(int(v))] = hist.get(str(int(v)), 0) + 1
            self._pending_hist = hist

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = dict(logs or {})
        if self.t0 is None:
            return
        now = time.time()
        if "train_runtime" in logs:
            self.writer.write({"event": "summary", "step": state.global_step, "wall_s": self.wall_s(now), "trl": logs})
            return
        row = self._row(state.global_step, logs, now)
        if self.controller is not None:
            decision = self.controller.observe(row)
            if decision is not None:
                from src.controller import apply_knob

                if decision.changed:
                    apply_knob(self.trainer, decision.knob, decision.new_value)
                row["decision"] = decision.to_dict()
                self.writer.write({"event": "decision", **decision.to_dict(), "wall_s": row["wall_s"]})
        self.writer.write(row)
        self._last_log_time = time.time()

    def on_save(self, args, state, control, **kwargs):
        # HF order is step_end -> log -> save, so the gap since the last log is (almost all) checkpoint writing.
        if self._last_log_time is not None:
            self.excluded_s += time.time() - self._last_log_time
            self._last_log_time = None

    def on_train_end(self, args, state, control, **kwargs):
        self.gpu.stop()
        if self.t0 is not None:
            self.writer.write({"event": "end", "step": state.global_step, "wall_s": self.wall_s()})

    def _staleness_hist(self) -> dict[str, int] | None:
        """Per-sample staleness counts for this step (``{"0": n0, "1": n1, ...}``).

        TRL appends one value per consumed sample to ``_metrics["train"]["sample/staleness_mean"]`` and reduces the
        list to a mean inside ``log()``, which runs after ``on_step_end`` and before ``on_log``. So the raw list is
        read in ``on_step_end`` and only attached to the row here.
        """
        return self._pending_hist

    # -- row assembly ------------------------------------------------------------------------------------------------
    def _row(self, step: int, logs: dict, now: float) -> dict[str, Any]:
        wall = self.wall_s(now)
        step_s = wall - self._last_wall
        self._last_wall = wall
        row: dict[str, Any] = {
            "step": step,
            "wall_s": wall,
            "step_s": step_s,
            "reward": _get(logs, "reward"),
            "completion_len": _get(logs, "completions/mean_length"),
            "clip_frac": _get(logs, "clip_ratio/region_mean"),
            "learning_rate": _get(logs, "learning_rate"),
            "loss": _get(logs, "loss"),
        }
        if self.kind == "async":
            trained = _get(logs, "batch/trained_tokens_per_step")
            row.update(
                reward_trained=_get(logs, "reward_trained"),
                gen_tokens_per_s=_get(logs, "rollout/generated_tokens_per_s"),
                train_tokens=trained,
                train_tokens_per_s=trained / step_s if trained and step_s > 0 else None,
                trainer_wait_s=_get(logs, "perf/rollout_wait_s"),
                gen_idle_s=_get(logs, "rollout/backpressure_s"),
                weight_sync_s=_get(logs, "perf/weight_sync_s"),
                staleness_mean=_get(logs, "sample/staleness_mean"),
                staleness_max=_get(logs, "sample/staleness_max"),
                dropped_stale=_get(logs, "sample/dropped_stale_count") or 0.0,
                samples_per_step=_get(logs, "batch/samples_per_step"),
                groups_per_step=_get(logs, "batch/groups_per_step"),
                max_staleness=self.trainer.args.max_staleness if self.trainer is not None else None,
                staleness_hist=self._staleness_hist(),
            )
        else:
            s = self.stats.pop_sync()
            gen_s = s["gen_s"]
            row.update(
                gen_tokens_per_s=s["completion_tokens"] / gen_s if gen_s > 0 else None,
                train_tokens=s["completion_tokens"] or None,
                train_tokens_per_s=s["completion_tokens"] / step_s if step_s > 0 else None,
                trainer_wait_s=gen_s if s["samples"] else None,  # 0 on steps that reuse a generation batch
                gen_idle_s=max(step_s - gen_s, 0.0),
                staleness_mean=0.0,
                staleness_max=0.0,
                dropped_stale=0.0,
                samples_per_step=s["samples"] or None,
                max_staleness=0,
            )
        row.update(self._offpolicy)
        row.update(self.gpu.pop())
        row["trl"] = logs
        return row
