"""Loading logged runs for analysis: runs/<config>-seed<k>/{metrics.jsonl, eval.jsonl, job.json}."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.logging_utils import read_jsonl


@dataclass
class Run:
    path: Path
    config: str
    seed: int
    rows: list[dict]
    events: list[dict]
    evals: list[dict] = field(default_factory=list)
    job: dict = field(default_factory=dict)

    def series(self, key: str, x: str = "step") -> tuple[np.ndarray, np.ndarray]:
        pts = [(r[x], r[key]) for r in self.rows if r.get(key) is not None and r.get(x) is not None]
        if not pts:
            return np.array([]), np.array([])
        xs, ys = zip(*pts)
        return np.asarray(xs, float), np.asarray(ys, float)

    def mean(self, key: str, skip_first: int = 3) -> float | None:
        """Mean over steps, skipping the first few (warm-up: first rollouts, CUDA graph capture, allocator)."""
        vals = [r[key] for r in self.rows[skip_first:] if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def start(self) -> dict:
        return next((e for e in self.events if e.get("event") == "start"), {})

    @property
    def decisions(self) -> list[dict]:
        return [e for e in self.events if e.get("event") == "decision"]

    @property
    def final_wall_s(self) -> float | None:
        return self.rows[-1]["wall_s"] if self.rows else None


def load_runs(runs_dir: str | Path, include: str | None = None) -> list[Run]:
    runs = []
    for d in sorted(Path(runs_dir).iterdir()):
        m = re.fullmatch(r"(.+)-seed(\d+)", d.name)
        metrics = d / "metrics.jsonl"
        if not m or not metrics.exists() or (include and not re.search(include, d.name)):
            continue
        records = read_jsonl(metrics)
        rows = sorted((r for r in records if "event" not in r), key=lambda r: r["step"])
        events = [r for r in records if "event" in r]
        evals = sorted(read_jsonl(d / "eval.jsonl"), key=lambda r: r["step"]) if (d / "eval.jsonl").exists() else []
        job = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
        runs.append(Run(d, m.group(1), int(m.group(2)), rows, events, evals, job))
    return runs


def by_config(runs: list[Run]) -> dict[str, list[Run]]:
    out: dict[str, list[Run]] = {}
    for r in runs:
        out.setdefault(r.config, []).append(r)
    return out


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    acc = None
    for i, v in enumerate(y):
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def first_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """First x at which y >= target (no interpolation: the first logged point that reaches it)."""
    hits = np.nonzero(y >= target)[0]
    return float(x[hits[0]]) if hits.size else None


def gpu_hours(run: Run) -> float | None:
    """From job.json written by slurm/_run.sh (whole job: startup + train + eval); None if absent."""
    j = run.job
    if {"start", "end", "n_gpus"} <= j.keys():
        return (j["end"] - j["start"]) / 3600 * j["n_gpus"]
    return None
