"""Controller-signal choice from pilot runs: by async-vs-sync separation relative to noise, not by smoothness."""

import json

import numpy as np

from src.logging_utils import JSONLWriter


def pilot(root, name, p95, clip, seed):
    rng = np.random.default_rng(seed)
    w = JSONLWriter(root / f"{name}-seed0" / "metrics.jsonl")
    w.write({"event": "start"})
    for step in range(1, 31):
        w.write(
            {
                "step": step,
                "wall_s": float(step),
                # smooth but barely different between async and sync (the 0.6B pilot's log-ratio p95)
                "logratio_abs_p95": p95 + rng.normal(0, 0.003),
                # noisier in relative terms but clearly separated (the 0.6B pilot's clip fraction)
                "clip_frac": max(0.0, clip + rng.normal(0, 0.0004)) if clip else 0.0,
            }
        )


def test_signal_chosen_by_separation_over_noise(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    pilot(runs, "pilot-async-s4", p95=0.0760, clip=0.0017, seed=0)  # "async" contains "sync": must not match
    pilot(runs, "pilot-sync", p95=0.0734, clip=0.0, seed=1)
    out = tmp_path / "t.json"
    monkeypatch.setattr("sys.argv", ["thresholds.py", "--runs", str(runs), "--out", str(out)])
    from analysis.thresholds import main

    main()
    s = json.loads(out.read_text())["suggestion"]
    assert s["signal"] == "clip_frac"
    assert s["low"] < s["high"]
    assert 0.0 < s["low"] < 0.0017 < s["high"] + 0.0004
