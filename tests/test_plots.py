"""End-to-end: synthetic logs in the real on-disk format -> every figure and the summary table."""

import json

import numpy as np

from src.logging_utils import JSONLWriter


def fake_run(root, config, seed, kind, s_per_step, staleness=0, ctrl=None):
    d = root / f"{config}-seed{seed}"
    w = JSONLWriter(d / "metrics.jsonl")
    w.write({"event": "start", "controller": ctrl})
    rng = np.random.default_rng(seed)
    for step in range(1, 61):
        row = {
            "step": step,
            "wall_s": step * s_per_step,
            "step_s": s_per_step,
            "reward": min(0.9, 0.3 + step / 100) + rng.normal(0, 0.02),
            "gen_tokens_per_s": 4000.0,
            "gpu_util_vllm": 60.0,
            "gpu_util_trainer": 70.0,
            "logratio_abs_p95": 0.05 + 0.02 * staleness,
            "clip_frac": 0.01 * staleness,
            "max_staleness": staleness,
        }
        if kind == "async":
            row["staleness_hist"] = {str(k): 10 for k in range(staleness + 1)}
        w.write(row)
    if ctrl:
        w.write({"event": "decision", "step": 20, "old_value": 4, "new_value": 3, "changed": True})
    e = JSONLWriter(d / "eval.jsonl")
    for step in (0, 30, 60):
        e.write({"step": step, "wall_s": step * s_per_step, "accuracy": 0.5 + step / 300})
    (d / "job.json").write_text(json.dumps({"start": 0, "end": 3600, "n_gpus": 2}))


def test_plots_end_to_end(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        fake_run(runs, "sync", seed, "sync", 20.0)
        fake_run(runs, "async-s1", seed, "async", 14.0, staleness=1)
        fake_run(runs, "async-s4", seed, "async", 11.0, staleness=4)
        fake_run(
            runs,
            "async-ctrl",
            seed,
            "async",
            12.0,
            staleness=3,
            ctrl={"high": 0.1, "low": 0.05, "ema_alpha": 0.2, "signal": "logratio_abs_p95"},
        )
    (runs / "_base_eval").mkdir()  # must be ignored
    out = tmp_path / "results"
    monkeypatch.setattr("sys.argv", ["plots.py", "--runs", str(runs), "--out", str(out)])
    from analysis.plots import main

    main()
    for name in (
        "reward_vs_wallclock",
        "reward_vs_step",
        "eval_vs_wallclock",
        "throughput",
        "staleness",
        "offpolicy",
        "controller",
    ):
        assert (out / f"{name}.png").stat().st_size > 10_000, name
    summary = (out / "summary.md").read_text()
    assert "| async-s4 | 2 |" in summary
    assert "Total logged compute: 16.00 GPU-hours" in summary
    # sync final acc 0.7 -> target 0.63; async-s4 reaches 0.6 at step 30 and 0.7 at step 60 = 660 s = 11 min
    assert "0.630" in summary
    assert "| async-s4 | 2 | 0.700 [0.700, 0.700] | 11.0 [11.0, 11.0] |" in summary
