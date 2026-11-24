"""Regression: 'async' contains 'sync', so run matching must be exact."""

import json

from tests.test_plots import fake_run


def test_thresholds_pick_sync_and_async_pilots_correctly(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    fake_run(runs, "pilot-async-s4", 0, "async", 11.0, staleness=4)  # listed first alphabetically
    fake_run(runs, "pilot-sync", 0, "sync", 20.0)
    out = tmp_path / "t.json"
    monkeypatch.setattr("sys.argv", ["thresholds.py", "--runs", str(runs), "--out", str(out)])
    from analysis.thresholds import main

    main()
    s = json.loads(out.read_text())["suggestion"]
    # fake signal: sync 0.05, async-s4 0.13 -> high = 0.13, low = midpoint 0.09
    assert s["signal"] == "logratio_abs_p95"  # constant fake signals: zero noise, first candidate wins the tie
    assert abs(s["high"] - 0.13) < 1e-9
    assert abs(s["low"] - 0.09) < 1e-9
