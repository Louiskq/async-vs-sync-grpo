"""Controller thresholds from PILOT runs only (SPEC M4: no tuning on the final runs).

    python analysis/thresholds.py --runs runs --include '^pilot-'

For each candidate signal it prints, per pilot run, the distribution of its EMA (as the controller sees it) and
its step-to-step noise (std of first differences of the raw signal, relative to its median). It then suggests:

* signal = the candidate with the lowest relative noise on the async pilot,
* high   = p75 of that signal's EMA on the async (max_staleness=4) pilot: tighten when drift is in the top
           quarter of what s=4 produces,
* low    = midpoint of the sync median (staleness 0: pure trainer/vLLM mismatch) and the async median: loosen
           only when drift is close to the on-policy floor.

The rule is a starting point; the chosen values and the reason go in DECISIONS/notes, then into async-ctrl.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.runs import ema, load_runs  # noqa: E402

CANDIDATES = ("logratio_abs_p95", "clip_frac", "is_outside_frac", "logratio_std")


def describe(run, key: str, alpha: float, skip: int = 3) -> dict | None:
    _, y = run.series(key)
    y = y[skip:]
    if y.size < 5:
        return None
    e = ema(y, alpha)
    med = float(np.median(y))
    return {
        "n": int(y.size),
        "median": med,
        "ema_p25": float(np.quantile(e, 0.25)),
        "ema_p50": float(np.quantile(e, 0.50)),
        "ema_p75": float(np.quantile(e, 0.75)),
        "rel_noise": float(np.std(np.diff(y)) / med) if med > 0 else float("inf"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--include", default="^pilot-")
    p.add_argument("--alpha", type=float, default=0.2, help="controller EMA alpha")
    p.add_argument("--out", default="results/pilot_thresholds.json")
    a = p.parse_args()
    runs = load_runs(a.runs, a.include)
    if not runs:
        raise SystemExit(f"no runs matching {a.include!r} under {a.runs}")
    report = {r.path.name: {k: describe(r, k, a.alpha) for k in CANDIDATES} for r in runs}
    for name, stats in report.items():
        print(name)
        for k, s in stats.items():
            if s:
                print(
                    f"  {k:18s} median {s['median']:.4g}  EMA p25/50/75 {s['ema_p25']:.4g}/{s['ema_p50']:.4g}/"
                    f"{s['ema_p75']:.4g}  rel.noise {s['rel_noise']:.3f}"
                )

    # Note "async" contains "sync": match the config name exactly, not by substring.
    sync = next((r for r in runs if r.config.removeprefix("pilot-") == "sync"), None)
    asy = next((r for r in runs if r.config.removeprefix("pilot-").startswith("async")), None)
    suggestion = None
    if sync and asy:
        noise = {k: s["rel_noise"] for k, s in report[asy.path.name].items() if s}
        signal = min(noise, key=noise.get)
        s_async, s_sync = report[asy.path.name][signal], report[sync.path.name].get(signal)
        if s_sync:
            suggestion = {
                "signal": signal,
                "high": s_async["ema_p75"],
                "low": (s_sync["median"] + s_async["median"]) / 2,
                "ema_alpha": a.alpha,
                "rule": "high = async pilot EMA p75; low = midpoint of sync and async medians; signal = least noisy",
            }
            print("\nsuggested (review before use):", json.dumps(suggestion, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"report": report, "suggestion": suggestion}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
