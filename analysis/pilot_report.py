"""M2 pilot report: the numbers the SPEC asks for before the sweep, plus a GPU-hour projection.

    python analysis/pilot_report.py --runs $RUNS_DIR [--include '^pilot-'] [--steps 300] [--seeds 2]

Per pilot run (skipping the first ``--skip`` steps as warm-up): s/step, where the step time goes (sync: generation
vs the rest; async: trainer waiting for data vs generator held back), GPU utilisation, rollouts and prompts per
optimizer step, the off-policy log-ratio distribution, and the fraction of zero-variance groups. Then the balance
check (is generation comparable to training?) and projected GPU-hours for the core sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.runs import Run, load_runs  # noqa: E402


def m(run: Run, key: str, skip: int) -> float | None:
    v = [r[key] for r in run.rows[skip:] if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def trl(run: Run, key: str, skip: int) -> float | None:
    v = [r["trl"][key] for r in run.rows[skip:] if r.get("trl", {}).get(key) is not None]
    return float(np.mean(v)) if v else None


def fmt(x, spec=".3g"):
    return "—" if x is None else format(x, spec)


def report(run: Run, skip: int) -> dict:
    is_async = run.rows and run.rows[0].get("staleness_hist") is not None
    zero_std = trl(run, "frac_zero_std_groups", skip) if is_async else trl(run, "frac_reward_zero_std", skip)
    out = {
        "run": run.path.name,
        "kind": "async" if is_async else "sync",
        "steps": len(run.rows),
        "startup_s": run.start.get("startup_s"),
        "s_per_step": m(run, "step_s", skip),
        "trainer_wait_s": m(run, "trainer_wait_s", skip),
        "gen_idle_s": m(run, "gen_idle_s", skip),
        "fwd_bwd_s": trl(run, "perf/fwd_bwd_s", skip),
        "gpu_util_vllm": m(run, "gpu_util_vllm", skip),
        "gpu_util_trainer": m(run, "gpu_util_trainer", skip),
        "gen_tokens_per_s": m(run, "gen_tokens_per_s", skip),
        "train_tokens_per_s": m(run, "train_tokens_per_s", skip),
        "rollouts_per_step": m(run, "samples_per_step", skip),
        "completion_len": m(run, "completion_len", skip),
        "reward": m(run, "reward", skip),
        "zero_std_group_frac": zero_std,
        "staleness_mean": m(run, "staleness_mean", skip),
        "dropped_stale_per_step": m(run, "dropped_stale", skip),
        "logratio_abs_p95": m(run, "logratio_abs_p95", skip),
        "logratio_std": m(run, "logratio_std", skip),
        "is_outside_frac": m(run, "is_outside_frac", skip),
        "clip_frac": m(run, "clip_frac", skip),
    }
    if is_async:
        groups = m(run, "groups_per_step", skip)
        out["groups_trained_per_step"] = groups
        # Groups with all-equal rewards are dropped before training, so prompts generated per step is higher.
        out["prompts_generated_per_step"] = groups / (1 - zero_std) if groups and zero_std is not None else None
    else:
        out["groups_trained_per_step"] = (out["rollouts_per_step"] or 0) / 8 or None
        out["prompts_generated_per_step"] = out["groups_trained_per_step"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--include", default="^pilot-")
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--eval_min", type=float, default=12.0, help="post-hoc eval minutes per run (7 checkpoints)")
    a = p.parse_args()
    runs = load_runs(a.runs, a.include)
    if not runs:
        raise SystemExit(f"no runs matching {a.include!r} under {a.runs}")
    reports = [report(r, a.skip) for r in runs]
    keys = [k for k in reports[0] if k != "run"]
    print("| metric | " + " | ".join(r["run"] for r in reports) + " |")
    print("|---|" + "---|" * len(reports))
    for k in keys:
        print(
            f"| {k} | " + " | ".join(fmt(r.get(k)) if not isinstance(r.get(k), str) else r[k] for r in reports) + " |"
        )

    print("\nBalance (async only helps if generation and training take comparable time):")
    for r in reports:
        if r["kind"] == "sync" and r["s_per_step"] and r["trainer_wait_s"] is not None:
            gen = r["trainer_wait_s"]
            print(
                f"  {r['run']}: generation {gen:.1f}s vs rest of step {r['s_per_step'] - gen:.1f}s per step "
                f"(generation share {gen / r['s_per_step']:.0%})"
            )
        if r["kind"] == "async" and r["s_per_step"]:
            print(
                f"  {r['run']}: trainer waits {fmt(r['trainer_wait_s'])}s/step for data, generator held back "
                f"{fmt(r['gen_idle_s'])}s/step; fwd+bwd {fmt(r['fwd_bwd_s'])}s of {r['s_per_step']:.1f}s"
            )

    print(f"\nProjection: {a.steps} steps x {a.seeds} seeds, 2 GPUs, + startup + {a.eval_min:.0f} min eval per run:")
    total = 0.0
    for r in reports:
        if not r["s_per_step"]:
            continue
        hours = (a.steps * r["s_per_step"] + (r["startup_s"] or 180) + a.eval_min * 60) / 3600
        gpu_h = 2 * hours * a.seeds
        total += gpu_h
        print(f"  like {r['run']}: {hours:.2f} h/run -> {gpu_h:.1f} GPU-h for {a.seeds} seeds")
    print(
        f"  per-config estimates above; core sweep = sync + 2 async configs (+ controller in M4): "
        f"~{total / len(reports) * 4:.1f} GPU-h for 4 configs at the mean rate"
    )


if __name__ == "__main__":
    main()
