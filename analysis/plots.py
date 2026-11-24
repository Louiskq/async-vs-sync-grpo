"""All figures and the results table, regenerated from logs.

    python analysis/plots.py [--runs runs] [--out results] [--include REGEX]

Figures (PNG, light mode, for the README):
  reward_vs_wallclock, reward_vs_step, eval_vs_wallclock, throughput, staleness, offpolicy, controller
Table: results/summary.md and results/summary.csv (final eval accuracy and time-to-target per config,
mean [min, max] across seeds). Target = 90% of the sync config's mean final eval accuracy (``--target_frac``).
"""

from __future__ import annotations

import argparse
import csv
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.runs import Run, by_config, ema, first_crossing, gpu_hours, load_runs  # noqa: E402

# Reference categorical palette (dataviz skill), assigned to configs in a FIXED order so a config keeps its colour
# in every figure, whichever subset of configs is present.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONFIG_ORDER = ["sync", "async-s1", "async-s4", "async-ctrl", "async-s8", "async-llmctrl"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SMOOTH = 0.1  # EMA alpha for display smoothing of per-step curves


def color(config: str) -> str:
    base = config.removeprefix("pilot-")
    if base in CONFIG_ORDER:
        return PALETTE[CONFIG_ORDER.index(base)]
    return PALETTE[zlib.crc32(base.encode()) % len(PALETTE)]  # stable across processes, unlike hash()


def order(configs) -> list[str]:
    def key(c):
        b = c.removeprefix("pilot-")
        return (CONFIG_ORDER.index(b) if b in CONFIG_ORDER else 99, c)

    return sorted(configs, key=key)


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def band(ax, runs: list[Run], key: str, x: str, label: str, smooth: float | None = SMOOTH, xscale: float = 1.0):
    """Mean across seeds as a line, min-max across seeds as a band, on a common x grid; direct label at the end."""
    curves = []
    for r in runs:
        xs, ys = r.series(key, x)
        if xs.size:
            curves.append((xs * xscale, ema(ys, smooth) if smooth else ys))
    if not curves:
        return
    hi = min(c[0][-1] for c in curves)
    grid = np.linspace(max(c[0][0] for c in curves), hi, 200)
    ys = np.array([np.interp(grid, cx, cy) for cx, cy in curves])
    c = color(runs[0].config)
    if len(curves) > 1:
        ax.fill_between(grid, ys.min(0), ys.max(0), color=c, alpha=0.18, linewidth=0)
    ax.plot(grid, ys.mean(0), color=c, label=f"{label} (n={len(curves)})")
    ax._end_labels = getattr(ax, "_end_labels", []) + [(grid[-1], ys.mean(0)[-1], label)]


def place_end_labels(ax, min_gap_pt: float = 11.0):
    """Direct labels at line ends, spread vertically so they never overlap (in display space)."""
    items = getattr(ax, "_end_labels", [])
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    gap = min_gap_pt * fig.dpi / 72
    top = ax.transAxes.transform((0, 1))[1] - gap / 2
    width = 7.0 * fig.dpi / 72  # px per character, roughly, at 9pt

    def clashes(px, py, label, placed):
        w = width * len(label)
        return [q for q in placed if abs(q[1] - py) < gap and px < q[0] + width * len(q[2]) and q[0] < px + w]

    placed = []
    for (px, py), label in sorted(((to_px((x, y)), label) for x, y, label in items), key=lambda t: t[0][1]):
        y = py
        while (hit := clashes(px, y, label, placed)) and y <= top:
            # Step above the clashing labels; +0.5 px so float rounding can't re-clash (that loops forever).
            y = max(q[1] for q in hit) + gap + 0.5
        if y > top:  # no room above: go below the original position instead
            y = py
            while hit := clashes(px, y, label, placed):
                y = min(q[1] for q in hit) - gap - 0.5
        placed.append((px, y, label))
    inv = ax.transData.inverted().transform
    for px, py, label in placed:
        x, y = inv((px, py))
        ax.annotate(
            label, (x, y), xytext=(4, 0), textcoords="offset points", color=INK_2, va="center", annotation_clip=False
        )
    ax._end_labels = []


def fmt_value(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"wrote {out / name}.png")


def fig_reward(groups, out, x: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg in order(groups):
        runs = groups[cfg]
        band(ax, runs, "reward", x, cfg, xscale=1 / 60 if x == "wall_s" else 1.0)
    ax.set_xlabel("wall-clock since first training step (min)" if x == "wall_s" else "optimizer step")
    ax.set_ylabel(f"train reward (EMA α={SMOOTH})")
    ax.set_title("Train reward vs " + ("wall-clock" if x == "wall_s" else "step") + " (band = min–max over seeds)")
    ax.legend(loc="lower right")
    place_end_labels(ax)
    save(fig, out, "reward_vs_wallclock" if x == "wall_s" else "reward_vs_step")


def fig_eval(groups, out, target: float | None):
    fig, ax = plt.subplots(figsize=(7, 4))
    any_eval = False
    for cfg in order(groups):
        runs = [r for r in groups[cfg] if r.evals]
        if not runs:
            continue
        any_eval = True
        steps = sorted(set.intersection(*[{e["step"] for e in r.evals if e.get("wall_s") is not None} for r in runs]))
        if not steps:
            continue
        acc = np.array([[next(e["accuracy"] for e in r.evals if e["step"] == s) for s in steps] for r in runs])
        wall = np.array([[next(e["wall_s"] for e in r.evals if e["step"] == s) for s in steps] for r in runs]) / 60
        c = color(cfg)
        ax.plot(wall.mean(0), acc.mean(0), color=c, marker="o", markersize=5, label=f"{cfg} (n={len(runs)})")
        if len(runs) > 1:
            ax.vlines(wall.mean(0), acc.min(0), acc.max(0), color=c, linewidth=2, alpha=0.4)
    if not any_eval:
        plt.close(fig)
        return
    if target is not None:
        ax.axhline(target, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(
            "target: 90% of sync's gain",
            (ax.get_xlim()[0], target),
            xytext=(2, 3),
            textcoords="offset points",
            color=INK_2,
        )
    ax.set_xlabel("wall-clock since first training step (min)")
    ax.set_ylabel("held-out GSM8K accuracy (greedy, n=500)")
    ax.set_title("Eval accuracy vs wall-clock (bars = min–max over seeds)")
    ax.legend(loc="lower right")
    save(fig, out, "eval_vs_wallclock")


def fig_throughput(groups, out):
    cfgs = order(groups)
    panels = [
        ("step_s", "seconds / step"),
        ("gen_tokens_per_s", "generated tokens / s (vLLM)"),
        ("gpu_util_vllm", "vLLM GPU utilisation (%)"),
        ("gpu_util_trainer", "trainer GPU utilisation (%)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.2))
    for ax, (key, title) in zip(axes, panels):
        means = [[r.mean(key) for r in groups[c]] for c in cfgs]
        for i, (c, vals) in enumerate(zip(cfgs, means)):
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            ax.bar(i, np.mean(vals), color=color(c), width=0.7, edgecolor=SURFACE, linewidth=2)
            if len(vals) > 1:
                ax.vlines(i, min(vals), max(vals), color=INK, linewidth=1)
            ax.annotate(
                fmt_value(np.mean(vals)),
                (i, np.mean(vals)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                color=INK_2,
            )
        ax.set_xticks(range(len(cfgs)), cfgs, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Throughput and utilisation (mean over steps, excluding the first 3; line = min–max over seeds)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "throughput")


def fig_staleness(groups, out):
    cfgs = [c for c in order(groups) if any(r.rows and r.rows[0].get("staleness_hist") is not None for r in groups[c])]
    if not cfgs:
        return
    kmax = max(
        int(k) for c in cfgs for r in groups[c] for row in r.rows for k in (row.get("staleness_hist") or {"0": 0})
    )
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.2 * len(cfgs), 3), sharey=True, sharex=True, squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        counts: dict[int, int] = {}
        for r in groups[cfg]:
            for row in r.rows:
                for k, n in (row.get("staleness_hist") or {}).items():
                    counts[int(k)] = counts.get(int(k), 0) + n
        total = sum(counts.values()) or 1
        ks = list(range(kmax + 1))
        ax.bar(ks, [counts.get(k, 0) / total for k in ks], color=color(cfg), width=0.7, edgecolor=SURFACE, linewidth=2)
        ax.set_title(cfg, fontsize=9)
        ax.set_xlabel("staleness (policy versions behind)")
        ax.set_xticks(ks)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("fraction of trained samples")
    fig.suptitle(
        "Staleness of trained samples (all steps, all seeds)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "staleness")


def fig_offpolicy(groups, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for cfg in order(groups):
        band(axes[0], groups[cfg], "logratio_abs_p95", "step", cfg)
        band(axes[1], groups[cfg], "clip_frac", "step", cfg)
    axes[0].set_title("p95 |log π_current − log π_behaviour| per step")
    axes[1].set_title("fraction of tokens PPO-clipped")
    for ax in axes:
        ax.set_xlabel("optimizer step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=6)
    for ax in axes:
        place_end_labels(ax)
    save(fig, out, "offpolicy")


def fig_controller(groups, out):
    ctrl_runs = [r for runs in groups.values() for r in runs if r.decisions]
    if not ctrl_runs:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    linestyles = ["-", "--", ":"]
    for i, r in enumerate(ctrl_runs):
        ls = linestyles[i % len(linestyles)]
        c = color(r.config)
        lbl = f"{r.config} seed {r.seed}"
        xs, ys = r.series("reward")
        axes[0].plot(xs, ema(ys, SMOOTH), color=c, linestyle=ls, label=lbl)
        cfg_ctrl = r.start.get("controller") or {}
        alpha = cfg_ctrl.get("ema_alpha", 0.2)
        signal = cfg_ctrl.get("signal", "logratio_abs_p95")
        xs, ys = r.series(signal)
        axes[1].plot(xs, ema(ys, alpha), color=c, linestyle=ls, label=lbl)
        xs, ys = r.series(
            "max_staleness" if cfg_ctrl.get("knob", "max_staleness") == "max_staleness" else "learning_rate"
        )
        axes[2].step(xs, ys, where="post", color=c, linestyle=ls, label=lbl)
        for d in r.decisions:
            if d["changed"]:
                axes[2].plot(
                    d["step"],
                    d["new_value"],
                    marker="v" if d["new_value"] < d["old_value"] else "^",
                    color=c,
                    markersize=8,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    linestyle="none",
                )
        for key in ("high", "low"):
            if key in cfg_ctrl:
                axes[1].axhline(cfg_ctrl[key], color=INK_2, linewidth=1, linestyle="--")
                axes[1].annotate(key, (0, cfg_ctrl[key]), xytext=(2, 3), textcoords="offset points", color=INK_2)
    axes[0].set_title(f"Train reward (EMA α={SMOOTH})")
    axes[1].set_title("Controller signal: drift = EMA(signal), with thresholds")
    axes[2].set_title("Knob value (▼ tighten, ▲ loosen)")
    axes[2].set_xlabel("optimizer step")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "controller")


def summary(groups, out, target_frac: float):
    """Results table. Two accuracy targets, because the SPEC's can be met by the base model's neighbourhood:

    * spec: ``target_frac`` x sync mean final accuracy (as written in the SPEC);
    * gain: base + ``target_frac`` x (sync mean final - base), i.e. 90% of sync's *improvement*. With a base model
      already at ~0.6 and sync ending near ~0.72, the spec target (~0.65) is crossed by every run at its first
      checkpoint, so it only measures time to step 50.
    """

    def final_acc(r: Run):
        # Only a complete eval counts: the last evaluated checkpoint must be the last training step.
        if not r.evals or not r.rows or r.evals[-1]["step"] < r.rows[-1]["step"]:
            return None
        return r.evals[-1]["accuracy"]

    def best_acc(r: Run):
        return max((e["accuracy"] for e in r.evals if e["step"] > 0), default=None)

    base_vals = [e["accuracy"] for runs in groups.values() for r in runs for e in r.evals if e["step"] == 0]
    base = float(np.mean(base_vals)) if base_vals else None
    n_eval = next((e["n"] for runs in groups.values() for r in runs for e in r.evals if e.get("n")), None)
    sync = [a for r in groups.get("sync", []) if (a := final_acc(r)) is not None]
    sync_final = float(np.mean(sync)) if sync else None
    target = target_frac * sync_final if sync_final is not None else None
    gain_target = base + target_frac * (sync_final - base) if sync_final is not None and base is not None else None
    sync_reward = [r.mean("reward", skip_first=len(r.rows) - 20) for r in groups.get("sync", []) if r.rows]
    reward_target = target_frac * float(np.mean(sync_reward)) if sync_reward else None

    def agg(vals, scale=1.0, fmt="{:.3f}"):
        vals = [v * scale for v in vals if v is not None]
        if not vals:
            return "—", None
        s = fmt.format(np.mean(vals))
        if len(vals) > 1:
            s += f" [{fmt.format(min(vals))}, {fmt.format(max(vals))}]"
        return s, float(np.mean(vals))

    def crossing(r: Run, tgt):
        ev = [(e["wall_s"], e["accuracy"]) for e in r.evals if e.get("wall_s") is not None]
        return first_crossing(*map(np.asarray, zip(*ev)), tgt) if ev and tgt is not None else None

    table = []
    sync_gain_time = None
    for cfg in order(groups):
        runs = groups[cfg]
        tt_gain = [crossing(r, gain_target) for r in runs]
        ttr = []
        for r in runs:
            xs, ys = r.series("reward", "wall_s")
            ttr.append(first_crossing(xs, ema(ys, SMOOTH), reward_target) if xs.size and reward_target else None)
        gain_mean = agg(tt_gain, 1 / 60, "{:.1f}")[1]
        if cfg == "sync":
            sync_gain_time = gain_mean
        table.append(
            {
                "config": cfg,
                "seeds": len(runs),
                "final_eval_acc": agg([final_acc(r) for r in runs])[0],
                "best_eval_acc": agg([best_acc(r) for r in runs])[0],
                "time_to_target_spec_min": agg([crossing(r, target) for r in runs], 1 / 60, "{:.1f}")[0],
                "time_to_target_gain_min": agg(tt_gain, 1 / 60, "{:.1f}")[0],
                "time_to_target_reward_min": agg(ttr, 1 / 60, "{:.1f}")[0],
                "s_per_step": agg([r.mean("step_s") for r in runs], 1, "{:.1f}")[0],
                "train_wall_min": agg([r.final_wall_s for r in runs], 1 / 60, "{:.1f}")[0],
                "gpu_hours": agg([gpu_hours(r) for r in runs], 1, "{:.2f}")[0],
                "_gain_mean": gain_mean,
            }
        )
    for row in table:
        g = row.pop("_gain_mean")
        row["wallclock_saving_vs_sync"] = (
            f"{1 - g / sync_gain_time:.0%}" if g is not None and sync_gain_time and row["config"] != "sync" else "—"
        )
    total = sum(h for runs in groups.values() for r in runs if (h := gpu_hours(r)) is not None)
    cols = list(table[0]) if table else []
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(table)
    se = f"{np.sqrt(sync_final * (1 - sync_final) / n_eval):.3f}" if sync_final is not None and n_eval else "n/a"

    def show(x):
        return f"{x:.3f}" if x is not None else "n/a"

    lines = [
        f"Base model eval accuracy (step 0) = {show(base)}; sync mean final = {show(sync_final)}; "
        f"binomial s.e. of one eval (n={n_eval}) ≈ {se}.",
        f"Target (spec) = {target_frac:.0%} of sync final = {show(target)}.",
        f"Target (gain) = base + {target_frac:.0%} of sync's improvement = {show(gain_target)}.",
        f"Target (reward) = {target_frac:.0%} of sync mean train reward, last 20 steps = {show(reward_target)}.",
        "Eval times are at checkpoint resolution (every save_steps), reward times per step. Wall-clock excludes "
        "startup and checkpoint writes. wallclock_saving_vs_sync uses the gain target.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
        *("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in table),
        "",
        f"Total logged compute: {total:.2f} GPU-hours (jobs with job.json).",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return gain_target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--include", default=None, help="regex on run dir names, e.g. '^(sync|async-s[14])-'")
    p.add_argument("--target_frac", type=float, default=0.9)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    groups = by_config(load_runs(a.runs, a.include))
    if not groups:
        raise SystemExit(f"no runs with metrics.jsonl under {a.runs}")
    target = summary(groups, out, a.target_frac)
    fig_reward(groups, out, "wall_s")
    fig_reward(groups, out, "step")
    fig_eval(groups, out, target)
    fig_throughput(groups, out)
    fig_staleness(groups, out)
    fig_offpolicy(groups, out)
    fig_controller(groups, out)


if __name__ == "__main__":
    main()
