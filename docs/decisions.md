# Decisions

Changes to the SPEC's fixed setup, each with the evidence behind it. The user made the choice each time; the data is
in `results/`.

## 2026-11-24 — Model: Qwen3-0.6B instead of Qwen3-1.7B (after the M2 pilot)

**Evidence:** the M2 pilot on Qwen3-1.7B (`results/pilot-qwen3-1.7b/report.md`, 25 steps each):

- Train reward was already 0.86 (async) / 0.88 (sync) at temperature 1.0 from step 1, so there was little headroom.
- 74–75% of prompt groups had identical rewards across all 8 samples (mostly all correct), so they carried no
  advantage signal. AsyncGRPOTrainer drops such groups, so to fill a batch it generated **105 prompts per step**
  (vs sync's 32) and the trainer waited 5.9 s/step for data.
- Generation therefore never ran ahead of training: with `max_staleness=4`, mean staleness was only 1.2. The
  variable under study (staleness) and the controller's knob would barely be exercised.

In the M0 smoke test, Qwen3-0.6B on GSM8K had train reward ≈ 0.63 with 37% uninformative groups: real headroom, at
about half the cost per run.

**Options considered:**
- 0.6B on GSM8K (chosen).
- 1.7B on a harder numeric math set (more data and parsing work, longer completions, higher cost).
- Keep 1.7B on GSM8K (near-ceiling, weak staleness and controller story).

**Consequence:** a 0.6B re-pilot is needed. The controller thresholds must come from pilot data on the setup that is
actually swept, so the 1.7B pilot is kept as evidence but not used for thresholds.

## 2026-11-24 — Learning rate 3e-6 instead of 1e-6; async token_budget 15000 (after the 0.6B pilot)

**Evidence** (`results/pilot-qwen3-0.6b-lr1e-6/report.md`): at LR 1e-6 async was ~34% faster per step (5.6 vs 8.5 s)
and reward rose in both runs. But staleness of 1–2 barely changed the policy: clip fraction was 0.17% (sync 0), and
p95 |log-ratio| was 0.076 vs 0.073, within its own step-to-step noise. The variable under study, and the controller's
input, would hardly move, so "the controller matches the best fixed setting" would hold almost by construction.

**Choice:** re-pilot at LR 3e-6, a common setting for sub-1B GRPO, so that staleness has a measurable effect. This
risks instability, which the pilot will show. All runs use the same LR.

**Batch:** at 4 × 18432 tokens async trained 315 rollouts/step, against sync's 256; 4 × 15000 targets 256. The new
pilot uses the final sweep config, so its thresholds apply directly.

**Also:** the controller signal is now chosen by async-vs-sync separation over noise, not by relative noise alone
(clip fraction 4.2× vs p95 |log-ratio| 0.6× at 1e-6). See `analysis/thresholds.py`.

## 2026-11-24 — Add async-s0 before building the controller (after M3)

**Evidence** (`results/m3/summary.md`): async-s1 and async-s4 were indistinguishable. s/step was 5.5 vs 5.9, final
eval 0.724 vs 0.724, and generation kept up with one step of lookahead. The clip fraction tracked training phase
(0.0025 → 0.0045 over the run, higher for s=1 than s=4 late on) more than staleness. A max_staleness controller would
therefore "match the best fixed setting" only because every setting is equal.

**Choice:** run async-s0 × 2 seeds first. That maps the actual staleness/throughput curve (s=0 → 1 → 4) before
deciding what M4 should control. In TRL, s=0 drops anything generated before the latest weight sync.

### Result of the async-s0 runs (2026-11-24)

| | sync | async-s0 | async-s1 | async-s4 |
|---|---|---|---|---|
| s/step | 9.9 | 13.1 | 5.5 | 5.9 |
| samples dropped as stale / step | — | 241 | 2.7 | 0 |
| best eval acc | 0.754 | 0.733 | 0.758 | 0.737 |

Fully on-policy async (s=0) is *slower than sync*: it discards what was generated while a step trained, and the trainer
idles. One step of staleness gives the whole speed-up; more gives nothing here (`results/m3/staleness_curve.png`).

**Instability seen in 2 of 8 runs.** sync seed 1 and async-s0 seed 1 had a completion-length blow-up after about step
250: >90% of completions hit the 512-token cap. async-s0 seed 1's eval fell to 0.494 at step 300. The other six runs
grew more slowly (truncation at most 26% in the last 50 steps). Both blow-ups were in on-policy configs, but two seeds
cannot support a claim that staleness protects against this. Step-300 accuracy is therefore partly contaminated, and
best accuracy and time-to-target are reported alongside it.

## 2026-11-24 — M4: the staleness controller was not run

**Evidence:** s=0 is strictly worse than s≥1 (13.1 vs 5.5–5.9 s/step), and s=1 and s=4 are indistinguishable in
speed and accuracy. A `max_staleness` controller would match the best fixed setting by construction. The drift
signals track training phase more than staleness. A learning-rate controller (the SPEC's fallback) would need
late-training thresholds, but only the M3 runs show late-training behaviour, and setting thresholds from them would
break the "pilot data only" rule unless a new, longer pilot were run first.

**Choice:** skip the controller runs and write up. The controller code, tests and `configs/async-ctrl.yaml` stay in the
repo. 7.8 GPU-hours of the budget are left unspent.
