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
