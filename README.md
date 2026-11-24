# Async GRPO + within-run hyperparameter control

**Headline result: pending.** It will read: *"Async GRPO reaches sync accuracy in X% less wall-clock time; an
adaptive controller matches/beats the best fixed staleness."* Each number in that sentence will come from
`results/summary.md`.

A small, reproducible comparison on Qwen3-0.6B / GSM8K (1.7B was near-ceiling: docs/decisions.md) with 2 GPUs (GPU 0 = vLLM, GPU 1 = trainer) of:

1. **Synchronous vs asynchronous GRPO.** Same model, data order, GPU split, vLLM server and hyperparameters; only
   whether generation overlaps training (TRL `GRPOTrainer` vs `AsyncGRPOTrainer`).
2. **A controller for the knob async introduces.** `max_staleness` is adjusted during the run from the measured
   off-policy drift (p95 |log π_current − log π_behaviour|), compared against the best fixed setting.

Background: [Async RL training landscape](https://huggingface.co/blog/async-rl-training-landscape), a survey with
no original experiments. This repo adds small-scale measurements.

## Status

| Milestone | State |
|---|---|
| M0 environment smoke test | passed on Isambard-AI GH200 (docs/setup.md) |
| M1 reward, data, logging | done: unit-tested (`pytest`) |
| M2 pilot | 1.7B pilot run → switched to 0.6B (docs/decisions.md); 0.6B re-pilot pending |
| M3 core sweep | done: results/m3/ (figures + summary.md), logs in results/logs/ — awaiting review |
| M4 controller | implemented and unit-tested; thresholds come from the pilot |
| M5 write-up | — |

## Reproduce

```bash
# setup (once): see docs/setup.md
sbatch slurm/setup_env.sh && sbatch slurm/smoke.sh

sbatch slurm/pilot.sh                              # M2: 25 steps each of sync and async-s4
python analysis/thresholds.py --include '^pilot-'  # controller thresholds from the pilot only

bash slurm/sweep.sh "0 1"                          # M3: sync, async-s1, async-s4 x 2 seeds
sbatch slurm/async-ctrl.sh 0; sbatch slurm/async-ctrl.sh 1   # M4

python analysis/plots.py --runs results/logs --out results/m3   # every M3 figure + summary.md, from committed logs
```

One run is `sbatch slurm/<config>.sh <seed>`, which wraps `python -m src.train_{sync,async} --config configs/<config>.yaml`.

## What is held fixed and what varies

All runs inherit `configs/base.yaml`. `tests/test_config.py` checks that `sync`, `async-s1` and `async-s4` share
every `common` hyperparameter and that the two async configs differ only in `max_staleness`. Where the two TRL
trainers differ by design, the difference is documented in `docs/notes.md`: masked IS in both, dropped
zero-advantage groups and token-budgeted batches in async.

## Layout

```
src/        data, reward, train_sync, train_async, signals, controller, logging_utils, eval, config
configs/    one YAML per run (base.yaml holds everything shared)
slurm/      setup_env, smoke, pilot, one script per run, sweep
analysis/   plots.py (all figures + summary table), thresholds.py (controller thresholds from the pilot)
docs/       setup.md (environment / ARM notes), notes.md (TRL internals: IS correction, staleness mutability)
tests/      reward parsing, signals, controller logic + knob setters on real TRL classes, logging, configs, plots
```

## Logged metrics

`<run>/metrics.jsonl` has one row per optimizer step. Fields:
- wall-clock since the first training step, excluding startup and checkpoint writes
- reward and completion length
- generated tokens/s and trained tokens/s
- GPU utilisation of both GPUs (NVML, 1 s)
- trainer time waiting for data and generator time held back
- staleness (mean, max, per-sample histogram) and samples dropped for staleness
- the log-ratio distribution and the clip fraction
- the controlled knob's value and every controller decision

`<run>/eval.jsonl` holds held-out accuracy (500 test problems, greedy) per checkpoint, evaluated after training so
eval does not distort wall-clock. `<run>/job.json` holds the job's wall time, used for the compute total.

## Compute

Budget ≤ 25 GPU-hours. The total is reported from `job.json` files in `results/summary.md`.

## Limitations

To be written after the results.
