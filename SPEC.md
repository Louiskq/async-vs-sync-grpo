# SPEC — Async GRPO + within-run hyperparameter control (small demo)

## Goal

A small, reproducible GitHub project that shows two things:

1. **Async RL works and is faster:** asynchronous GRPO (generation and training on separate GPUs, overlapping) reaches the same accuracy as synchronous GRPO in less wall-clock time.
2. **Async creates a new hyperparameter worth controlling:** a simple controller that adjusts a knob during the run, using signals that async training produces (staleness / off-policy drift), matches or beats the best fixed setting.

Headline result, stated on the first line of the README: *"Async GRPO reaches sync accuracy in X% less wall-clock time; an adaptive controller matches/beats the best fixed staleness."*

Budget: **≤ 25 GPU-hours total**, about 1 week of work. Keep the scope small; stretch items are clearly marked.

Background reading: https://huggingface.co/blog/async-rl-training-landscape (survey of 16 async RL libraries; it contains no original experiments, which is the gap this project fills at small scale).

---

## Fixed experimental setup

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` (use `Qwen/Qwen3-0.6B` for smoke tests) |
| Task | GSM8K (`openai/gsm8k`, `main`), thinking disabled |
| Reward | Exact match on the final numeric answer (robust parsing: last number / `####` / boxed); 1.0 or 0.0 |
| Eval | Held-out GSM8K test subset (e.g. 500 problems), greedy, run every N steps by a separate process or at fixed checkpoints, so eval cost doesn't distort training wall-clock |
| Rollouts | 32 prompts × 8 generations = 256 rollouts/step (adjust after the pilot) |
| Completion length | `max_completion_length` ≈ 512 |
| Steps | ~300 |
| Hardware | 2 GPUs, the **same split for every run**: GPU 0 = vLLM, GPU 1 = trainer |
| Seeds | 2 per config (3 if budget allows) |

Everything else (LR, β/KL, clip ε, optimizer, batch) is held identical across runs except the variable under study.

---

## Environment

- Target cluster: **Isambard-AI (GH200, aarch64/ARM)**, Slurm. Fallback: any x86 node with ≥2 GPUs.
- The ARM vLLM install is the #1 risk. **Milestone 0 exists to retire it on day one.** Prefer a prebuilt aarch64 wheel / container if one is available; otherwise document the build steps in `docs/setup.md`.
- Install order (TRL's AsyncGRPO docs warn of a dependency conflict):
  ```bash
  pip install 'vllm>=0.17.1'
  pip install 'transformers>=5.2.0' --no-deps
  pip install trl   # pin the exact version that ends up working
  ```
- **Pin every version** in `requirements.lock` / `environment.md`. `AsyncGRPOTrainer` is under `trl.experimental`, so the API may change.
- Verify all parameter names below against the **installed** TRL source, not the docs. The notes here come from the TRL v1.3.0 docs.

### Known TRL AsyncGRPO facts (verify)

- Import: `from trl.experimental.async_grpo import AsyncGRPOTrainer` (+ `AsyncGRPOConfig`).
- Distributed training supports only FSDP2 (no DeepSpeed ZeRO). With 1 trainer GPU this doesn't matter.
- The vLLM server must run on a separate GPU, with dev mode and NCCL weight transfer:
  ```bash
  CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-1.7B \
      --max-model-len <prompt_len + max_completion_length> \
      --logprobs-mode processed_logprobs \
      --weight-transfer-config '{"backend":"nccl"}'
  CUDA_VISIBLE_DEVICES=1 accelerate launch train_async_grpo.py
  ```
- Relevant config knobs: `max_staleness` (how many weight updates a sample may lag before it's discarded), `weight_sync_steps`, `max_inflight_tasks` (defaults to `max_staleness × per_device_train_batch_size × gradient_accumulation_steps × num_processes`).
- The docs don't say whether importance-sampling (IS) correction is applied. **Read the source and record the answer in `docs/notes.md`.** The project logs the behaviour-vs-current log-ratio either way.

---

## Repository layout

```
async-grpo-hpcontrol/
├── README.md                 # headline result, plots, how to reproduce
├── requirements.lock
├── configs/                  # one YAML per run config
├── src/
│   ├── data.py               # GSM8K loading, prompt formatting
│   ├── reward.py             # answer extraction + exact-match reward (unit-tested)
│   ├── train_sync.py         # GRPOTrainer baseline (vLLM server mode)
│   ├── train_async.py        # AsyncGRPOTrainer
│   ├── signals.py            # computes off-policy/staleness signals from logged tensors
│   ├── controller.py         # rule-based controller (+ optional LLM controller)
│   ├── eval.py               # held-out eval of checkpoints
│   └── logging_utils.py      # unified metrics → JSONL (+ optional wandb)
├── slurm/                    # sbatch scripts: pilot, sync, async, controller
├── analysis/
│   └── plots.py              # all figures from logged JSONL
├── tests/                    # reward parsing, signal computation, controller logic
└── docs/
    ├── setup.md              # environment/ARM notes
    └── notes.md              # findings about TRL internals (IS, staleness mutability)
```

---

## Metrics to log (every step, to JSONL)

- `step`, **wall-clock seconds since the first training step** (exclude vLLM startup/model load, and record those separately)
- Mean train reward, mean completion length
- Tokens generated/s (vLLM side) and tokens trained/s (trainer side)
- GPU utilisation for both GPUs (sample `nvidia-smi` / NVML every ~1 s in a side thread)
- Trainer idle time waiting for data; generator idle time waiting for weights
- **Staleness per batch:** distribution of `current_version − sample_version` (mean, max)
- Samples discarded for staleness
- **Off-policy signals:** per-token log-ratio `log π_current − log π_behaviour` (mean, std, p95 of |ratio|), fraction of tokens clipped
- Current value of any controlled knob
- Eval accuracy (at eval points)

---

## Runs

| ID | Trainer | Knob | Purpose |
|---|---|---|---|
| `sync` | `GRPOTrainer`, vLLM server mode, same 2-GPU split | — | Baseline |
| `async-s1` | `AsyncGRPOTrainer` | `max_staleness=1` | Low-staleness async |
| `async-s4` | `AsyncGRPOTrainer` | `max_staleness=4` | Higher-throughput async |
| `async-ctrl` | `AsyncGRPOTrainer` | controller-driven | HP control |
| *(stretch)* `async-s8` | `AsyncGRPOTrainer` | `max_staleness=8` | Shows where staleness starts hurting |
| *(stretch)* `async-llmctrl` | `AsyncGRPOTrainer` | LLM controller | Ties in the dissertation's agentic controller |

Core: 4 configs × 2 seeds = 8 runs (~1–2 GPU-h each).

---

## Controller design

**Primary knob: `max_staleness`** (trades throughput against off-policyness, so it's the most interesting knob for async).
- First, check in the TRL source whether `max_staleness` (and the derived `max_inflight_tasks`) can be changed at runtime, e.g. from a `TrainerCallback` mutating the rollout worker's state. If yes, use it. If it needs a small, clean patch, do that in a subclass and document it.
- **Fallback knob: learning rate** (easy to change via optimizer param groups in a callback), still driven by the off-policy signals below. Record which knob was used and why in `docs/notes.md`.

**Rule-based controller (core):**
- Runs every `K` steps (e.g. K=10), using an EMA of the signals.
- Signal: `drift = EMA(p95 |log-ratio|)` (or the clip fraction, whichever is less noisy in the pilot).
- If `drift > high_threshold`, tighten: staleness −1 (min 1), or LR × 0.7.
- If `drift < low_threshold` and reward isn't falling, loosen: staleness +1 (max 8), or LR × 1.2 up to a cap.
- Hysteresis/cooldown so it doesn't oscillate. Thresholds are set from the pilot's signal distribution, not guessed.
- Log every decision (step, signal values, old → new value, reason).

**LLM controller (stretch):** same interface (`decide(signal_history) -> new_value, reason`), backed by an LLM call every K steps with a compact signal summary. Calls must be non-blocking for training; if a call is slow, keep the current value.

---

## Milestones and gates

Stop at each gate and report before continuing.

**M0 — Environment smoke test (gate: async works end-to-end)**
- vLLM server and `AsyncGRPOTrainer` run on Qwen3-0.6B for ~20 steps on 2 GPUs; weights sync over NCCL; metrics JSONL written.
- Deliver: `docs/setup.md` with exact working install steps + pinned versions.

**M1 — Reward + data + logging (gate: tests pass)**
- `reward.py` with unit tests covering GSM8K answer formats; `signals.py` tested on synthetic tensors.
- Unified logger including GPU-utilisation sampling and staleness/log-ratio metrics.

**M2 — Pilot (gate: report measured numbers)**
- 20–30 steps each of `sync` and `async-s4` on Qwen3-1.7B.
- Report: seconds/step, generation vs training time per step, GPU utilisation, log-ratio distribution.
- **Check the balance:** async only helps if generation and training times are comparable. If generation is far faster, increase generations per prompt or completion length; if far slower, reduce them. Propose the final step config and the projected GPU-hours before running the full sweep.
- Report findings on IS correction and on whether `max_staleness` can be changed mid-run.

**M3 — Core sweep (gate: plots reviewed)**
- `sync`, `async-s1`, `async-s4` × 2 seeds, via Slurm scripts.
- Plots: reward vs **wall-clock** (seed bands), reward vs step, throughput/GPU-util bars, staleness distribution.

**M4 — Controller (gate: controller trace reviewed)**
- Implement and tune thresholds on pilot data only (no tuning on the final runs).
- `async-ctrl` × 2 seeds. Plot: knob value + signal over time with decision markers, overlaid on the reward curve; compare against the best fixed setting.

**M5 — Write-up**
- README: headline result, the 3–4 key figures, a table of final eval accuracy and time-to-target-accuracy per config (mean ± range across seeds), exact reproduce commands, compute used, limitations (small model, short completions, 2 seeds).

*(Stretch, only after M5)*: `async-s8`, `async-llmctrl`, a third seed.

---

## Acceptance criteria

- One command per run (`sbatch slurm/<run>.sh` or `python -m src.train_* --config configs/<run>.yaml`); all plots regenerate from logs with `python analysis/plots.py`.
- Sync vs async comparison uses an identical GPU split, model, data order, and hyperparameters apart from the variable under study.
- Wall-clock excludes startup; time-to-target-accuracy is reported (target = e.g. 90% of the sync run's final eval accuracy).
- Every claim in the README is backed by a figure or table from the logged runs.
- Total compute logged in the README; stays within ~25 GPU-hours (flag before exceeding).

## Non-goals

- Multi-node, MoE, LoRA, models >2B, long-context reasoning, beating SOTA GSM8K scores, writing a custom async trainer from scratch (patch/subclass TRL only where needed for the controller).
