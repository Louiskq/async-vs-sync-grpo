# Notes on TRL internals (TRL 1.18.2)

Read from the installed source (`trl/trainer/async_grpo_trainer.py`, `async_grpo_config.py`, `grpo_trainer.py`,
`grpo_config.py`), not from the docs. The SPEC's notes came from the v1.3.0 docs, and several of them no longer hold.
Line references are to TRL 1.18.2.

## Differences from the SPEC's assumptions

| SPEC assumed (v1.3.0 docs) | TRL 1.18.2 source | What this project does |
|---|---|---|
| `from trl.experimental.async_grpo import AsyncGRPOTrainer` | Promoted to the stable API: `from trl import AsyncGRPOTrainer, AsyncGRPOConfig`. The experimental path still works but warns that it will be removed in v2. | Stable import. |
| `max_inflight_tasks` knob | Gone. In-flight generation is bounded by a **token allowance** (below), `max_concurrent_rollouts` (default 512) and a KV-cache-driven concurrency cap. | Nothing to set. |
| Step = fixed number of rollouts | Each optimizer step is `gradient_accumulation_steps` micro-batches of ≤ `token_budget` *unique* tokens (tree-packed, so a group's shared prompt counts once). The number of samples per step varies. | `token_budget × GA` is calibrated in the pilot to ≈ 256 rollouts/step. `batch/samples_per_step` is logged every step. |
| — | Groups whose rewards are all equal (zero advantage) are **dropped** before training. Sync `GRPOTrainer` keeps them in the batch, where they give zero gradient. | Async therefore consumes more prompts per optimizer step than sync. Compare on wall-clock, and report prompts consumed. |
| — | Default objective is **score centering** (`score_centering=True`), not the PPO clipped objective. | `score_centering: false`, so both trainers use the clipped objective. |
| — | No KL term and no reference model. | Sync `beta: 0.0`. |
| — | Default `dtype="float32"` (fp32 master weights, bf16 autocast with `bf16=True`). Sync `GRPOTrainer` loads in the checkpoint's dtype unless told otherwise. | Both load fp32 with bf16 autocast (`model_init_kwargs.dtype` for sync). |
| — | Rewards run in a spawned child process with `CUDA_VISIBLE_DEVICES=""`. | `gsm8k_reward` is a module-level, picklable, CPU-only function. |

The vLLM server command is **the same for both trainers**: sync `GRPOTrainer` in server mode now talks to
`vllm serve` (dev mode + NCCL weight-transfer engine), not to `trl vllm-serve`. So the 2-GPU split and the server
are identical across sync and async runs (`slurm/_run.sh`).

## Is importance-sampling correction applied? Yes, in both trainers

**Async** (`AsyncGRPOTrainer.compute_loss`, ~L3455–3480): `old_log_probs` are the **vLLM sampler's** per-token
logprobs, returned with each completion. The current policy's logprobs come from the training forward pass. Then:

```
log_ratio = log_probs - old_log_probs          # log pi_theta - log pi_behaviour (staleness + numerical mismatch)
coef_1    = exp(log_ratio)
is_masked = (coef_1 < vllm_importance_sampling_clip_min) | (coef_1 > vllm_importance_sampling_clip_max)  # 0.5, 5.0
# score_centering=False:
loss_tok  = -min(coef_1 * A, clip(coef_1, 1-eps, 1+eps) * A) * ~is_masked
```

There is one gradient step per batch, so there is no separate "old policy". The PPO ratio *is* the importance
ratio against the behaviour policy, and it covers both staleness (the sample was generated `s` weight versions ago)
and trainer/vLLM numerical mismatch. Tokens outside [0.5, 5] are dropped from the loss ("masked IS").

**Sync** (`GRPOTrainer._generate_and_score_completions`, ~L2828–2872): with `vllm_importance_sampling_correction=True`
(the default), the trainer recomputes its own logprobs at generation time (`old_per_token_logps`, an extra forward
pass) and multiplies the per-token loss by `exp(old - vllm)`, masked outside the bounds. The PPO ratio
`pi_theta / pi_old` is 1 on the single step. Its default mode is `sequence_mask` with `clip_max=3`. This project
sets `token_mask` and [0.5, 5] so that both trainers mask the same tokens.

**What we log:** the same quantity, `log pi_current - log pi_behaviour`, for every trained token in both trainers:
mean, std, mean |·|, **p95 |·|**, max |·|, and the fraction outside [0.5, 5]. TRL itself only logs the mean and max.
In async this captures staleness plus mismatch; in sync, mismatch only (the staleness-0 floor).
- Async: a forward hook on the model inside `compute_loss` reads `outputs.log_probs`. Token selection mirrors TRL's
  first lines, and no loss code is copied (`src/train_async.py`).
- Sync: read from the batch dict TRL returns (`old_per_token_logps − sampling_per_token_logps`) (`src/train_sync.py`).

## Can `max_staleness` be changed mid-run? Yes, without patching TRL

`max_staleness` is read in exactly three places, and all of them are in the **trainer** process:

1. `RolloutQueueDataset.__iter__` (~L2311) reads `self.max_staleness` on every sample: the drop filter.
2. `AsyncGRPOTrainer._log_step_metrics` (~L3628) reads `self.args.max_staleness` every optimizer step to size the
   generation lookahead: `lookahead = max(max_staleness, 1) × tokens consumed last step`. It writes
   `consumed + lookahead` into the shared `mp.Value` `token_allowance`.
3. `get_train_dataloader` (~L3387) sets the initial lookahead once.

The rollout child never reads `config.max_staleness` (it only appears there in a log message, and in LoRA-adapter
bookkeeping, which we don't use). It starts a new group only while queued + in-flight tokens < `token_allowance`
(`AsyncRolloutLoop._fits_token_allowance`, ~L1038). So changing the value in the trainer process takes effect:

- **immediately** for the drop filter, and
- **at the next step end** for generation throttling, which reaches the child through the shared value.

`src/controller.py:set_max_staleness` sets `trainer.args.max_staleness` and `trainer._rollout_dataset.max_staleness`.
`tests/test_controller.py::test_set_max_staleness_changes_real_trl_drop_filter` drives the real TRL
`RolloutQueueDataset` and shows that the drop behaviour changes mid-iteration.

**Cost of tightening:** samples already queued that exceed the new limit get dropped. That is wasted generation,
and it is visible in `dropped_stale`.

**Knob used:** `max_staleness`, since the primary knob works. The learning-rate fallback is implemented and tested
(`set_learning_rate` updates the optimizer param groups and the constant scheduler's `base_lrs`) but is not used.

## Staleness definition

`staleness = trainer model_version at consumption − model_version when the group's first rollout started`
(`RolloutGroup.model_version`). `model_version` increments on every weight sync (`weight_sync_steps=1` → every
optimizer step). TRL logs only the per-step mean and max. We also record the per-sample histogram
(`staleness_hist`), read from TRL's metric list before it is averaged.

## Other things worth knowing

- `vllm_server_timeout` (1800 s) also covers waiting for the server at startup, so the trainer can start while vLLM is
  still loading.
- The trainer warns when the vLLM dtype (bf16 here) differs from its own dtype (fp32 master weights). That is expected:
  the mismatch is part of what the log-ratio measures, and it is the same for both trainers.
- `lr_scheduler_type` defaults to `constant` in AsyncGRPO but `linear` in GRPO. Both are set to `constant`.
- The async epoch logic counts prompts actually trained on. With `max_steps` set, `max_steps` is the stop condition.
