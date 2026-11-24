| metric | pilot-async-s4-seed0 | pilot-sync-seed0 |
|---|---|---|
| kind | async | sync |
| steps | 25 | 25 |
| startup_s | 182 | 163 |
| s_per_step | 3.83 | 8.86 |
| trainer_wait_s | 0.923 | 2.75 |
| gen_idle_s | — | 6.11 |
| fwd_bwd_s | 1.74 | — |
| gpu_util_vllm | 70.8 | 20.4 |
| gpu_util_trainer | 54 | 64.5 |
| gen_tokens_per_s | 2.24e+04 | 2.47e+04 |
| train_tokens_per_s | 1.59e+04 | 7.68e+03 |
| rollouts_per_step | 246 | 256 |
| completion_len | 226 | 267 |
| reward | 0.625 | 0.677 |
| zero_std_group_frac | 0.378 | 0.409 |
| staleness_mean | 2.14 | 0 |
| dropped_stale_per_step | 0 | 0 |
| logratio_abs_p95 | 0.0905 | 0.0734 |
| logratio_std | 0.0448 | 0.0315 |
| is_outside_frac | 0.000192 | 9.21e-06 |
| clip_frac | 0.00441 | 0 |
| groups_trained_per_step | 30.5 | 32 |
| prompts_generated_per_step | 49.1 | 32 |

Balance (async only helps if generation and training take comparable time):
  pilot-async-s4-seed0: trainer waits 0.923s/step for data, generator held back —s/step; fwd+bwd 1.74s of 3.8s
  pilot-sync-seed0: generation 2.7s vs rest of step 6.1s per step (generation share 31%)

Projection: 300 steps x 2 seeds, 2 GPUs, + startup + 12 min eval per run:
  like pilot-async-s4-seed0: 0.57 h/run -> 2.3 GPU-h for 2 seeds
  like pilot-sync-seed0: 0.98 h/run -> 3.9 GPU-h for 2 seeds
  per-config estimates above; core sweep = sync + 2 async configs (+ controller in M4): ~12.4 GPU-h for 4 configs at the mean rate
