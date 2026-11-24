| metric | pilot-async-s4-seed0 | pilot-sync-seed0 |
|---|---|---|
| kind | async | sync |
| steps | 25 | 25 |
| startup_s | 177 | 178 |
| s_per_step | 5.61 | 8.46 |
| trainer_wait_s | 2.39 | 2.42 |
| gen_idle_s | — | 6.04 |
| fwd_bwd_s | 2.16 | — |
| gpu_util_vllm | 43.7 | 20.2 |
| gpu_util_trainer | 41.7 | 66.3 |
| gen_tokens_per_s | 1.89e+04 | 2.39e+04 |
| train_tokens_per_s | 1.29e+04 | 6.82e+03 |
| rollouts_per_step | 315 | 256 |
| completion_len | 216 | 226 |
| reward | 0.608 | 0.632 |
| zero_std_group_frac | 0.373 | 0.335 |
| staleness_mean | 1.67 | 0 |
| dropped_stale_per_step | 0 | 0 |
| logratio_abs_p95 | 0.076 | 0.0734 |
| logratio_std | 0.0323 | 0.0307 |
| is_outside_frac | 7.06e-06 | 1.6e-06 |
| clip_frac | 0.00171 | 0 |
| groups_trained_per_step | 39.3 | 32 |
| prompts_generated_per_step | 62.7 | 32 |

Balance (async only helps if generation and training take comparable time):
  pilot-async-s4-seed0: trainer waits 2.39s/step for data, generator held back —s/step; fwd+bwd 2.16s of 5.6s
  pilot-sync-seed0: generation 2.4s vs rest of step 6.0s per step (generation share 29%)

Projection: 300 steps x 2 seeds, 2 GPUs, + startup + 12 min eval per run:
  like pilot-async-s4-seed0: 0.72 h/run -> 2.9 GPU-h for 2 seeds
  like pilot-sync-seed0: 0.95 h/run -> 3.8 GPU-h for 2 seeds
  per-config estimates above; core sweep = sync + 2 async configs (+ controller in M4): ~13.4 GPU-h for 4 configs at the mean rate
