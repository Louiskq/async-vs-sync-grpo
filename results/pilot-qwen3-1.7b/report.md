| metric | pilot17b-async-s4-seed0 | pilot17b-sync-seed0 |
|---|---|---|
| kind | async | sync |
| steps | 25 | 25 |
| startup_s | 199 | 222 |
| s_per_step | 10.2 | 13.3 |
| trainer_wait_s | 5.87 | 3.38 |
| gen_idle_s | — | 9.94 |
| fwd_bwd_s | 3.13 | — |
| gpu_util_vllm | 50.5 | 15.6 |
| gpu_util_trainer | 37.7 | 72 |
| gen_tokens_per_s | 1.92e+04 | 1.83e+04 |
| train_tokens_per_s | 6.57e+03 | 4.57e+03 |
| rollouts_per_step | 220 | 256 |
| completion_len | 232 | 238 |
| reward | 0.859 | 0.881 |
| zero_std_group_frac | 0.738 | 0.75 |
| staleness_mean | 1.16 | 0 |
| dropped_stale_per_step | 0 | 0 |
| logratio_abs_p95 | 0.0562 | 0.0437 |
| logratio_std | 0.0407 | 0.0346 |
| is_outside_frac | 7.29e-05 | 4.95e-05 |
| clip_frac | 0.00403 | 0 |
| groups_trained_per_step | 27.5 | 32 |
| prompts_generated_per_step | 105 | 32 |

Balance (async only helps if generation and training take comparable time):
  pilot17b-async-s4-seed0: trainer waits 5.87s/step for data, generator held back —s/step; fwd+bwd 3.13s of 10.2s
  pilot17b-sync-seed0: generation 3.4s vs rest of step 9.9s per step (generation share 25%)

Projection: 300 steps x 2 seeds, 2 GPUs, + startup + 12 min eval per run:
  like pilot17b-async-s4-seed0: 1.11 h/run -> 4.4 GPU-h for 2 seeds
  like pilot17b-sync-seed0: 1.37 h/run -> 5.5 GPU-h for 2 seeds
  per-config estimates above; core sweep = sync + 2 async configs (+ controller in M4): ~19.8 GPU-h for 4 configs at the mean rate
