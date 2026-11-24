Base model eval accuracy (step 0) = 0.598; sync mean final = 0.732; binomial s.e. of one eval (n=500) ≈ 0.020.
Target (spec) = 90% of sync final = 0.659.
Target (gain) = base + 90% of sync's improvement = 0.719.
Target (reward) = 90% of sync mean train reward, last 20 steps = 0.722.
Eval times are at checkpoint resolution (every save_steps), reward times per step. Wall-clock excludes startup and checkpoint writes. wallclock_saving_vs_sync uses the gain target.

| config | seeds | final_eval_acc | best_eval_acc | time_to_target_spec_min | time_to_target_gain_min | time_to_target_reward_min | s_per_step | train_wall_min | gpu_hours | wallclock_saving_vs_sync |
|---|---|---|---|---|---|---|---|---|---|---|
| sync | 2 | 0.732 [0.732, 0.732] | 0.754 [0.740, 0.768] | 7.9 [7.4, 8.4] | 16.3 [15.1, 17.4] | 8.4 [7.3, 9.6] | 9.9 [9.2, 10.6] | 49.3 [45.8, 52.9] | 2.02 [1.90, 2.14] | — |
| async-s1 | 2 | 0.724 [0.718, 0.730] | 0.758 [0.730, 0.786] | 3.7 [3.6, 3.8] | 12.6 [7.6, 17.6] | 3.4 [2.9, 3.9] | 5.5 [5.5, 5.5] | 27.5 [27.5, 27.5] | 1.42 [1.41, 1.43] | 22% |
| async-s4 | 2 | 0.724 [0.724, 0.724] | 0.737 [0.726, 0.748] | 3.9 [3.8, 4.0] | 10.6 [8.9, 12.4] | 4.5 [4.2, 4.9] | 5.9 [5.8, 6.0] | 29.5 [28.8, 30.1] | 1.42 [1.40, 1.43] | 35% |

Total logged compute: 9.72 GPU-hours (jobs with job.json).
