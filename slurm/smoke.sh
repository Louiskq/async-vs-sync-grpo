#!/bin/bash
#SBATCH --job-name=agrpo-smoke
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# M0: Qwen3-0.6B, 20 steps async then 20 steps sync, no checkpoints, no eval.
#   sbatch slurm/smoke.sh
export SKIP_EVAL=1
bash slurm/_run.sh configs/smoke-async.yaml 0
bash slurm/_run.sh configs/smoke-sync.yaml 0
source slurm/_common.sh
python - <<'PY'
import os
from src.logging_utils import read_jsonl
for name in ("smoke-async-seed0", "smoke-sync-seed0"):
    rows = [r for r in read_jsonl(f"{os.environ['RUNS_DIR']}/{name}/metrics.jsonl") if "event" not in r]
    last = rows[-1]
    print(f"{name}: {len(rows)} steps, {last['wall_s']:.0f}s, reward {last['reward']}, "
          f"p95|lr| {last.get('logratio_abs_p95')}, staleness_max {last.get('staleness_max')}")
PY
