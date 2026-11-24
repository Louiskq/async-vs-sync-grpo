#!/bin/bash
# Submit the core sweep (M3): sync, async-s1, async-s4 x seeds. Run on the login node: bash slurm/sweep.sh [seeds]
set -euo pipefail
SEEDS=${1:-"0 1"}
for run in sync async-s1 async-s4; do
    for seed in $SEEDS; do
        sbatch "slurm/$run.sh" "$seed"
    done
done
