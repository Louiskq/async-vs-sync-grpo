#!/bin/bash
#SBATCH --job-name=agrpo-sync
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/sync.sh [seed] [dotted.override=value ...]   (seed defaults to 0)
# Account/partition come from SBATCH_ACCOUNT / SBATCH_PARTITION in your environment.
SEED=${1:-0}; shift || true
exec bash slurm/_run.sh configs/sync.yaml "$SEED" "$@"
