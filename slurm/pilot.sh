#!/bin/bash
#SBATCH --job-name=agrpo-pilot
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Overrides are parsed as YAML: a bare no would become False, hence the quoted "no".
# M2: 25 steps each of sync and async-s4 on the base model, logged as pilot-* runs. No checkpoints, no eval.
#   sbatch slurm/pilot.sh
export SKIP_EVAL=1
STEPS=${STEPS:-25}
bash slurm/_run.sh configs/sync.yaml 0 name=pilot-sync common.max_steps=$STEPS common.save_strategy='"no"'
bash slurm/_run.sh configs/async-s4.yaml 0 name=pilot-async-s4 common.max_steps=$STEPS common.save_strategy='"no"'
