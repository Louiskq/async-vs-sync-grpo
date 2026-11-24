#!/bin/bash
# One run on 2 GPUs: first assigned GPU = vLLM server, second = trainer; then held-out eval on the first.
#   bash slurm/_run.sh <config.yaml> <seed> [dotted.override=value ...]
# Called from the per-run sbatch scripts; requires an allocation with 2 GPUs.
source slurm/_common.sh
CONFIG=$1; SEED=$2; shift 2
NAME=$(python -m src.config --config "$CONFIG" --get name "$@")
TRAINER=$(python -m src.config --config "$CONFIG" --get trainer "$@")
MODEL=$(python -m src.config --config "$CONFIG" --get model "$@")
MAXLEN=$(python -m src.config --config "$CONFIG" --get max_model_len "$@")
OUT=$RUNS_DIR/$NAME-seed$SEED
mkdir -p "$OUT"
# Ports from the job id, so two jobs sharing a node don't collide.
JOB=${SLURM_JOB_ID:-$$}
PORT=$((8000 + JOB % 1000)); GROUP_PORT=$((51000 + JOB % 1000))
OVERRIDES=("$@" "common.vllm_server_base_url=http://localhost:$PORT")
[[ $TRAINER == sync ]] && OVERRIDES+=("sync.vllm_group_port=$GROUP_PORT")

# Use the GPUs Slurm assigned (first = vLLM, second = trainer), never hardcoded indices.
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if (( ${#GPUS[@]} < 2 )); then echo "need 2 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2; exit 1; fi
VLLM_GPU=${GPUS[0]}; TRAIN_GPU=${GPUS[1]}
# PCI bus ids let the trainer's NVML sampler label the two GPUs by role (logging_utils.GPUSampler).
export AGRPO_VLLM_BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$VLLM_GPU")
export AGRPO_TRAINER_BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$TRAIN_GPU")

JOB_START=$(date +%s)
export RUN_LAUNCH_TIME=$(python -c 'import time; print(time.time())')
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv > "$OUT/gpus.csv"

CUDA_VISIBLE_DEVICES=$VLLM_GPU VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAXLEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --seed "$SEED" \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend": "nccl"}' \
    > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# The trainer waits for the server itself (vllm_server_timeout); startup time is logged in metrics.jsonl.
CUDA_VISIBLE_DEVICES=$TRAIN_GPU python -m "src.train_$TRAINER" --config "$CONFIG" --seed "$SEED" --output_dir "$OUT" \
    "${OVERRIDES[@]}" 2>&1 | tee "$OUT/train.log"
TRAIN_END=$(date +%s)

kill $VLLM_PID; wait $VLLM_PID 2>/dev/null || true
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
    CUDA_VISIBLE_DEVICES=$VLLM_GPU python -m src.eval --run_dir "$OUT" --delete_after 2>&1 | tee "$OUT/eval.log"
fi
JOB_END=$(date +%s)
# Whole job on 2 GPUs (startup + train + eval): used for the compute total in the README.
printf '{"start": %s, "train_end": %s, "end": %s, "n_gpus": 2, "job_id": "%s"}\n' \
    "$JOB_START" "$TRAIN_END" "$JOB_END" "$JOB" > "$OUT/job.json"
