# Sourced by job scripts: environment for vLLM + TRL on the node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f slurm/env.sh ]]; then echo "missing slurm/env.sh (copy slurm/env.example.sh)" >&2; exit 1; fi
source slurm/env.sh
# CUDA forward-compat libs only on drivers older than 580 (the first with CUDA 13). On a newer driver they
# break CUDA init with error 803 ("unsupported display driver / cuda driver combination").
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [[ -n "${CUDA_COMPAT_DIR:-}" && -n "$DRIVER_MAJOR" && "$DRIVER_MAJOR" -lt 580 ]]; then
    export LD_LIBRARY_PATH=$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}
fi
PYVER=$("$VENV/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
CU13=$VENV/lib/$PYVER/site-packages/nvidia/cu13
if [[ -d $CU13 ]]; then export CUDA_HOME=$CU13; export PATH=$CUDA_HOME/bin:$PATH; fi
export PATH=$VENV/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA indices == nvidia-smi/NVML indices
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p logs "$RUNS_DIR"
