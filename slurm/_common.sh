# Sourced by job scripts: environment for vLLM + TRL on the node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f slurm/env.sh ]]; then echo "missing slurm/env.sh (copy slurm/env.example.sh)" >&2; exit 1; fi
source slurm/env.sh
if [[ -n "${CUDA_COMPAT_DIR:-}" ]]; then export LD_LIBRARY_PATH=$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}; fi
PYVER=$("$VENV/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
CU13=$VENV/lib/$PYVER/site-packages/nvidia/cu13
if [[ -d $CU13 ]]; then export CUDA_HOME=$CU13; export PATH=$CUDA_HOME/bin:$PATH; fi
export PATH=$VENV/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA indices == nvidia-smi/NVML indices
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p logs "$RUNS_DIR"
