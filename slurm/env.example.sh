# Copy to slurm/env.sh (gitignored) and fill in for your cluster. Sourced by every job.
# Also export SBATCH_ACCOUNT=<your slurm account> (and SBATCH_PARTITION if needed) in your shell: sbatch reads them.

export VENV=$HOME/venv-agrpo                       # built by slurm/setup_env.sh
export SCRATCH_DIR=/scratch/<project>/<user>
export HF_HOME=$SCRATCH_DIR/hf_cache
export RUNS_DIR=$SCRATCH_DIR/agrpo-runs            # checkpoints are large; keep runs on scratch

# Isambard-AI only: vLLM >= 0.23 wheels are built against CUDA 13, nodes run a CUDA 12.7 driver.
# See docs/setup.md. Leave CUDA_COMPAT_DIR empty on a machine whose driver supports CUDA 13.
export CUDA_COMPAT_DIR=$SCRATCH_DIR/cuda-compat/usr/local/cuda-13.0/compat
