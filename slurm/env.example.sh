# Copy to slurm/env.sh (gitignored) and fill in for your cluster. Sourced by every job.
# Also export SBATCH_ACCOUNT=<your slurm account> (and SBATCH_PARTITION if needed) in your shell: sbatch reads them.

export VENV=$HOME/venv-agrpo                       # built by slurm/setup_env.sh
export SCRATCH_DIR=/scratch/<project>/<user>
export HF_HOME=$SCRATCH_DIR/hf_cache
export RUNS_DIR=$SCRATCH_DIR/agrpo-runs            # checkpoints are large; keep runs on scratch

# CUDA 13 forward-compat libs for drivers older than 580 (vLLM >= 0.23 wheels are built against CUDA 13).
# Applied automatically only when the node's driver is < 580; on newer drivers they would break CUDA. See docs/setup.md.
export CUDA_COMPAT_DIR=$SCRATCH_DIR/cuda-compat/usr/local/cuda-13.0/compat
