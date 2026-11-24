# Environment setup

Target: Isambard-AI (GH200, aarch64), Slurm. Any Linux node with ≥ 2 NVIDIA GPUs and a CUDA 13-capable driver also
works: leave `CUDA_COMPAT_DIR` empty there.

## Versions

| Package | Version | Why |
|---|---|---|
| trl | 1.18.2 | AsyncGRPOTrainer in the stable API; all notes in `docs/notes.md` are against this version |
| vllm | 0.26.0 | inside TRL 1.18.2's supported range (≥ 0.23, < 0.34); already verified on Isambard-AI |
| torch, transformers | whatever vLLM 0.26.0 pins | installed by vLLM, recorded in `requirements.lock` |

The SPEC's install-order workaround (`transformers --no-deps`) was for TRL 1.3. With TRL 1.18.2 a plain install
resolves: vLLM first (it pins torch), then TRL. `requirements.lock` and `environment.md` are generated on the cluster
by `slurm/setup_env.sh`: they are the source of truth for what ran.

**The ARM risk.** vLLM publishes `manylinux_2_28_aarch64` wheels on PyPI for every release from 0.25 onwards (checked
2026-11-23), so nothing needs to be built from source. The remaining risk is the driver: see below.

## Isambard-AI: CUDA 13 wheels and the driver

**Update 2026-11-24: the nodes now run driver 580.196.07, which supports CUDA 13 natively.** The forward-compat
libraries (fix 1 below) are now *harmful*: with them on `LD_LIBRARY_PATH`, CUDA init fails with error 803
(`system has unsupported display driver / cuda driver combination`). This was verified on a GH200 node, with and
without them. `slurm/_common.sh` therefore applies them only when the driver major version is < 580.

Before that, the nodes ran driver 565.57 (CUDA 12.7). vLLM ≥ 0.23 is built against CUDA 13.0, so it failed with
`The NVIDIA driver on your system is too old (found version 12070)` unless you applied four userspace fixes. All four
are in `slurm/setup_env.sh` and `slurm/_common.sh`; fixes 2–4 still apply to anything that JIT-compiles:

1. **Forward-compat libraries.** Extract `cuda-compat-13-0-580.95.05-1.aarch64.rpm` (NVIDIA's sbsa repo) to scratch,
   and prepend its `compat/` directory to `LD_LIBRARY_PATH` in every job.
2. **`CUDA_HOME` → the pip CUDA 13 toolkit** (`site-packages/nvidia/cu13`), with `nvcc` and the venv's `ninja` on
   `PATH`, for anything vLLM/flashinfer JIT-compiles.
3. **Pin nvcc, crt and nvvm together to 13.0.88.** pip otherwise assembles a mixed 13.4/13.0 toolchain that breaks JIT
   builds (`CUDA compiler and CUDA toolkit headers are incompatible`, then `Unsupported .version 9.4`).
4. **Symlinks** `lib64 → lib`, `libcudart.so`, and `lib/stubs/libcuda.so`, which the pip wheels don't ship.

## Steps

```bash
git clone <repo> && cd async-grpo-hpcontrol
cp slurm/env.example.sh slurm/env.sh      # fill in VENV, SCRATCH_DIR, HF_HOME, RUNS_DIR, CUDA_COMPAT_DIR
export SBATCH_ACCOUNT=<account>           # sbatch reads it; add SBATCH_PARTITION if your cluster needs one
mkdir -p logs
sbatch slurm/setup_env.sh                 # on a compute node: builds the venv, writes requirements.lock
sbatch slurm/smoke.sh                     # M0: 20 async + 20 sync steps on Qwen3-0.6B
```

**Python:** build the venv from the Cray PE Python (`/opt/cray/pe/python/3.11.7`). The system `/usr/bin/python3.11`
lacks the `sqlite3` module, which vLLM imports; `setup_env.sh` checks for this before installing.

Pre-download the models on a compute node rather than the login node, which has a 4 GiB memory cap:

```bash
srun --gres=gpu:1 --time=00:30:00 bash -c 'source slurm/env.sh; $VENV/bin/hf download Qwen/Qwen3-1.7B; $VENV/bin/hf download Qwen/Qwen3-0.6B'
```

## Smoke test result (M0)

_Not yet run._ The first `slurm/smoke.sh` output goes here: steps completed, s/step, and whether weight sync worked.
