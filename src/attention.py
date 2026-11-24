"""One FlashAttention kernel for both trainers, chosen by the config key ``attention_kernel``.

AsyncGRPOTrainer's packed "tree attention" calls ``trl.trainer.async_grpo_trainer._flash_attn()``, which loads
``kernels-community/flash-attn3`` from the Hugging Face kernel hub. That repo has no build for aarch64 with
torch 2.10+/CUDA 13 (checked 2026-11-24), which is what vLLM 0.26 pins on Isambard-AI, so training dies at the first
forward with no kernel. ``kernels-community/flash-attn2`` has that build, and its ``flash_attn_varlen_func`` takes
the same arguments TRL passes (``softmax_scale``, ``causal``, ``window_size``, ``softcap``). FA2 also aligns the
causal mask bottom-right when a segment has fewer queries than keys, which tree attention relies on
(``scripts/check_tree_attention.py`` verifies this numerically on the GPU).

For a fair comparison the sync trainer is given the same kernel through transformers' hub-kernel attention
(``attn_implementation="<repo>@<revision>"``).
"""

from __future__ import annotations

KERNELS = {
    "fa2": ("kernels-community/flash-attn2", 1),
    "fa3": ("kernels-community/flash-attn3", 1),
}


def kernel_spec(name: str) -> tuple[str, int]:
    if name not in KERNELS:
        raise ValueError(f"attention_kernel must be one of {sorted(KERNELS)}, got {name!r}")
    return KERNELS[name]


def hf_attn_implementation(name: str) -> str:
    """The transformers ``attn_implementation`` string for the same kernel (repo pinned to the same version)."""
    repo, version = kernel_spec(name)
    return f"{repo}@v{version}"


def load_kernel(name: str):
    from kernels import get_kernel

    repo, version = kernel_spec(name)
    return get_kernel(repo, version=version)


def install_async_kernel(name: str):
    """Make AsyncGRPOTrainer's tree attention use ``name``. Loads eagerly so a missing build fails at startup.

    ``tree_attention`` looks ``_flash_attn`` up as a module global on every call, so replacing the module attribute
    is enough; nothing else in TRL 1.18.2 references it.
    """
    import trl.trainer.async_grpo_trainer as agt

    kernel = load_kernel(name)
    agt._flash_attn = lambda: kernel
    return kernel
