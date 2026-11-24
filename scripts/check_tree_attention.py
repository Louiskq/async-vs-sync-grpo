"""GPU check: the configured flash-attention kernel gives TRL's tree attention the semantics it assumes.

    python scripts/check_tree_attention.py [fa2|fa3]

Tree attention calls ``flash_attn_varlen_func`` with segments that have FEWER queries than keys (a segment's own
tokens attend to its ancestors' K/V plus their own causal past), relying on a bottom-right-aligned causal mask.
This compares the kernel against a plain PyTorch reference with that mask, with GQA (Qwen3-1.7B: 16 query heads,
8 KV heads, head_dim 128) in bf16. Exits non-zero on mismatch. Runs in seconds; slurm/smoke.sh calls it first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attention import load_kernel  # noqa: E402


def reference(q, k, v, cu_q, cu_k, scale):
    """Per segment: query i (of n_q) sees keys 0 .. n_k - n_q + i (bottom-right causal)."""
    out = torch.empty_like(q)
    groups = q.shape[1] // k.shape[1]
    for s in range(len(cu_q) - 1):
        qs, ks = q[cu_q[s] : cu_q[s + 1]], k[cu_k[s] : cu_k[s + 1]]
        vs = v[cu_k[s] : cu_k[s + 1]]
        n_q, n_k = qs.shape[0], ks.shape[0]
        ks, vs = ks.repeat_interleave(groups, dim=1), vs.repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs.float(), ks.float()) * scale
        mask = torch.arange(n_k, device=q.device)[None, :] <= (torch.arange(n_q, device=q.device)[:, None] + n_k - n_q)
        scores = scores.masked_fill(~mask, float("-inf"))
        out[cu_q[s] : cu_q[s + 1]] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs.float()).to(q.dtype)
    return out


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "fa2"
    kernel = load_kernel(name)
    torch.manual_seed(0)
    dev, dt, hq, hk, d = "cuda", torch.bfloat16, 16, 8, 128
    q_lens, k_lens = [7, 120, 1, 300], [57, 120, 40, 812]  # fewer queries than keys, equal, single query, long
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    q = torch.randn(sum(q_lens), hq, d, device=dev, dtype=dt)
    k = torch.randn(sum(k_lens), hk, d, device=dev, dtype=dt)
    v = torch.randn(sum(k_lens), hk, d, device=dev, dtype=dt)
    scale = d**-0.5
    out = kernel.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
    )
    out = out[0] if isinstance(out, tuple) else out
    ref = reference(q, k, v, cu_q.tolist(), cu_k.tolist(), scale)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"{name}: max |kernel - reference| = {err:.4g} on {torch.cuda.get_device_name()}")
    if not err < 2e-2:
        raise SystemExit(f"{name} kernel does not match bottom-right causal varlen attention")


if __name__ == "__main__":
    main()
