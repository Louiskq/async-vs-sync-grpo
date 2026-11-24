"""Shared entry-point plumbing for train_sync.py and train_async.py."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

from src.config import apply_overrides, load_config


def parse_args(description: str) -> tuple[dict[str, Any], argparse.Namespace]:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    p.add_argument("--output_dir", default=None, help="default: runs/<name>-seed<seed>")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. common.max_steps=25")
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), a.overrides)
    if a.seed is not None:
        cfg["seed"] = a.seed
    cfg.setdefault("seed", 0)
    a.output_dir = a.output_dir or f"runs/{cfg['name']}-seed{cfg['seed']}"
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(a.output_dir) / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg, a


def launch_time() -> float | None:
    """Job launch time exported by slurm/_run.sh, so startup (vLLM + model load) is measured separately."""
    t = os.environ.get("RUN_LAUNCH_TIME")
    return float(t) if t else None


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "trl", "vllm", "accelerate", "datasets", "kernels"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def run_info(cfg: dict, kind: str) -> dict[str, Any]:
    return {
        "name": cfg["name"],
        "kind": kind,
        "model": cfg["model"],
        "seed": cfg["seed"],
        "attention_kernel": cfg.get("attention_kernel"),
        "versions": versions(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
