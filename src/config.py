"""Run configs: YAML with single inheritance (``base: other.yaml``) and deep merge.

Layout of a resolved config::

    name, trainer (sync|async), model, max_model_len, seed
    data:        {train_limit, eval_size, eval_seed}
    common:      kwargs for BOTH GRPOConfig and AsyncGRPOConfig (the held-identical hyperparameters)
    sync:        kwargs only for GRPOConfig
    async:       kwargs only for AsyncGRPOConfig
    controller:  ControllerConfig kwargs, or null

Every key under ``common``/``sync``/``async`` must be a field of the target TRL config class: unknown keys raise
instead of being dropped, so a typo can't silently make two runs differ.

CLI: ``python -m src.config --config configs/x.yaml --get model`` prints one resolved value (used by slurm).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    return deep_merge(_load(path.parent / base), cfg) if base is not None else cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Resolve inheritance. ``name`` defaults to the file stem of the file loaded, not of its bases."""
    path = Path(path)
    cfg = _load(path)
    with open(path) as f:
        own = yaml.safe_load(f) or {}
    cfg["name"] = own.get("name", path.stem)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """``a.b=value`` overrides from the command line; values are parsed as YAML (so 25 -> int, null -> None)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                node[p] = {}  # e.g. overriding controller.high when the base has controller: null
            node = node[p]
        node[leaf] = yaml.safe_load(raw)
    return cfg


def trainer_kwargs(cfg: dict, config_cls: type, section: str) -> dict[str, Any]:
    """Merge ``common`` + ``section`` and check every key is a field of ``config_cls``."""
    kwargs = {**(cfg.get("common") or {}), **(cfg.get(section) or {})}
    fields = {f.name for f in dataclasses.fields(config_cls)}
    unknown = sorted(set(kwargs) - fields)
    if unknown:
        raise ValueError(f"{config_cls.__name__} has no field(s) {unknown} (from config sections common/{section})")
    return kwargs


def get_path(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--get", required=True, help="dotted key, e.g. model or async.max_staleness")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    print(get_path(apply_overrides(load_config(a.config), a.overrides), a.get))


if __name__ == "__main__":
    main()
