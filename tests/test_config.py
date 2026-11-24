from pathlib import Path

import pytest

from src.config import apply_overrides, load_config, trainer_kwargs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RUNS = sorted(p for p in CONFIGS.glob("*.yaml") if p.name != "base.yaml")


def test_name_is_the_loaded_file_not_its_base():
    assert load_config(CONFIGS / "async-s4.yaml")["name"] == "async-s4"
    assert load_config(CONFIGS / "smoke-async.yaml")["name"] == "smoke-async"


def test_inheritance_deep_merges():
    cfg = load_config(CONFIGS / "smoke-async.yaml")
    assert cfg["model"] == "Qwen/Qwen3-0.6B"
    assert cfg["async"]["max_staleness"] == 4  # from async-s4.yaml
    assert cfg["async"]["score_centering"] is False  # from base.yaml
    assert cfg["common"]["max_steps"] == 20


def test_overrides():
    cfg = apply_overrides(load_config(CONFIGS / "sync.yaml"), ["common.max_steps=25", "controller.high=null"])
    assert cfg["common"]["max_steps"] == 25 and cfg["controller"]["high"] is None
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["no_equals_sign"])


def test_unknown_key_is_rejected():
    from trl import GRPOConfig

    with pytest.raises(ValueError, match="max_staleness"):
        trainer_kwargs({"common": {"max_staleness": 1}}, GRPOConfig, "sync")


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.stem)
def test_every_run_config_builds_its_trl_config(path, tmp_path):
    from trl import AsyncGRPOConfig, GRPOConfig

    cfg = load_config(path)
    cls = {"sync": GRPOConfig, "async": AsyncGRPOConfig}[cfg["trainer"]]
    kwargs = trainer_kwargs(cfg, cls, cfg["trainer"])
    kwargs["bf16"] = False  # CPU test machine; everything else is constructed as on the cluster
    args = cls(output_dir=str(tmp_path), **kwargs)
    assert args.max_completion_length == cfg["common"]["max_completion_length"]
    assert args.max_steps % args.save_steps == 0 or args.save_strategy == "no"


def test_common_keys_are_valid_for_both_trainers():
    from trl import AsyncGRPOConfig, GRPOConfig

    common = load_config(CONFIGS / "base.yaml")["common"]
    trainer_kwargs({"common": common}, GRPOConfig, "sync")
    trainer_kwargs({"common": common}, AsyncGRPOConfig, "async")


def test_core_runs_differ_only_in_the_variable_under_study():
    s1, s4 = (load_config(CONFIGS / f"async-s{n}.yaml") for n in (1, 4))
    sync = load_config(CONFIGS / "sync.yaml")
    assert s1["common"] == s4["common"] == sync["common"]
    assert {k for k in s1["async"] if s1["async"][k] != s4["async"][k]} == {"max_staleness"}
    assert s1["model"] == s4["model"] == sync["model"]


def test_sync_batch_is_32_prompts_by_8_generations():
    sync = load_config(CONFIGS / "sync.yaml")
    s = sync["sync"]
    assert s["per_device_train_batch_size"] * s["gradient_accumulation_steps"] == 256
    assert sync["common"]["num_generations"] == 8


def test_controller_config_requires_pilot_thresholds():
    from src.controller import ControllerConfig

    ctrl = load_config(CONFIGS / "async-ctrl.yaml")["controller"]
    assert ctrl["high"] is None and ctrl["low"] is None
    with pytest.raises(TypeError):
        ControllerConfig(**ctrl)


def test_slurm_overrides_parse_to_the_intended_types():
    """Every dotted override in the slurm scripts must parse to what TrainingArguments expects (YAML: no -> False)."""
    import re
    import shlex

    from trl import AsyncGRPOConfig, GRPOConfig

    slurm = CONFIGS.parent / "slurm"
    for script in slurm.glob("*.sh"):
        for line in script.read_text().splitlines():
            if not line.startswith("bash slurm/_run.sh"):
                continue
            words = shlex.split(re.sub(r"\$\{?STEPS\}?(:-25)?", "25", line))
            config, overrides = words[2], [w for w in words[4:] if "=" in w]
            cfg = apply_overrides(load_config(CONFIGS.parent / config), overrides)
            cls = {"sync": GRPOConfig, "async": AsyncGRPOConfig}[cfg["trainer"]]
            kwargs = {**trainer_kwargs(cfg, cls, cfg["trainer"]), "bf16": False}
            cls(output_dir="/tmp/unused", **kwargs)  # raises on save_strategy=False
            assert cfg["name"].startswith(("pilot-", "smoke-")) or cfg["name"] == config.split("/")[-1][:-5]


def test_attention_kernel_is_shared_and_pinned():
    from src.attention import hf_attn_implementation, kernel_spec

    for path in RUNS:
        assert load_config(path)["attention_kernel"] == "fa2", path.stem
    assert hf_attn_implementation("fa2") == "kernels-community/flash-attn2@v1"
    assert kernel_spec("fa2") == ("kernels-community/flash-attn2", 1)
    with pytest.raises(ValueError):
        kernel_spec("sdpa")


def test_install_async_kernel_replaces_trl_loader(monkeypatch):
    import sys
    import types

    import trl.trainer.async_grpo_trainer as agt

    from src.attention import install_async_kernel

    loaded = []
    fake_kernel = object()
    monkeypatch.setitem(
        sys.modules,
        "kernels",
        types.SimpleNamespace(get_kernel=lambda repo, version: loaded.append((repo, version)) or fake_kernel),
    )
    monkeypatch.setattr(agt, "_flash_attn", agt._flash_attn)  # restored after the test
    install_async_kernel("fa2")
    assert loaded == [("kernels-community/flash-attn2", 1)]
    assert agt._flash_attn() is fake_kernel
