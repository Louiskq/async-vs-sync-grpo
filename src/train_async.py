"""Async GRPO on GSM8K with TRL's AsyncGRPOTrainer (generation on a separate vLLM server, overlapped with training).

    python -m src.train_async --config configs/async-s4.yaml --seed 0 [common.max_steps=25 ...]

Expects a vLLM server already running (see slurm/_run.sh for the exact command).
"""

from __future__ import annotations

from trl import AsyncGRPOConfig, AsyncGRPOTrainer

from src.attention import install_async_kernel
from src.common import launch_time, parse_args, run_info
from src.config import trainer_kwargs
from src.controller import ControllerConfig, RuleController
from src.data import load_train
from src.logging_utils import StepStats, UnifiedMetricsCallback
from src.reward import gsm8k_reward


class InstrumentedAsyncGRPOTrainer(AsyncGRPOTrainer):
    """Adds the per-token log-ratio distribution (log pi_current - log pi_vLLM) that TRL only logs as mean/max.

    A forward hook captures the model output's ``log_probs`` inside TRL's ``compute_loss``; the token selection
    below mirrors the first lines of that method (TRL 1.18.2) so no loss code is copied or changed.
    """

    step_stats: StepStats

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        captured = {}

        def hook(_module, _args, _kwargs, output):
            captured["log_probs"] = output.log_probs.detach()

        handle = model.register_forward_hook(hook, with_kwargs=True)
        try:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            handle.remove()
        if "log_probs" in captured:
            selected = inputs["token_index"] >= 0
            current = captured["log_probs"].flatten()[inputs["token_index"][selected]]
            self.step_stats.logratio.add(current - inputs["old_log_probs"][selected])
        return loss


def main() -> None:
    cfg, cli = parse_args(__doc__)
    if cfg["trainer"] != "async":
        raise SystemExit(f"{cfg['name']} is a {cfg['trainer']!r} config; use src.train_{cfg['trainer']}")
    install_async_kernel(cfg["attention_kernel"])  # before training: fail fast if the kernel has no build here
    kwargs = trainer_kwargs(cfg, AsyncGRPOConfig, "async")
    args = AsyncGRPOConfig(output_dir=cli.output_dir, seed=cfg["seed"], **kwargs)
    train = load_train(seed=cfg["seed"], limit=cfg.get("data", {}).get("train_limit"))

    trainer = InstrumentedAsyncGRPOTrainer(
        model=cfg["model"], reward_funcs=gsm8k_reward, args=args, train_dataset=train
    )
    trainer.step_stats = StepStats()

    controller = None
    if cfg.get("controller"):
        ctrl_cfg = ControllerConfig(**cfg["controller"])
        initial = args.max_staleness if ctrl_cfg.knob == "max_staleness" else args.learning_rate
        controller = RuleController(ctrl_cfg, initial_value=initial)

    callback = UnifiedMetricsCallback(
        kind="async",
        output_dir=cli.output_dir,
        stats=trainer.step_stats,
        run_info={**run_info(cfg, "async"), "controller": cfg.get("controller")},
        is_clip_min=args.vllm_importance_sampling_clip_min,
        is_clip_max=args.vllm_importance_sampling_clip_max,
        controller=controller,
        launch_time=launch_time(),
    ).bind(trainer)
    trainer.add_callback(callback)
    trainer.train()


if __name__ == "__main__":  # required: the rollout worker is a spawned process that re-imports this module
    main()
