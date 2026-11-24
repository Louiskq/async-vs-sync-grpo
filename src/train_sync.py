"""Synchronous GRPO baseline on GSM8K: TRL's GRPOTrainer in vLLM server mode, same 2-GPU split as async.

    python -m src.train_sync --config configs/sync.yaml --seed 0 [common.max_steps=25 ...]

Generation and training alternate: the trainer GPU waits while the vLLM GPU generates, and vice versa.
Expects the same vLLM server as the async runs (see slurm/_run.sh).
"""

from __future__ import annotations

import time

from trl import GRPOConfig, GRPOTrainer

from src.attention import hf_attn_implementation
from src.common import launch_time, parse_args, run_info
from src.config import trainer_kwargs
from src.data import load_train
from src.logging_utils import StepStats, UnifiedMetricsCallback
from src.reward import gsm8k_reward


class InstrumentedGRPOTrainer(GRPOTrainer):
    """Times generation and records the trainer-vs-vLLM log-ratio, read from the batch TRL already builds.

    With vLLM IS correction on, ``_generate_and_score_completions`` returns ``old_per_token_logps`` (the trainer's
    logprobs at generation time; the current policy, since sync is on-policy) and ``sampling_per_token_logps``
    (vLLM's). Their difference is the same quantity the async trainer logs, so the two runs are comparable: for
    sync it measures only the trainer/inference numerical mismatch (staleness 0).
    """

    step_stats: StepStats

    def _generate(self, prompts):
        t0 = time.perf_counter()
        out = super()._generate(prompts)
        self.step_stats.gen_s += time.perf_counter() - t0  # includes the weight push to vLLM before generating
        return out

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        mask = out["completion_mask"].bool()
        if out.get("old_per_token_logps") is not None and out.get("sampling_per_token_logps") is not None:
            self.step_stats.logratio.add((out["old_per_token_logps"] - out["sampling_per_token_logps"])[mask])
        self.step_stats.completion_tokens += float(mask.sum())
        self.step_stats.samples += int(mask.shape[0])
        return out


def main() -> None:
    cfg, cli = parse_args(__doc__)
    if cfg["trainer"] != "sync":
        raise SystemExit(f"{cfg['name']} is a {cfg['trainer']!r} config; use src.train_{cfg['trainer']}")
    if cfg.get("controller"):
        raise SystemExit("the controller is only defined for async runs")
    kwargs = trainer_kwargs(cfg, GRPOConfig, "sync")
    # Same FlashAttention kernel as the async trainer's tree attention (src/attention.py).
    kwargs["model_init_kwargs"] = {
        **(kwargs.get("model_init_kwargs") or {}),
        "attn_implementation": hf_attn_implementation(cfg["attention_kernel"]),
    }
    args = GRPOConfig(output_dir=cli.output_dir, seed=cfg["seed"], **kwargs)
    if args.shuffle_dataset:
        raise SystemExit("sync must read prompts in order (shuffle_dataset: false) to match the async data order")
    train = load_train(seed=cfg["seed"], limit=cfg.get("data", {}).get("train_limit"))

    trainer = InstrumentedGRPOTrainer(model=cfg["model"], reward_funcs=gsm8k_reward, args=args, train_dataset=train)
    trainer.step_stats = StepStats()
    callback = UnifiedMetricsCallback(
        kind="sync",
        output_dir=cli.output_dir,
        stats=trainer.step_stats,
        run_info=run_info(cfg, "sync"),
        is_clip_min=args.vllm_importance_sampling_clip_min,
        is_clip_max=args.vllm_importance_sampling_clip_max,
        launch_time=launch_time(),
    ).bind(trainer)
    trainer.add_callback(callback)
    trainer.train()


if __name__ == "__main__":
    main()
