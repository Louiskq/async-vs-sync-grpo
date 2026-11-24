"""Held-out GSM8K eval of a run's checkpoints, after training, so eval never competes with training wall-clock.

    python -m src.eval --run_dir runs/async-s4-seed0 [--delete_after]

Greedy decoding with offline vLLM on the fixed 500-problem test subset, same prompt format and chat template as
training. Writes one row per checkpoint to ``<run_dir>/eval.jsonl``: step, accuracy, and the training wall-clock at
that step (read from metrics.jsonl), so accuracy can be plotted against wall-clock. Step 0 (the base model) is
evaluated once per model and cached under ``runs/_base_eval/``. Already-evaluated steps are skipped, so the script
can be re-run after an interruption. ``--delete_after`` removes each checkpoint once its result is written
(fp32 checkpoints of a 1.7B model are ~7 GB each).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from pathlib import Path

from src.data import load_eval
from src.logging_utils import JSONLWriter, read_jsonl
from src.reward import completion_text, is_correct


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def wall_at_step(run_dir: Path) -> dict[int, float]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    return {r["step"]: r["wall_s"] for r in read_jsonl(path) if "event" not in r}


def evaluate(model_path: str, tokenizer: str, cfg: dict, seed: int = 0) -> dict:
    from vllm import LLM, SamplingParams

    data = cfg.get("data", {})
    ds = load_eval(size=data.get("eval_size", 500), seed=data.get("eval_seed", 1234))
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer,
        dtype="bfloat16",
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.85,
        seed=seed,
    )
    params = SamplingParams(temperature=0.0, max_tokens=cfg["common"]["max_completion_length"])
    outputs = llm.chat(list(ds["prompt"]), params, chat_template_kwargs=cfg["common"].get("chat_template_kwargs") or {})
    texts = [o.outputs[0].text for o in outputs]
    correct = [is_correct(completion_text(t), g) for t, g in zip(texts, ds["gold"], strict=True)]
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return {
        "accuracy": sum(correct) / len(correct),
        "n": len(correct),
        "mean_completion_chars": sum(map(len, texts)) / len(texts),
        "examples": [
            {"completion": t, "gold": g, "correct": c} for t, g, c in list(zip(texts, ds["gold"], correct))[:5]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--delete_after", action="store_true")
    p.add_argument("--skip_base", action="store_true")
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    out_path = run_dir / "eval.jsonl"
    done = {r["step"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    writer = JSONLWriter(out_path)
    walls = wall_at_step(run_dir)

    if 0 not in done and not a.skip_base:
        cache = run_dir.parent / "_base_eval" / (cfg["model"].replace("/", "__") + ".json")
        if cache.exists():
            base = json.loads(cache.read_text())
        else:
            base = evaluate(cfg["model"], cfg["model"], cfg)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(base))
        writer.write({"step": 0, "wall_s": 0.0, "checkpoint": cfg["model"], **base})

    for step, path in checkpoints(run_dir):
        if step in done:
            continue
        result = evaluate(str(path), cfg["model"], cfg)
        writer.write({"step": step, "wall_s": walls.get(step), "checkpoint": path.name, **result})
        print(f"step {step}: accuracy {result['accuracy']:.3f}", flush=True)
        if a.delete_after:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()
