"""GSM8K loading and prompt formatting, shared by training (sync and async) and eval."""

from __future__ import annotations

from datasets import Dataset, load_dataset

from src.reward import extract_gold

INSTRUCTION = (
    "Solve the following math problem. Reason step by step, then give the final answer "
    "on its own line in the form '#### <number>'."
)


def format_prompt(question: str) -> list[dict[str, str]]:
    """Conversational prompt for one question (TRL's chat format; the chat template is applied by the trainer)."""
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{question.strip()}"}]


def _to_rows(ds: Dataset) -> Dataset:
    return ds.map(
        lambda ex: {"prompt": format_prompt(ex["question"]), "gold": extract_gold(ex["answer"])},
        remove_columns=ds.column_names,
    )


def load_train(seed: int, limit: int | None = None) -> Dataset:
    """GSM8K train split, shuffled once with ``seed``.

    The trainers must then read it in order (sync: ``shuffle_dataset=False``; async always iterates in order), so
    every config sees the same prompt order for a given seed.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return _to_rows(ds)


def load_eval(size: int = 500, seed: int = 1234) -> Dataset:
    """Fixed held-out subset of the GSM8K test split (same subset for every run)."""
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    return _to_rows(ds.select(range(min(size, len(ds)))))
