"""GSM8K answer extraction and exact-match reward.

The reward is correctness only: 1.0 if the extracted final answer equals the gold answer numerically, else 0.0.
Extraction tries, in order, the last ``#### <n>`` marker, the last ``\\boxed{...}``, the last "answer is <n>" /
"answer: <n>" phrase, and finally the last number in the text. Numbers are compared as Decimals, so "18", "18.0",
"$18" and "18.00" all match a gold "18".
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# A number: optional minus (not when it is a hyphen after a word or digit, e.g. "3-4" or "x-1"), then either
# comma-grouped thousands ("1,234,567") or plain digits, then an optional decimal part. A leading "$" is allowed
# and ignored.
_NUM = r"(?<![\w.])-?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
_HASH_RE = re.compile(r"####\s*\$?\s*(" + _NUM.replace(r"(?<![\w.])", "") + ")")
_ANSWER_PHRASE_RE = re.compile(
    r"answer(?:\s+is)?\s*[:=]?\s*\**\s*\$?\s*(" + _NUM.replace(r"(?<![\w.])", "") + ")", re.IGNORECASE
)
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def normalize_number(s: str) -> Decimal | None:
    """Parse a number string like "-1,234.50" or "$18" into a Decimal, or None if it isn't one."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _boxed_contents(text: str) -> list[str]:
    """Return the contents of every ``\\boxed{...}``, matching nested braces."""
    out = []
    for m in _BOXED_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append(text[m.end() : i - 1])
    return out


def extract_answer(text: str) -> Decimal | None:
    """Extract the final numeric answer from a model completion, or None if there is no number."""
    if not text:
        return None
    hashes = _HASH_RE.findall(text)
    if hashes:
        return normalize_number(hashes[-1])
    for box in reversed(_boxed_contents(text)):
        nums = _NUM_RE.findall(box)
        if nums:
            return normalize_number(nums[-1])
    phrases = _ANSWER_PHRASE_RE.findall(text)
    if phrases:
        return normalize_number(phrases[-1])
    nums = _NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])
    return None


def extract_gold(answer_field: str) -> str:
    """Gold answer from a GSM8K ``answer`` field ("...reasoning... #### 72") as a canonical number string."""
    gold = answer_field.split("####")[-1]
    value = normalize_number(gold)
    if value is None:
        raise ValueError(f"GSM8K answer field has no number after '####': {answer_field!r}")
    return format(value.normalize(), "f")


def is_correct(completion_text: str, gold: str) -> bool:
    pred = extract_answer(completion_text)
    gold_value = normalize_number(gold)
    return pred is not None and gold_value is not None and pred == gold_value


def completion_text(completion) -> str:
    """Text of one completion in any format TRL passes: a string, a message dict, or a list of messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content") or ""
    if isinstance(completion, list):
        return "".join(completion_text(m) for m in completion if not isinstance(m, dict) or m.get("role") != "user")
    raise TypeError(f"Unsupported completion type: {type(completion).__name__}")


def gsm8k_reward(completions: list, gold: list[str], **kwargs) -> list[float]:
    """TRL reward function: 1.0 for a correct final answer, else 0.0.

    Module-level and stateless so it pickles into AsyncGRPOTrainer's spawned rollout process. ``gold`` is the
    dataset column of that name (see ``data.py``).
    """
    return [1.0 if is_correct(completion_text(c), g) else 0.0 for c, g in zip(completions, gold, strict=True)]
