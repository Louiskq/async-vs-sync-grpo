from decimal import Decimal

import pytest

from src.reward import completion_text, extract_answer, extract_gold, gsm8k_reward, is_correct


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so she has 18 eggs.\n#### 18", "18"),
        ("#### 1,234", "1234"),
        ("#### $18.00", "18"),
        ("####-5", "-5"),
        ("first #### 3 then corrected\n#### 4", "4"),  # last marker wins
        ("The answer is \\boxed{72}.", "72"),
        ("\\boxed{\\$1,000}", "1000"),
        ("\\boxed{\\text{72 apples}}", "72"),  # nested braces
        ("The answer is 42.", "42"),
        ("Answer: **7**", "7"),
        ("She pays 3 + 4 = 7 dollars", "7"),  # last number fallback
        ("between 3-4 hours", "4"),  # hyphen is not a minus sign
        ("It costs -3 degrees", "-3"),
        ("a 50% discount", "50"),
        ("0.5 of 10 is 5.", "5"),  # trailing sentence period is not a decimal point
        ("1, 2, 3", "3"),  # list commas are not thousands separators
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == Decimal(expected)


def test_marker_beats_later_numbers():
    assert extract_answer("#### 10\nCheck: 10 - 2 = 8") == Decimal(10)


def test_boxed_beats_last_number():
    assert extract_answer("\\boxed{12} (that is 3 times 4)") == Decimal(12)


@pytest.mark.parametrize("text", ["", "no numbers here", "#### "])
def test_extract_answer_none(text):
    assert extract_answer(text) is None


@pytest.mark.parametrize(
    ("field", "gold"),
    [
        ("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72", "72"),
        ("... #### 1,000", "1000"),
        ("... #### -3", "-3"),
        ("... #### 2.50", "2.5"),
    ],
)
def test_extract_gold(field, gold):
    assert extract_gold(field) == gold


def test_extract_gold_rejects_missing_number():
    with pytest.raises(ValueError):
        extract_gold("no marker")


def test_is_correct_numeric_equality():
    assert is_correct("#### 18.0", "18")
    assert not is_correct("#### 19", "18")
    assert not is_correct("no idea", "18")


def test_completion_text_formats():
    assert completion_text("x") == "x"
    assert completion_text({"role": "assistant", "content": "y"}) == "y"
    assert completion_text([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]) == "ab"
    assert completion_text([{"role": "assistant", "content": None}]) == ""


def test_gsm8k_reward_conversational_batch():
    completions = [
        [{"role": "assistant", "content": "#### 5"}],
        [{"role": "assistant", "content": "#### 6"}],
        "#### 5",
    ]
    assert gsm8k_reward(completions=completions, gold=["5", "5", "5"], prompts=None) == [1.0, 0.0, 1.0]


def test_gsm8k_reward_is_picklable():
    import pickle

    assert pickle.loads(pickle.dumps(gsm8k_reward)) is gsm8k_reward
