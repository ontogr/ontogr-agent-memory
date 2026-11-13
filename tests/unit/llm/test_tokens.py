"""Character-based estimates used to size local text budgets."""

import pytest

from ontogr_mem.llm.tokens import estimate_tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("x", 1),
        ("xxxx", 1),
        ("xxxxx", 2),
        ("😀😀😀😀😀", 2),
    ],
)
def test_estimate_tokens_rounds_up_by_character_count(text: str, expected: int) -> None:
    """Unicode characters count once regardless of their encoded byte length."""
    assert estimate_tokens(text) == expected
