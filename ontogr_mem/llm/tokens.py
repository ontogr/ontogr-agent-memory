"""Approximate token counts for local text budgets."""


def estimate_tokens(text: str) -> int:
    """Estimate one token per four characters, rounding up."""
    return -(-len(text) // 4)
