"""Token and cost totals recorded for one operation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperationUsage:
    """Token counts and known model cost for one operation.

    Attributes:
        tokens_in: Input tokens charged to this operation.
        tokens_out: Output tokens charged to this operation.
        cost_usd: Priced cost, or None when at least one call was unpriced.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = 0.0

    def __post_init__(self) -> None:
        """Reject negative token counts."""
        if self.tokens_in < 0 or self.tokens_out < 0:
            raise ValueError("token counts must not be negative")
