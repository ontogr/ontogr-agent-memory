"""Price model usage with an optional per-million-token rate table."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Price:
    """Input and output rates in dollars per million tokens.

    Attributes:
        input_per_m: Input-token rate in dollars per million tokens.
        output_per_m: Output-token rate in dollars per million tokens.
    """

    input_per_m: float
    output_per_m: float


class PriceTable:
    """Resolve model prices by the longest matching model-name prefix.

    Attributes:
        valid_as_of: Date associated with the configured prices, if known.
    """

    def __init__(
        self,
        prices: Mapping[str, Price] | None = None,
        *,
        valid_as_of: date | None = None,
    ) -> None:
        """Create a table of model-name prefixes and their token rates."""
        self._prices = dict(prices or {})
        self.valid_as_of = valid_as_of

    def price(self, model: str) -> Price | None:
        """Return the longest configured prefix match, if one exists."""
        name = model.partition("/")[2] if "/" in model else model
        matches = (prefix for prefix in self._prices if name.startswith(prefix))
        prefix = max(matches, key=len, default=None)
        return None if prefix is None else self._prices[prefix]

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float | None:
        """Return the known total cost, or ``None`` for an unknown model."""
        price = self.price(model)
        if price is None:
            return None
        return (tokens_in * price.input_per_m + tokens_out * price.output_per_m) / 1e6
