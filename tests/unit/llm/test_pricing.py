"""Price lookup and cost accounting for model usage."""

import pytest

from ontogr_mem.llm.pricing import Price, PriceTable


class TestPriceTable:
    """Model names resolve to the most specific configured price."""

    def test_longest_matching_prefix_wins(self) -> None:
        """A specific prefix takes precedence over a general one."""
        prices = PriceTable(
            {
                "gpt-4o": Price(1.0, 3.0),
                "gpt-4o-mini": Price(2.0, 6.0),
            }
        )

        assert prices.price("gpt-4o-mini-2026") == Price(2.0, 6.0)

    def test_provider_prefix_is_removed_before_matching(self) -> None:
        """Qualified model names match the model portion."""
        prices = PriceTable({"gpt-4o-mini": Price(2.0, 6.0)})

        assert prices.price("provider/gpt-4o-mini") == Price(2.0, 6.0)

    def test_unknown_model_has_no_price_or_cost(self) -> None:
        """An unknown model remains explicitly unpriced."""
        prices = PriceTable()

        assert prices.price("unknown") is None
        assert prices.cost("unknown", 1000, 500) is None

    def test_cost_is_per_million_tokens(self) -> None:
        """Input and output charges are summed per million tokens."""
        prices = PriceTable({"gpt-4o": Price(2.0, 6.0)})

        assert prices.cost("gpt-4o-mini", 1000, 500) == pytest.approx(0.005)
