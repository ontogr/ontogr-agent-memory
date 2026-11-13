"""The checked conversion from the test fake to the real Memory type."""

from datetime import datetime

import pytest

from ontogr_mem.service.contracts import MemoryContext
from tests.support.fake_memory import FakeMemory, as_memory


class TestAsMemory:
    """Behavior of the helper that turns a FakeMemory into a Memory."""

    def test_returns_the_same_fake_when_signatures_match(self) -> None:
        """Returns the plain fake unchanged when every real argument is accepted."""
        fake = FakeMemory()

        memory = as_memory(fake)

        assert memory is fake

    def test_names_the_missing_parameter_when_a_method_lacks_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises AssertionError naming the parameter that the fake does not accept."""
        fake = FakeMemory()

        async def search_without_budget(
            query: str,
            *,
            as_of: datetime | None = None,
            include_history: bool = False,
            limit: int = 10,
        ) -> MemoryContext:
            return MemoryContext(state="none", rendered="")

        monkeypatch.setattr(fake, "search", search_without_budget)

        with pytest.raises(AssertionError, match="budget_tokens"):
            as_memory(fake)
