"""Run the keyword port contract over isolated memory stores."""

import pytest

from tests.contracts.keyword_search import KeywordSearchContract
from tests.support.keyword import memory_keyword_harness


class TestMemoryKeywordSearch(KeywordSearchContract):
    """The in-memory search obeys the same observable contract as servers."""

    @pytest.fixture
    def harness(self):
        """Return the tenant under test."""
        return memory_keyword_harness()

    @pytest.fixture
    def other_harness(self):
        """Return an unrelated tenant."""
        return memory_keyword_harness()
