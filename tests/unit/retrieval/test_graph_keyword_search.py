"""Port-boundary error behavior for the graph keyword adapter."""

import pytest

from ontogr_mem.graphstore.cypher import PROPERTY_SCOPE
from ontogr_mem.graphstore.errors import QueryTimeoutError
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.keyword_search import GraphKeywordSearch
from ontogr_mem.retrieval.errors import KeywordTimeoutError
from ontogr_mem.retrieval.keyword import KeywordKind


class FailingRunner:
    """A scoped runner that raises at the storage boundary."""

    scope = PROPERTY_SCOPE

    def __init__(self, error: Exception) -> None:
        """Keep the server-side failure that the adapter must expose."""
        self.error = error

    async def run(self, spec: QuerySpec) -> Rows:
        """Raise the configured failure instead of returning graph rows."""
        raise self.error


class TestGraphKeywordSearchErrors:
    """Map only known graph failures and preserve unexpected exceptions."""

    async def test_maps_a_graph_query_timeout_to_keyword_timeout(self):
        """A typed driver timeout becomes the public keyword timeout."""
        error = QueryTimeoutError("server query timed out")
        search = GraphKeywordSearch(FailingRunner(error))

        with pytest.raises(
            KeywordTimeoutError, match="server query timed out"
        ) as caught:
            await search.search(KeywordKind.EPISODE, "valid query", limit=5)

        assert caught.value.__cause__ is error

    async def test_propagates_an_unexpected_value_error_unchanged(self):
        """A programming error is not converted to a keyword storage failure."""
        error = ValueError("invalid adapter input")
        search = GraphKeywordSearch(FailingRunner(error))

        with pytest.raises(ValueError, match="invalid adapter input") as caught:
            await search.search(KeywordKind.EPISODE, "valid query", limit=5)

        assert caught.value is error
