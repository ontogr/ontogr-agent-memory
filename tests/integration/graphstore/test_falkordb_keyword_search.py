"""Keyword contract with real fulltext indexes in isolated FalkorDB graphs."""

import pytest

from ontogr_mem.graphstore.keyword_search import GraphKeywordSearch
from ontogr_mem.graphstore.schemas import KEYWORD
from ontogr_mem.retrieval import KeywordIndexError, KeywordKind
from tests.contracts.keyword_search import KeywordSearchContract
from tests.support.keyword import graph_keyword_harness
from tests.support.tenants import new_tenant


class TestFalkorDBKeywordSearch(KeywordSearchContract):
    """FalkorDB satisfies the keyword port contract."""

    @pytest.fixture
    async def harness(self, falkordb_store):
        """Provision the optional indexes and erase the tenant afterwards."""
        handle = await falkordb_store.open(new_tenant(), [KEYWORD])
        try:
            yield graph_keyword_harness(falkordb_store, handle)
        finally:
            await falkordb_store.erase_tenant(handle.tenant)

    @pytest.fixture
    async def other_harness(self, falkordb_store):
        """Provision a second isolated corpus."""
        handle = await falkordb_store.open(new_tenant(), [KEYWORD])
        try:
            yield graph_keyword_harness(falkordb_store, handle)
        finally:
            await falkordb_store.erase_tenant(handle.tenant)

    async def test_missing_index_is_an_error(self, falkordb_store):
        """A graph without optional indexes does not masquerade as an empty hit."""
        handle = await falkordb_store.open(new_tenant())
        try:
            keyword = GraphKeywordSearch(falkordb_store.runner(handle))
            with pytest.raises(KeywordIndexError):
                await keyword.search(KeywordKind.EPISODE, "Berlin", limit=10)
        finally:
            await falkordb_store.erase_tenant(handle.tenant)
