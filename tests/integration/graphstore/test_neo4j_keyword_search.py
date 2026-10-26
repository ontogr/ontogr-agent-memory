"""Keyword contract with the shared Neo4j fulltext indexes."""

import pytest

from ontogr_mem.graphstore.schemas import KEYWORD
from tests.contracts.keyword_search import KeywordSearchContract
from tests.support.keyword import graph_keyword_harness
from tests.support.tenants import new_tenant


class TestNeo4jKeywordSearch(KeywordSearchContract):
    """Neo4j returns only the requested tenant's documents."""

    @pytest.fixture
    async def harness(self, neo4j_store):
        """Open one tenant with optional fulltext indexes."""
        handle = await neo4j_store.open(new_tenant(), [KEYWORD])
        try:
            yield graph_keyword_harness(neo4j_store, handle)
        finally:
            await neo4j_store.erase_tenant(handle.tenant)

    @pytest.fixture
    async def other_harness(self, neo4j_store):
        """Open another tenant sharing the same indexes."""
        handle = await neo4j_store.open(new_tenant(), [KEYWORD])
        try:
            yield graph_keyword_harness(neo4j_store, handle)
        finally:
            await neo4j_store.erase_tenant(handle.tenant)
