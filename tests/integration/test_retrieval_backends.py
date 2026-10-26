"""Run the shared retrieval contract on isolated graph tenants."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from tests.integration.test_retrieval import TestRetrieval as _RetrievalContract
from tests.support.retrieval_backends import graph_retrieval_backend
from tests.support.retrieval_world import RetrievalWorld, build_world


class TestRetrievalFalkorDB(_RetrievalContract):
    """Run the retrieval contract on FalkorDB's graph-native stores."""

    @pytest.fixture
    async def retrieval_world(
        self, falkordb_store: FalkorDBGraphStore
    ) -> AsyncGenerator[RetrievalWorld]:
        """Build and erase one isolated FalkorDB scenario tenant."""
        async with graph_retrieval_backend(
            falkordb_store, tenant_keyed=False
        ) as backend:
            yield await build_world(backend)


class TestRetrievalNeo4j(_RetrievalContract):
    """Run the retrieval contract on Neo4j's graph-native stores."""

    @pytest.fixture
    async def retrieval_world(
        self, neo4j_store: Neo4jGraphStore
    ) -> AsyncGenerator[RetrievalWorld]:
        """Build and erase one isolated Neo4j scenario tenant."""
        async with graph_retrieval_backend(neo4j_store, tenant_keyed=True) as backend:
            yield await build_world(backend)
