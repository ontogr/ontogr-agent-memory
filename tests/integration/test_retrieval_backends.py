"""Run the shared retrieval contract on isolated graph tenants."""

from collections.abc import AsyncGenerator

import pytest

from tests.integration.test_retrieval import RetrievalContract
from tests.support.graph_backend import GraphBackend
from tests.support.retrieval_backends import graph_retrieval_backend
from tests.support.retrieval_world import RetrievalWorld, build_world


class TestRetrievalBackends(RetrievalContract):
    """Run the retrieval contract on FalkorDB and on Neo4j."""

    @pytest.fixture
    async def retrieval_world(
        self, graph_backend: GraphBackend
    ) -> AsyncGenerator[RetrievalWorld]:
        """Build one scenario tenant on the backend under test."""
        yield await build_world(await graph_retrieval_backend(graph_backend))
