"""Run the shared retrieval contract on isolated graph tenants."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from tests.integration.test_retrieval import RetrievalContract
from tests.support.falkordb import falkordb_test_settings
from tests.support.graph_backend import GraphBackend
from tests.support.neo4j import neo4j_test_settings
from tests.support.retrieval_backends import graph_retrieval_backend
from tests.support.retrieval_world import RetrievalWorld, build_world
from tests.support.services import service_unreachable


@pytest_asyncio.fixture(scope="class", loop_scope="class", params=["falkordb", "neo4j"])
async def class_graph_backend(request) -> AsyncGenerator[GraphBackend]:
    """One backend per class, so the scenario is built once for all its tests."""
    if request.param == "falkordb":
        store = FalkorDBGraphStore(falkordb_test_settings())
        address = "FalkorDB"
    else:
        store = Neo4jGraphStore(neo4j_test_settings())
        address = "Neo4j"
    if not (await store.health()).ok:
        await store.close()
        service_unreachable(address, request.param)
    backend = GraphBackend(store, request.param)
    yield backend
    await backend.erase()
    await store.close()


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def retrieval_world(class_graph_backend: GraphBackend) -> RetrievalWorld:
    """Build the scenario once per backend; the contract only reads it."""
    return await build_world(await graph_retrieval_backend(class_graph_backend))


@pytest.mark.asyncio(loop_scope="class")
class TestRetrievalBackends(RetrievalContract):
    """Run the retrieval contract on FalkorDB and on Neo4j."""
