"""Fixtures shared by integration tests."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.store_urls import parse_graph_url
from tests.support.falkordb import falkordb_test_settings, falkordb_test_url
from tests.support.graph_backend import GraphBackend
from tests.support.neo4j import neo4j_test_settings, neo4j_test_url
from tests.support.services import service_unreachable


@pytest.fixture
async def falkordb_store() -> AsyncGenerator[FalkorDBGraphStore]:
    """A store on the compose FalkorDB; the test fails if the server is down."""
    settings = falkordb_test_settings()
    target = parse_graph_url(falkordb_test_url())
    store = FalkorDBGraphStore(target, settings)
    if not (await store.health()).ok:
        await store.close()
        service_unreachable("FalkorDB", target.identity())
    yield store
    await store.close()


@pytest.fixture
async def neo4j_store() -> AsyncGenerator[Neo4jGraphStore]:
    """A store on the compose Neo4j; the test fails if the server is down."""
    settings = neo4j_test_settings()
    target = parse_graph_url(neo4j_test_url())
    store = Neo4jGraphStore(target, settings)
    if not (await store.health()).ok:
        await store.close()
        service_unreachable("Neo4j", target.identity())
    yield store
    await store.close()


@pytest.fixture(params=["falkordb", "neo4j"])
async def graph_backend(
    request, falkordb_store: FalkorDBGraphStore, neo4j_store: Neo4jGraphStore
) -> AsyncGenerator[GraphBackend]:
    """Run the test on FalkorDB and on Neo4j, erasing the tenants it opened."""
    store = falkordb_store if request.param == "falkordb" else neo4j_store
    backend = GraphBackend(store, request.param)
    yield backend
    await backend.erase()
