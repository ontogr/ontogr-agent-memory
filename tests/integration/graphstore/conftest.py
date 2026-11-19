"""Fixtures for the graph store tests."""

from collections.abc import AsyncGenerator, Awaitable, Callable

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.store_urls import parse_graph_url
from tests.support.falkordb import falkordb_test_settings, falkordb_test_url
from tests.support.graph_schema import TEST_PART
from tests.support.neo4j import neo4j_test_settings, neo4j_test_url
from tests.support.tenants import new_tenant


@pytest.fixture
async def falkordb_handle(falkordb_store: FalkorDBGraphStore):
    """A fresh tenant graph with the test schema, erased afterwards."""
    handle = await falkordb_store.open(new_tenant(), [TEST_PART])
    yield handle
    await falkordb_store.erase_tenant(handle.tenant)


@pytest.fixture
async def falkordb_store_factory() -> AsyncGenerator[
    Callable[..., Awaitable[FalkorDBGraphStore]]
]:
    """Create more stores on the same server; they are closed afterwards."""
    made: list[FalkorDBGraphStore] = []

    async def make(**resilience: float) -> FalkorDBGraphStore:
        store = FalkorDBGraphStore(
            parse_graph_url(falkordb_test_url()), falkordb_test_settings(**resilience)
        )
        made.append(store)
        return store

    yield make
    for store in made:
        await store.close()


@pytest.fixture
async def falkordb_unreachable_store() -> AsyncGenerator[FalkorDBGraphStore]:
    """A store pointed at a port nothing listens on."""
    store = FalkorDBGraphStore(
        parse_graph_url(falkordb_test_url(port=1)),
        falkordb_test_settings(max_attempts=2, retry_delay=0.01),
    )
    yield store
    await store.close()


@pytest.fixture
async def neo4j_handle(neo4j_store: Neo4jGraphStore):
    """A fresh tenant with the test schema, erased afterwards."""
    handle = await neo4j_store.open(new_tenant(), [TEST_PART])
    yield handle
    await neo4j_store.erase_tenant(handle.tenant)


@pytest.fixture
async def neo4j_store_factory() -> AsyncGenerator[
    Callable[..., Awaitable[Neo4jGraphStore]]
]:
    """Create more stores on the same server; they are closed afterwards."""
    made: list[Neo4jGraphStore] = []

    async def make(**resilience: float) -> Neo4jGraphStore:
        store = Neo4jGraphStore(
            parse_graph_url(neo4j_test_url()), neo4j_test_settings(**resilience)
        )
        made.append(store)
        return store

    yield make
    for store in made:
        await store.close()


@pytest.fixture
async def neo4j_unreachable_store() -> AsyncGenerator[Neo4jGraphStore]:
    """A store pointed at a port nothing listens on."""
    store = Neo4jGraphStore(
        parse_graph_url(neo4j_test_url(uri="bolt://localhost:1")),
        neo4j_test_settings(max_attempts=2, retry_delay=0.01),
    )
    yield store
    await store.close()
