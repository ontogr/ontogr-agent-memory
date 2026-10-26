"""Fixtures shared by integration tests."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.vectorstore.pgvector import PgVectorSettings, Pool, create_pool
from tests.support.falkordb import falkordb_test_settings
from tests.support.neo4j import neo4j_test_settings
from tests.support.services import service_unreachable


@pytest.fixture
async def falkordb_store() -> AsyncGenerator[FalkorDBGraphStore]:
    """A store on the compose FalkorDB; the test fails if the server is down."""
    settings = falkordb_test_settings()
    store = FalkorDBGraphStore(settings)
    if not (await store.health()).ok:
        await store.close()
        service_unreachable("FalkorDB", f"{settings.host}:{settings.port}")
    yield store
    await store.close()


@pytest.fixture
async def neo4j_store() -> AsyncGenerator[Neo4jGraphStore]:
    """A store on the compose Neo4j; the test fails if the server is down."""
    settings = neo4j_test_settings()
    store = Neo4jGraphStore(settings)
    if not (await store.health()).ok:
        await store.close()
        service_unreachable("Neo4j", settings.uri)
    yield store
    await store.close()


@pytest.fixture
async def pg_pool() -> AsyncGenerator[Pool]:
    """A pool on the compose pgvector; the test fails if the server is down."""
    settings = PgVectorSettings()
    try:
        pool = await create_pool(settings)
    except Exception as error:
        service_unreachable("pgvector", settings.dsn)
        raise error
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as error:
        await pool.close()
        service_unreachable("pgvector", settings.dsn)
        raise error
    yield pool
    await pool.close()
