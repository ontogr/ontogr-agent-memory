"""Shared fixtures for the Postgres relational-store adapters."""

import pytest

from ontogr_mem._extras import import_extra
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.relational_store.engine import pg_connection
from ontogr_mem.relational_store.schema.runner import (
    ensure_postgres_schema,
)


DSN = StateSettings(dsn="postgresql://memory:memorytest123@127.0.0.1:15433/memory").dsn
DATA_TABLES = ("jobs", "erasures", "audit_rows", "operation_rows")


@pytest.fixture
async def pool():
    """Return a per-test pool with the schema ensured, then close it.

    The pool is created inside the test's event loop rather than through the
    process-wide cache because a pool bound to a finished loop never
    resolves another await.
    """
    asyncpg = import_extra("asyncpg", "postgres")
    pool = await asyncpg.create_pool(DSN)
    try:
        await ensure_postgres_schema(pool)
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def clean_tables(pool):
    """Empty every data table and restart identities before a case runs."""
    async with pg_connection(pool) as connection:
        await connection.execute(
            "TRUNCATE TABLE jobs, erasures, audit_rows, operation_rows RESTART IDENTITY"
        )
