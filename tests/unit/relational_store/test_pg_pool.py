"""The Postgres pool is created once per DSN even under concurrent first use."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

import ontogr_mem._extras
from ontogr_mem.relational_store import engine


class TestPgPool:
    """Behavior of the process-wide Postgres pool cache."""

    @pytest.fixture
    def dsn(self):
        """Provide a DSN that no other test shares, and drop its cached pool after."""
        dsn = f"postgresql://unit-{uuid4().hex}/ontogr_mem"
        yield dsn
        engine._pg_pools.pop(dsn, None)

    @pytest.fixture
    def created_pools(self, monkeypatch):
        """Replace asyncpg with a fake whose pools are recorded as created."""
        created = []

        async def create_pool(dsn):
            await asyncio.sleep(0)
            pool = SimpleNamespace(dsn=dsn)
            created.append(pool)
            return pool

        fake_asyncpg = SimpleNamespace(create_pool=create_pool)
        monkeypatch.setattr(
            ontogr_mem._extras, "import_extra", lambda module, extra: fake_asyncpg
        )
        return created

    async def test_concurrent_first_callers_share_one_pool(self, dsn, created_pools):
        """Two callers racing on a new DSN receive the same single pool."""
        first, second = await asyncio.gather(engine.pg_pool(dsn), engine.pg_pool(dsn))

        assert first is second
        assert len(created_pools) == 1

    async def test_returns_the_cached_pool_on_later_calls(self, dsn, created_pools):
        """A later call for the same DSN does not create another pool."""
        first = await engine.pg_pool(dsn)
        again = await engine.pg_pool(dsn)

        assert again is first
        assert len(created_pools) == 1
