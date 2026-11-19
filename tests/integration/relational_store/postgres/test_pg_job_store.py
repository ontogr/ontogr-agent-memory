"""The Postgres job store passes the shared job and erase contracts."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

from ontogr_mem.relational_store.engine import pg_connection
from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from ontogr_mem.relational_store.schema.runner import (
    SCHEMA_FILES,
    ensure_postgres_schema,
    pending_revisions,
    revision_checksum,
)
from tests.contracts.erase_journal import EraseJournalContract
from tests.contracts.job_store import JobStoreContract


T0 = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)


class TestPostgresJobStore(JobStoreContract, EraseJournalContract):
    """Contract behavior plus the Postgres-only claim queries."""

    @pytest.fixture
    def make_store(self, pool, clean_tables) -> Callable[[], Awaitable[StateStore]]:
        """Return a factory for one open store on the shared database."""

        async def factory() -> StateStore:
            return PostgresJobStore(pool, owner_id="test-owner", lease_seconds=60)

        return factory

    async def test_schema_revision_is_recorded(self, pool):
        """The baseline revision is applied once with its file checksum."""
        await ensure_postgres_schema(pool)
        ((revision, _, checksum),) = pending_revisions(SCHEMA_FILES / "files_pg")
        async with pg_connection(pool) as connection:
            recorded = await connection.fetchrow(
                "SELECT revision, checksum FROM schema_migrations"
            )
        assert recorded is not None
        assert (recorded["revision"], recorded["checksum"]) == (revision, checksum)
        assert checksum == revision_checksum(
            (SCHEMA_FILES / "files_pg" / "0001_baseline.sql").read_text("utf-8")
        )
