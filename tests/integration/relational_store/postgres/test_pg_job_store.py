"""The Postgres job store passes the shared job and erase contracts."""

from collections.abc import Awaitable, Callable
from datetime import datetime

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
from tests.contracts.job_store import T0, JobStoreContract
from tests.support.clock import MovingClock


class TestPostgresJobStore(JobStoreContract, EraseJournalContract):
    """Contract behavior plus the Postgres-only claim queries."""

    @pytest.fixture
    def make_store(self, pool, clean_tables) -> Callable[[], Awaitable[StateStore]]:
        """Return a factory for one open store on the shared database."""

        async def factory() -> StateStore:
            clock = MovingClock(T0)

            class ContractClock:
                def now(self) -> datetime:
                    return clock.now()

            return PostgresJobStore(
                pool, ContractClock(), owner_id="test-owner", lease_seconds=60
            )

        return factory

    async def test_schema_revision_is_recorded(self, pool):
        """Every shipped revision is recorded with its file checksum."""
        await ensure_postgres_schema(pool)
        revisions = pending_revisions(SCHEMA_FILES / "files_pg")
        async with pg_connection(pool) as connection:
            recorded = await connection.fetch(
                "SELECT revision, checksum FROM schema_migrations"
            )
        assert {row["revision"]: row["checksum"] for row in recorded} == {
            revision: checksum for revision, _, checksum in revisions
        }
        assert all(checksum == revision_checksum(sql) for _, sql, checksum in revisions)
