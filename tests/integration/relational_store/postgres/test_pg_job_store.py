"""The Postgres job store passes the shared job and erase contracts."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
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
from tests.contracts.job_store import JobStoreContract, make_job


T0 = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)


class TestPostgresJobStore(JobStoreContract, EraseJournalContract):
    """Contract behavior plus the Postgres-only claim queries."""

    @pytest.fixture
    def make_store(self, pool, clean_tables) -> Callable[[], Awaitable[StateStore]]:
        """Return a factory for one open store on the shared database."""

        async def factory() -> StateStore:
            return PostgresJobStore(pool)

        return factory

    async def test_claim_takes_disjoint_queued_batches(self, pool, clean_tables):
        """Two workers claim disjoint batches oldest first."""
        store = PostgresJobStore(pool)
        for index in range(4):
            job, _ = await store.create_job(
                replace(make_job(job_id=f"j{index}"), idempotency_key=f"req-{index}")
            )
        first = await store.claim("worker-a", limit=2, now=T0)
        second = await store.claim("worker-b", limit=2, now=T0)
        third = await store.claim("worker-a", limit=2, now=T0)

        assert [job.job_id for job in first] == ["j0", "j1"]
        assert [job.job_id for job in second] == ["j2", "j3"]
        assert third == []
        running = await store.requeue_running()
        assert len(running) == 4

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
