"""Shared lifecycle behavior for erase journals and their memory twin."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.relational_store.jobs.erase_journal import (
    EraseJournal,
    ErasureRecord,
    ErasureStatus,
)


T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def make_erasure(
    erasure_id: str = "erase-1", tenant: str = "tenant_a"
) -> ErasureRecord:
    """Return one running journal record with only ids and counts."""
    return ErasureRecord(
        erasure_id=erasure_id,
        tenant=tenant,
        mode="erase",
        principal_id="caller",
        selector_kind="episode_ids",
        target="digest",
        episode_ids=("episode-1",),
        fact_ids=("fact-1",),
        observation_ids=("observation-1",),
        entity_ids=("entity-1",),
        created_at=T0,
    )


class EraseJournalContract:
    """Journal lifecycle and tenant-filtering rules shared by adapters."""

    @pytest.fixture
    def make_store(self) -> Callable[[], Awaitable[EraseJournal]]:
        """Return a factory for isolated open journals."""
        raise NotImplementedError

    @pytest.fixture
    async def journal(self, make_store) -> EraseJournal:
        """Return one journal for a contract case."""
        return await make_store()

    async def test_create_step_finish_and_read_keep_every_field(self, journal):
        """The journal retains its target and terminal outcome."""
        record = make_erasure()
        assert await journal.create(record) == record.erasure_id

        await journal.step(record.erasure_id, "retract")
        running = await journal.get(record.erasure_id)
        assert running is not None
        assert running.last_step == "retract"
        assert running.status is ErasureStatus.RUNNING
        assert running.episode_ids == ("episode-1",)
        assert running.fact_ids == ("fact-1",)
        assert running.observation_ids == ("observation-1",)
        assert running.entity_ids == ("entity-1",)

        await journal.finish(
            record.erasure_id,
            ErasureStatus.COMPLETED,
            (),
            (("episodes", 1), ("facts", 1)),
            T0 + timedelta(seconds=1),
        )

        result = await journal.get(record.erasure_id)
        assert result is not None
        assert result.status is ErasureStatus.COMPLETED
        assert result.finished_at == T0 + timedelta(seconds=1)
        assert result.residue == ()
        assert result.counts == (("episodes", 1), ("facts", 1))

    async def test_running_filters_by_tenant_and_returns_oldest_first(self, journal):
        """Recovery sees only running records for its requested Tenant."""
        first = make_erasure("erase-1")
        second = make_erasure("erase-2", "tenant_b")
        third = make_erasure("erase-3")
        await journal.create(first)
        await journal.create(second)
        await journal.create(third)
        await journal.finish(
            third.erasure_id,
            ErasureStatus.FAILED,
            ("model_cache",),
            (),
            T0 + timedelta(seconds=3),
        )

        assert [row.erasure_id for row in await journal.running("tenant_a")] == [
            first.erasure_id
        ]
        assert [row.erasure_id for row in await journal.running(None)] == [
            first.erasure_id,
            second.erasure_id,
        ]
        assert await journal.get("missing") is None
