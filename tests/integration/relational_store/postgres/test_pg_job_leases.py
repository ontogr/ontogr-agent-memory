"""Postgres lease behavior shared by independent job-store owners."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.data_models.jobs import JobProgress, JobState
from ontogr_mem.relational_store.jobs.errors import LeaseLostError
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from tests.contracts.erase_journal import make_erasure
from tests.contracts.job_store import make_job
from tests.support.clock import FixedClock


class TestPostgresJobLeases:
    """Verify a live lease blocks another owner until recovery expires it."""

    async def test_second_owner_claims_job_after_first_owner_lease_expires(
        self, pool, clean_tables
    ):
        """Release the second owner's claim only after the first lease expires."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        first = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=60)
        second = PostgresJobStore(pool, clock, owner_id="second", lease_seconds=60)
        now = clock.now()
        first_job, _ = await first.create_job(make_job())
        await first.create_job(
            replace(make_job(job_id="job_b"), idempotency_key="request-two")
        )

        claimed = await first.claim_next("tenant_a", now, lambda job: True)

        assert claimed is not None
        assert claimed.status is JobState.RUNNING
        assert claimed.job_id == first_job.job_id
        assert (
            await second.claim_next(
                "tenant_a", now + timedelta(seconds=1), lambda job: True
            )
            is None
        )
        assert await second.tenant_has_live_lease("tenant_a", now)

        assert (
            await second.claim_next(
                "tenant_a", now + timedelta(seconds=59), lambda job: True
            )
            is None
        )
        assert await second.tenant_has_live_lease(
            "tenant_a", now + timedelta(seconds=59)
        )

        await second.requeue_expired(now + timedelta(seconds=61))
        recovered = await second.claim_next(
            "tenant_a", now + timedelta(seconds=61), lambda job: True
        )

        assert recovered is not None
        assert recovered.job_id == claimed.job_id
        assert recovered.status is JobState.RUNNING
        assert await second.tenant_has_live_lease(
            "tenant_a", now + timedelta(seconds=61)
        )

    async def test_submission_waits_for_tenant_lock_and_is_admitted_after_release(
        self, pool, clean_tables
    ):
        """Serialize admission with clear while allowing the later submission."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        store = PostgresJobStore(pool, clock, owner_id="owner", lease_seconds=60)
        job = make_job()

        async with store.tenant_lock(job.tenant_id):
            submission = asyncio.create_task(store.create_job(job))
            await asyncio.sleep(0)
            assert not submission.done()

        created, replayed = await submission

        assert created.job_id == job.job_id
        assert not replayed

    async def test_renewed_lease_blocks_claim_past_original_expiry(
        self, pool, clean_tables
    ):
        """Renewal keeps another owner out beyond the original lease deadline."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        first = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=60)
        second = PostgresJobStore(pool, clock, owner_id="second", lease_seconds=60)
        now = clock.now()
        await first.create_job(make_job())
        claimed = await first.claim_next("tenant_a", now, lambda _: True)
        assert claimed is not None

        renewed_at = now + timedelta(seconds=30)
        await first.renew_leases(renewed_at)
        after_original_expiry = now + timedelta(seconds=61)

        assert (
            await second.claim_next("tenant_a", after_original_expiry, lambda _: True)
            is None
        )
        assert await second.tenant_has_live_lease("tenant_a", after_original_expiry)

    async def test_expired_owner_cannot_save_running_job(self, pool, clean_tables):
        """Reject an owner write after its lease timestamp expires."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        first = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=1)
        now = clock.now()
        await first.create_job(make_job())
        claimed = await first.claim_next("tenant_a", now, lambda _: True)
        assert claimed is not None

        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE jobs SET lease_expires_at = $1 WHERE job_id = $2",
                int((now - timedelta(seconds=1)).timestamp() * 1_000_000),
                claimed.job_id,
            )

        assert not await first.save_job(replace(claimed, attempts=2))

    async def test_expired_owner_cannot_change_progress(self, pool, clean_tables):
        """Reject progress writes after the owner's lease expires."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        store = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=1)
        await store.create_job(make_job())
        claimed = await store.claim_next("tenant_a", clock.now(), lambda _: True)
        assert claimed is not None
        await store.set_job_progress(
            claimed.job_id, JobProgress(stage="index", done=1, total=3)
        )
        before = await store.get_job(claimed.job_id)

        clock.advance(seconds=2)
        with pytest.raises(LeaseLostError):
            await store.set_job_progress(
                claimed.job_id, JobProgress(stage="index", done=2, total=3)
            )

        assert await store.get_job(claimed.job_id) == before

    async def test_running_to_queued_save_clears_lease_for_reclaim(
        self, pool, clean_tables
    ):
        """Allow another owner to claim a job saved back to the queue."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        first = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=60)
        second = PostgresJobStore(pool, clock, owner_id="second", lease_seconds=60)
        now = clock.now()
        job, _ = await first.create_job(make_job())
        claimed = await first.claim_next("tenant_a", now, lambda _: True)
        assert claimed is not None

        queued = replace(claimed, status=JobState.QUEUED)
        assert await first.save_job(queued)
        assert await second.claim_next("tenant_a", now, lambda _: True) is not None

    async def test_recovery_transfers_expired_erasure_to_one_owner(
        self, pool, clean_tables
    ):
        """Only the winner receives a tenant-scoped recovery lease."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        first = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=60)
        second = PostgresJobStore(pool, clock, owner_id="second", lease_seconds=60)
        third = PostgresJobStore(pool, clock, owner_id="third", lease_seconds=60)
        record = make_erasure()
        await first.create(record)

        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE erasures SET lease_expires_at = NULL WHERE erasure_id = $1",
                record.erasure_id,
            )

        now = clock.now()
        claimed = await second.claim_recoverable("tenant_a", now)

        assert [item.erasure_id for item in claimed] == [record.erasure_id]
        assert await third.claim_recoverable("tenant_a", now) == []
        assert await third.claim_recoverable("tenant_b", now) == []
        await second.step(record.erasure_id, "retract")
