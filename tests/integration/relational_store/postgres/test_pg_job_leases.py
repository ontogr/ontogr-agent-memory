"""Postgres lease behavior shared by independent job-store owners."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.data_models.jobs import JobProgress, JobState
from ontogr_mem.relational_store.jobs.errors import LeaseLostError
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from ontogr_mem.service.contracts import AddMemory, ForgetMode, ListEpisodes, MemoryKind
from ontogr_mem.service.errors import ConflictError
from ontogr_mem.service.forget.selector import ForgetSelector
from ontogr_mem.service.principal import Grant
from tests.contracts.erase_journal import make_erasure
from tests.contracts.job_store import make_job
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.jobs import wait_for_job
from tests.support.memory_service import memory_service_harness


class TestPostgresJobLeases:
    """Verify a live lease blocks another owner until recovery expires it."""

    async def test_live_tenant_lease_blocks_forget_until_released(
        self, pool, clean_tables, graph_backend: GraphBackend
    ):
        """Forget conflicts while a Postgres job is leased, then removes its fact."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        state = PostgresJobStore(pool, clock, owner_id="forget-owner", lease_seconds=60)
        harness = await memory_service_harness(graph_backend, state=state)
        admin = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.ADMIN})}}
        )
        receipt = await harness.service.add_memory(
            admin,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                wait_seconds=5,
            ),
        )
        indexed_job = await wait_for_job(harness.service, admin, receipt.job_id)
        fact_id = indexed_job.result["fact_ids"][0]
        selector = ForgetSelector(fact_id=fact_id)
        preview = await harness.service.preview_forget(
            admin, tenant="t1", selector=selector, mode=ForgetMode.DELETE
        )
        lease_job, _ = await state.create_job(
            make_job("forget-lease-job", tenant_id="t1")
        )
        claimed = await state.claim_next("t1", clock.now(), lambda _: True)
        assert claimed is not None
        assert claimed.job_id == lease_job.job_id

        with pytest.raises(ConflictError):
            await harness.service.forget(
                admin,
                tenant="t1",
                selector=selector,
                mode=ForgetMode.DELETE,
                confirm_token=preview.confirm_token,
            )

        await state.release_job(lease_job.job_id)
        result = await harness.service.forget(
            admin,
            tenant="t1",
            selector=selector,
            mode=ForgetMode.DELETE,
            confirm_token=preview.confirm_token,
        )

        assert result.status == "completed"
        assert result.removed.episodes == 1
        page = await harness.service.list_episodes(admin, ListEpisodes(tenant="t1"))
        assert page.episodes == []

    async def test_recovery_leaves_job_running_before_lease_expires(
        self, pool, clean_tables
    ):
        """Recovery keeps a running job queued until its lease expires."""
        now = datetime(2026, 10, 23, 12, tzinfo=UTC)
        clock = FixedClock(now)
        store = PostgresJobStore(pool, clock, owner_id="owner", lease_seconds=60)
        job, _ = await store.create_job(make_job())
        claimed = await store.claim_next(job.tenant_id, now, lambda _: True)

        assert claimed is not None
        await store.requeue_expired(now + timedelta(seconds=30))

        assert await store.get_job(job.job_id) == claimed
        assert await store.queued_jobs() == []

    async def test_erasure_lease_starts_at_store_clock(self, pool, clean_tables):
        """Set lease timestamps from the store clock and retain record creation time."""
        now = datetime(2026, 10, 24, 12, tzinfo=UTC)
        clock = FixedClock(now)
        store = PostgresJobStore(pool, clock, owner_id="owner", lease_seconds=60)
        record = make_erasure()

        await store.create(record)

        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT created_at, lease_expires_at, heartbeat_at FROM erasures "
                "WHERE erasure_id = $1",
                record.erasure_id,
            )

        assert row["created_at"] == int(record.created_at.timestamp() * 1_000_000)
        expected_expiry = int((now + timedelta(seconds=60)).timestamp() * 1_000_000)
        assert row["lease_expires_at"] == expected_expiry
        assert row["heartbeat_at"] == int(now.timestamp() * 1_000_000)

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
            async with pool.acquire() as connection:
                tenant_lock_key = await connection.fetchval(
                    "SELECT hashtextextended($1, 0)", job.tenant_id
                )
                for _ in range(100):
                    waiting = await connection.fetchval(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM pg_locks AS waiter "
                        "JOIN pg_stat_activity AS activity "
                        "ON activity.pid = waiter.pid "
                        "WHERE activity.datname = current_database() "
                        "AND waiter.locktype = 'advisory' "
                        "AND waiter.classid = (($1::bigint >> 32) & 4294967295)::oid "
                        "AND waiter.objid = ($1::bigint & 4294967295)::oid "
                        "AND waiter.objsubid = 1 AND NOT waiter.granted "
                        "AND activity.wait_event_type = 'Lock'"
                        ")",
                        tenant_lock_key,
                    )
                    if waiting:
                        break
                    await asyncio.sleep(0.01)
                assert waiting

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

    async def test_saving_unknown_job_raises_key_error(self, pool, clean_tables):
        """Report a missing job separately from a lost lease."""
        store = PostgresJobStore(
            pool,
            FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC)),
            owner_id="owner",
            lease_seconds=60,
        )

        with pytest.raises(KeyError, match="missing"):
            await store.save_job(make_job(job_id="missing"))

    async def test_expired_owner_cannot_change_progress(self, pool, clean_tables):
        """Reject progress writes after the owner's lease expires."""
        clock = FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC))
        store = PostgresJobStore(pool, clock, owner_id="first", lease_seconds=1)
        job = make_job()
        await store.create_job(job)
        claimed = await store.claim_next("tenant_a", clock.now(), lambda _: True)
        assert claimed is not None
        assert claimed.job_id == job.job_id
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

    async def test_missing_job_progress_raises_key_error(self, pool, clean_tables):
        """Report a missing job separately from a lost lease."""
        store = PostgresJobStore(
            pool,
            FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC)),
            owner_id="owner",
            lease_seconds=60,
        )

        with pytest.raises(KeyError, match="missing"):
            await store.set_job_progress(
                "missing", JobProgress(stage="index", done=1, total=3)
            )

    async def test_missing_erasure_step_raises_key_error(self, pool, clean_tables):
        """Report a missing erase row separately from a lost lease."""
        store = PostgresJobStore(
            pool,
            FixedClock(datetime(2026, 10, 23, 12, tzinfo=UTC)),
            owner_id="owner",
            lease_seconds=60,
        )

        with pytest.raises(KeyError, match="missing"):
            await store.step("missing", "retract")

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
        other_tenant_record = make_erasure("erase-2", tenant="tenant_b")
        await first.create(record)
        await first.create(other_tenant_record)

        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE erasures SET lease_expires_at = NULL "
                "WHERE erasure_id = ANY($1)",
                [record.erasure_id, other_tenant_record.erasure_id],
            )

        now = clock.now()
        claimed = await second.claim_recoverable("tenant_a", now)

        assert [item.erasure_id for item in claimed] == [record.erasure_id]
        assert await third.claim_recoverable("tenant_a", now) == []
        second_tenant_claimed = await second.claim_recoverable("tenant_b", now)
        assert [item.erasure_id for item in second_tenant_claimed] == [
            other_tenant_record.erasure_id
        ]
        await second.step(record.erasure_id, "retract")
