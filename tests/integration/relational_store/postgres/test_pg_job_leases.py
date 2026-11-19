"""Postgres lease behavior shared by independent job-store owners."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from tests.contracts.job_store import make_job


class TestPostgresJobLeases:
    """Verify a live lease blocks another owner until recovery expires it."""

    async def test_second_owner_claims_job_after_first_owner_lease_expires(
        self, pool, clean_tables
    ):
        """Release the second owner's claim only after the first lease expires."""
        first = PostgresJobStore(pool, owner_id="first", lease_seconds=60)
        second = PostgresJobStore(pool, owner_id="second", lease_seconds=60)
        now = datetime.now(UTC)
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
