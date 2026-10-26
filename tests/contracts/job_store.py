"""Shared behavior for durable job storage and its in-memory twin."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.relational_store.jobs.errors import IdempotencyConflictError
from ontogr_mem.relational_store.jobs.ports import StateStore


T0 = datetime(2026, 10, 23, 12, 0, 0, 123456, tzinfo=UTC)


def make_job(job_id: str = "j1", tenant_id: str = "tenant_a") -> Job:
    """Return a queued job with nested Unicode payload and precise timestamps."""
    return Job(
        job_id=job_id,
        tenant_id=tenant_id,
        op="index",
        payload={"text": "café", "nested": {"values": [1, "é", None]}},
        request_hash="hash-one",
        status=JobState.QUEUED,
        attempts=0,
        created_at=T0,
        updated_at=T0,
        idempotency_key="request-one",
        trace_id="trace-one",
    )


class JobStoreContract:
    """Storage semantics shared by every job adapter.

    Subclasses supply a factory returning a fresh, open store on each call.
    """

    @pytest.fixture
    def make_store(self) -> Callable[[], Awaitable[StateStore]]:
        """Return a factory for isolated stores."""
        raise NotImplementedError

    @pytest.fixture
    async def store(self, make_store) -> StateStore:
        """Return one open store for a contract case."""
        return await make_store()

    async def test_creates_and_reads_a_job_with_every_field(self, store):
        """All record fields survive creation, saving, and reading."""
        job = make_job()
        created, replay = await store.create_job(job)
        saved = replace(
            job,
            status=JobState.FAILED,
            attempts=2,
            result={"nested": ["résultat", {"ok": False}]},
            error={"type": "RuntimeError", "message": "échec"},
            updated_at=T0 + timedelta(seconds=1),
        )

        await store.save_job(saved)

        assert (created, replay) == (job, False)
        assert await store.get_job(job.job_id) == saved

    async def test_missing_job_reads_as_none(self, store):
        """An unknown id has no record."""
        assert await store.get_job("nope") is None

    async def test_same_key_and_same_request_returns_the_stored_job_as_replay(
        self, store
    ):
        """A replay returns the first record, including its latest state."""
        job = make_job()
        await store.create_job(job)
        done = replace(job, status=JobState.DONE, result={"ok": True})
        await store.save_job(done)

        replay = await store.create_job(replace(job, job_id="j2"))

        assert replay == (done, True)
        assert await store.get_job("j2") is None

    async def test_same_key_and_different_request_raises_conflict(self, store):
        """A conflicting request does not insert or change a job."""
        job = make_job()
        await store.create_job(job)

        with pytest.raises(IdempotencyConflictError):
            await store.create_job(replace(job, job_id="j2", request_hash="hash-two"))

        assert await store.get_job("j1") == job
        assert await store.get_job("j2") is None

    async def test_key_is_scoped_to_its_tenant(self, store):
        """Two tenants may use the same idempotency key independently."""
        first = make_job()
        second = make_job("j2", "tenant_b")

        assert await store.create_job(first) == (first, False)
        assert await store.create_job(second) == (second, False)
        assert await store.get_job("j1") == first
        assert await store.get_job("j2") == second

    async def test_an_empty_string_key_is_a_set_key(self, store):
        """An empty key still replays and detects conflicts."""
        job = replace(make_job(), idempotency_key="")
        await store.create_job(job)

        assert await store.create_job(replace(job, job_id="j2")) == (job, True)
        with pytest.raises(IdempotencyConflictError):
            await store.create_job(replace(job, job_id="j3", request_hash="other"))
        assert await store.get_job("j2") is None
        assert await store.get_job("j3") is None

    async def test_jobs_without_a_key_never_replay(self, store):
        """Identical requests without keys create independent jobs."""
        first = replace(make_job(), idempotency_key=None)
        second = replace(first, job_id="j2")

        assert await store.create_job(first) == (first, False)
        assert await store.create_job(second) == (second, False)
        assert await store.get_job("j2") == second

    async def test_save_replaces_status_attempts_result_and_error(self, store):
        """Saving a successful retry clears the preceding error."""
        job = make_job()
        await store.create_job(job)
        await store.save_job(replace(job, error={"type": "Error", "message": "x"}))
        done = replace(
            job,
            status=JobState.DONE,
            attempts=3,
            result={"count": 2},
            updated_at=T0 + timedelta(microseconds=1),
        )

        await store.save_job(done)

        assert await store.get_job("j1") == done

    async def test_timestamps_keep_microseconds(self, store):
        """Storage preserves both timestamps without millisecond rounding."""
        job = replace(make_job(), updated_at=T0 + timedelta(microseconds=7))
        await store.create_job(job)

        assert await store.get_job("j1") == job

    async def test_requeue_running_returns_running_jobs_to_queued_oldest_first(
        self, store
    ):
        """Recovery includes queued jobs and preserves creation order for ties."""
        jobs = [
            replace(
                make_job(f"j{i}"),
                idempotency_key=None,
                created_at=T0 + timedelta(seconds=seconds),
            )
            for i, seconds in enumerate([5, -3, -3])
        ]
        for job in jobs:
            await store.create_job(job)
        for job in jobs[:2]:
            await store.save_job(replace(job, status=JobState.RUNNING, attempts=1))

        recovered = await store.requeue_running()

        expected = [replace(jobs[1], attempts=1), jobs[2], replace(jobs[0], attempts=1)]
        assert recovered == expected
        assert [await store.get_job(job.job_id) for job in expected] == expected

    async def test_requeue_leaves_done_and_failed_jobs_alone(self, store):
        """Recovery never reruns a terminal job."""
        jobs = [
            replace(make_job("done"), status=JobState.DONE, idempotency_key=None),
            replace(make_job("failed"), status=JobState.FAILED, idempotency_key=None),
        ]
        for job in jobs:
            await store.create_job(job)

        assert await store.requeue_running() == []
        assert [await store.get_job(job.job_id) for job in jobs] == jobs

    async def test_counts_are_scoped_to_tenants_and_statuses(self, store):
        """Counts ignore unrelated tenants and never count repeated filters twice."""
        for job in [
            replace(make_job("a"), idempotency_key=None),
            replace(make_job("b"), status=JobState.DONE, idempotency_key=None),
            make_job("c", "tenant_b"),
        ]:
            await store.create_job(job)

        assert await store.count_jobs(["tenant_a"], [JobState.QUEUED]) == 1
        assert await store.count_jobs(["tenant_a"], [JobState.DONE]) == 1
        assert (
            await store.count_jobs(
                ["tenant_a", "tenant_a", "tenant_b"],
                [JobState.QUEUED, JobState.QUEUED],
            )
            == 2
        )
        assert await store.count_jobs([], [JobState.QUEUED]) == 0
        assert await store.count_jobs(["tenant_a"], []) == 0
        assert await store.count_jobs(["unknown"], list(JobState)) == 0

    async def test_tenant_ids_are_sorted_and_unique(self, store):
        """Tenant discovery includes terminal jobs and omits duplicates."""
        assert await store.tenant_ids() == []
        for job in [
            make_job("a", "z"),
            make_job("b", "a"),
            replace(make_job("c", "z"), idempotency_key=None, status=JobState.DONE),
        ]:
            await store.create_job(job)

        assert await store.tenant_ids() == ["a", "z"]

    async def test_progress_is_stored_with_the_job(self, store):
        """A job read returns the last progress value written to its record."""
        job = make_job()
        await store.create_job(job)
        progress = JobProgress(stage="index", done=25, total=100)

        await store.set_job_progress(job.job_id, progress)

        stored = await store.get_job(job.job_id)
        assert stored is not None
        assert stored.progress == progress

    async def test_progress_cannot_regress_or_exceed_its_total(self, store):
        """Progress writes reject backwards counts and values past the total."""
        job = make_job()
        await store.create_job(job)
        await store.set_job_progress(
            job.job_id, JobProgress(stage="index", done=20, total=100)
        )

        with pytest.raises(ValueError):
            await store.set_job_progress(
                job.job_id, JobProgress(stage="index", done=19, total=100)
            )
        with pytest.raises(ValueError):
            await store.set_job_progress(
                job.job_id, JobProgress(stage="index", done=30, total=25)
            )

    async def test_trace_id_round_trips_with_the_job(self, store):
        """The stored job keeps the trace id supplied at submission."""
        job = make_job()

        await store.create_job(job)

        stored = await store.get_job(job.job_id)
        assert stored is not None
        assert stored.trace_id == "trace-one"

    async def test_replayed_flag_is_not_stored(self, store):
        """Response-only replay state never becomes part of a stored record."""
        job = make_job()
        await store.create_job(replace(job, replayed=True))
        assert await store.get_job("j1") == job

        await store.save_job(replace(job, replayed=True, attempts=1))

        assert await store.get_job("j1") == replace(job, attempts=1)

    async def test_mutating_payloads_does_not_change_stored_records(self, store):
        """Nested mutable values are detached on writes and reads."""
        job = make_job()
        created, _ = await store.create_job(job)
        job.payload["nested"]["values"].append("caller mutation")
        created.payload["text"] = "changed"
        stored = await store.get_job("j1")
        assert stored == make_job()
        stored.payload["text"] = "read mutation"
        assert await store.get_job("j1") == make_job()

    async def test_saved_results_and_errors_are_detached(self, store):
        """Caller and returned result mutations cannot alter persisted data."""
        job = make_job()
        await store.create_job(job)
        result = {"items": ["original"]}
        saved = replace(
            job,
            result=result,
            error={"type": "RuntimeError", "message": "original"},
        )
        await store.save_job(saved)
        result["items"].append("mutation")
        saved.error["message"] = "changed"

        stored = await store.get_job("j1")

        assert stored.result == {"items": ["original"]}
        assert stored.error == {"type": "RuntimeError", "message": "original"}
        stored.result["items"].append("read mutation")
        stored.error["message"] = "read mutation"
        reread = await store.get_job("j1")
        assert reread.result == {"items": ["original"]}
        assert reread.error == {"type": "RuntimeError", "message": "original"}

    async def test_saving_an_unknown_job_does_not_insert_it(self, store):
        """Save only replaces an existing record."""
        with pytest.raises(KeyError):
            await store.save_job(make_job())

        assert await store.get_job("j1") is None
