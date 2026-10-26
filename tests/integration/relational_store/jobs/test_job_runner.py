"""Exercise tenant job ordering, replay, recovery and failures on both stores."""

import asyncio
import hashlib
import sqlite3
from collections.abc import Sequence
from typing import Any

import pytest

from ontogr_mem.data_models.jobs import Job, JobState
from ontogr_mem.data_models.tenant import InvalidTenantError
from ontogr_mem.errors import AgentMemoryError
from ontogr_mem.relational_store.jobs.errors import (
    IdempotencyConflictError,
    UnknownJobOpError,
)
from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.relational_store.jobs.runner import JobRunner
from tests.support.clock import FixedClock


class TestJobRunner:
    """Run handlers through the public runner API and inspect stored outcomes."""

    async def test_a_job_runs_and_stores_its_result(
        self, job_store: StateStore
    ) -> None:
        """A successful submission persists its result, attempt and clock times."""
        clock = FixedClock()

        async def index(job: Job) -> dict[str, Any]:
            return {"count": job.payload["count"]}

        runner = JobRunner(job_store, {"index": index}, clock)
        job = await runner.submit("t", "index", {"count": 2}, wait_s=5)
        await runner.drain()

        assert job.status == JobState.DONE
        assert job.result == {"count": 2}
        assert job.attempts == 1
        assert job.created_at == clock.now()
        assert job.updated_at == clock.now()
        assert await runner.get(job.job_id) == job
        assert await runner.get("missing") is None

    async def test_jobs_of_one_tenant_run_in_submit_order_one_at_a_time(
        self, job_store: StateStore
    ) -> None:
        """One tenant never overlaps handlers or changes submission order."""
        calls: list[str] = []

        async def index(job: Job) -> dict[str, Any]:
            calls.append(f"start {job.payload['n']}")
            await asyncio.sleep(0)
            calls.append(f"end {job.payload['n']}")
            return {}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        for n in (1, 2, 3):
            await runner.submit("t", "index", {"n": n})
        await runner.drain()

        assert calls == ["start 1", "end 1", "start 2", "end 2", "start 3", "end 3"]

    async def test_different_tenants_run_in_parallel(
        self, job_store: StateStore
    ) -> None:
        """A blocked tenant does not prevent another tenant from finishing."""
        entered = asyncio.Event()
        release = asyncio.Event()

        async def index(job: Job) -> dict[str, Any]:
            if job.tenant_id == "a":
                entered.set()
                await release.wait()
            return {"tenant": job.tenant_id}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        try:
            first = await runner.submit("a", "index", {})
            await asyncio.wait_for(entered.wait(), 5)
            second = await runner.submit("b", "index", {}, wait_s=5)
            running = await runner.get(first.job_id)
            assert second.status == JobState.DONE
            assert running is not None and running.status == JobState.RUNNING
        finally:
            release.set()
            await runner.drain()

    @pytest.mark.parametrize(
        ("failure", "failures", "status", "attempts"),
        [
            (RuntimeError("boom"), 2, JobState.DONE, 3),
            (RuntimeError("boom"), 9, JobState.FAILED, 3),
            (AgentMemoryError("invalid"), 9, JobState.FAILED, 1),
            (sqlite3.OperationalError("locked"), 2, JobState.DONE, 3),
        ],
    )
    async def test_retries_only_untyped_handler_errors(
        self,
        job_store: StateStore,
        failure: Exception,
        failures: int,
        status: JobState,
        attempts: int,
    ) -> None:
        """Typed errors fail once while unexpected errors use the attempt budget."""
        calls = 0

        async def index(job: Job) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls <= failures:
                raise failure
            return {"ok": True}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        job = await runner.submit("t", "index", {}, wait_s=5)
        await runner.drain()

        assert job.status == status
        assert job.attempts == attempts
        assert calls == attempts
        if status == JobState.FAILED:
            assert job.error == {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        else:
            assert job.result == {"ok": True}
            assert job.error is None

    async def test_wait_timeout_and_replay_wait_preserve_the_stored_flag(
        self, job_store: StateStore
    ) -> None:
        """Replay waits share one execution and never persist the response flag."""
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def index(job: Job) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return {"ok": True}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        try:
            first = await runner.submit("t", "index", {}, idempotency_key="key")
            await asyncio.wait_for(entered.wait(), 5)
            timed = await runner.submit(
                "t", "index", {}, idempotency_key="key", wait_s=0.01
            )
            assert timed.status == JobState.RUNNING
            assert timed.replayed
            waiter = asyncio.create_task(
                runner.submit("t", "index", {}, idempotency_key="key", wait_s=5)
            )
            await asyncio.sleep(0)
            release.set()
            finished = await waiter
            await runner.drain()
            replay = await runner.submit(
                "t", "index", {}, idempotency_key="key", wait_s=5
            )
            stored = await runner.get(first.job_id)
            assert not first.replayed
            assert finished.status == JobState.DONE and finished.replayed
            assert replay.status == JobState.DONE and replay.replayed
            assert first.job_id == timed.job_id == finished.job_id == replay.job_id
            assert stored is not None and not stored.replayed
            assert calls == 1
        finally:
            release.set()
            await runner.stop()

    async def test_first_submit_wait_times_out_without_cancelling_the_handler(
        self, job_store: StateStore
    ) -> None:
        """A wait timeout returns unfinished work without interrupting execution."""
        release = asyncio.Event()

        async def index(job: Job) -> dict[str, Any]:
            await release.wait()
            return {"ok": True}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        try:
            job = await runner.submit("t", "index", {}, wait_s=0.02)
            assert job.status in (JobState.QUEUED, JobState.RUNNING)
            assert not job.replayed
            release.set()
            await runner.drain()
            stored = await runner.get(job.job_id)
            assert stored is not None and stored.status == JobState.DONE
        finally:
            release.set()
            await runner.stop()

    async def test_hash_is_canonical_and_key_conflicts_do_not_enqueue(
        self, job_store: StateStore
    ) -> None:
        """Canonical Unicode JSON ignores key order but detects changed requests."""
        calls: list[str] = []

        async def index(job: Job) -> dict[str, Any]:
            calls.append(job.job_id)
            return {}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        first = await runner.submit(
            "t", "index", {"z": "é", "a": [1]}, idempotency_key="key", wait_s=5
        )
        replay = await runner.submit(
            "t", "index", {"a": [1], "z": "é"}, idempotency_key="key"
        )
        with pytest.raises(IdempotencyConflictError):
            await runner.submit("t", "index", {"a": [2]}, idempotency_key="key")
        await runner.drain()

        canonical = '{"op":"index","payload":{"a":[1],"z":"é"}}'
        assert first.request_hash == hashlib.sha256(canonical.encode()).hexdigest()
        assert replay.job_id == first.job_id and replay.replayed
        assert calls == [first.job_id]

    async def test_unknown_ops_and_invalid_tenants_are_not_stored(
        self, job_store: StateStore
    ) -> None:
        """Submission validation fails before persistence or lane creation."""
        runner = JobRunner(job_store, {}, FixedClock())
        with pytest.raises(UnknownJobOpError):
            await runner.submit("t", "missing", {})
        with pytest.raises(InvalidTenantError):
            await runner.submit("invalid tenant", "missing", {})
        await runner.drain()
        assert await job_store.tenant_ids() == []

    async def test_finished_jobs_survive_runner_restart_and_lane_recreation(
        self, job_store: StateStore
    ) -> None:
        """Idle lanes accept new work and completed jobs do not rerun on restart."""
        calls: list[int] = []

        async def index(job: Job) -> dict[str, Any]:
            calls.append(job.payload["n"])
            return {"n": job.payload["n"]}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        jobs = []
        for n in range(8):
            jobs.append(await runner.submit("t", "index", {"n": n}, wait_s=5))
        await runner.drain()
        restarted = JobRunner(job_store, {"index": index}, FixedClock())
        assert await restarted.start() == 0
        await restarted.drain()
        assert await restarted.get(jobs[0].job_id) == jobs[0]
        assert calls == list(range(8))

    async def test_stop_and_repeated_start_do_not_duplicate_pending_jobs(
        self, job_store: StateStore
    ) -> None:
        """Recovery resumes interrupted work once before the remaining queued work."""
        entered = asyncio.Event()
        release = asyncio.Event()
        calls: list[int] = []

        async def index(job: Job) -> dict[str, Any]:
            calls.append(job.payload["n"])
            if job.payload["n"] == 1 and job.attempts == 1:
                entered.set()
                await release.wait()
            return {}

        runner = JobRunner(job_store, {"index": index}, FixedClock())
        try:
            first = await runner.submit("t", "index", {"n": 1})
            await asyncio.wait_for(entered.wait(), 5)
            second = await runner.submit("t", "index", {"n": 2})
            assert await runner.start() == 0
            await runner.stop()
            interrupted = await runner.get(first.job_id)
            assert interrupted is not None
            assert interrupted.status == JobState.RUNNING
            assert await runner.start() == 2
            await runner.start()
            await asyncio.wait_for(runner.drain(), 5)
            recovered = await runner.get(first.job_id)
            queued = await runner.get(second.job_id)
            assert recovered is not None and recovered.attempts == 2
            assert queued is not None and queued.status == JobState.DONE
            assert calls == [1, 1, 2]
        finally:
            release.set()
            await runner.stop()

    async def test_recovered_unknown_op_fails_without_an_attempt(
        self, job_store: StateStore
    ) -> None:
        """A removed handler produces a typed stored failure during recovery."""
        now = FixedClock().now()
        job = Job(
            job_id="unknown",
            tenant_id="t",
            op="removed",
            payload={},
            request_hash="hash",
            status=JobState.RUNNING,
            attempts=1,
            created_at=now,
            updated_at=now,
        )
        await job_store.create_job(job)
        runner = JobRunner(job_store, {}, FixedClock())
        assert await runner.start() == 1
        await runner.drain()
        failed = await runner.get(job.job_id)
        assert failed is not None and failed.status == JobState.FAILED
        assert failed.attempts == 1
        assert failed.error is not None
        assert failed.error["type"] == "UnknownJobOpError"

    @pytest.mark.parametrize("method", ["get_job", "save_job"])
    async def test_store_failure_stops_only_its_lane_until_restart(
        self, job_store: StateStore, method: str
    ) -> None:
        """Storage failures suspend a tenant without losing its pending jobs."""
        entered = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        class FailingStore:
            """Inject a storage outage without changing handler behavior."""

            broken = False

            async def create_job(self, job: Job) -> tuple[Job, bool]:
                """Create a job through the real adapter."""
                return await job_store.create_job(job)

            async def get_job(self, job_id: str) -> Job | None:
                """Read a job unless its tenant has a simulated outage."""
                job = await job_store.get_job(job_id)
                if self.broken and method == "get_job" and job and job.tenant_id == "a":
                    raise sqlite3.OperationalError("storage outage")
                return job

            async def save_job(self, job: Job) -> None:
                """Save a job unless its tenant has a simulated outage."""
                if self.broken and method == "save_job" and job.tenant_id == "a":
                    raise sqlite3.OperationalError("storage outage")
                await job_store.save_job(job)

            async def requeue_running(self) -> list[Job]:
                """Recover jobs through the real adapter."""
                return await job_store.requeue_running()

            async def count_jobs(
                self, tenant_ids: Sequence[str], statuses: Sequence[JobState]
            ) -> int:
                """Count jobs through the real adapter."""
                return await job_store.count_jobs(tenant_ids, statuses)

            async def tenant_ids(self) -> list[str]:
                """List tenants through the real adapter."""
                return await job_store.tenant_ids()

        store = FailingStore()

        async def index(job: Job) -> dict[str, Any]:
            calls.append(job.payload["name"])
            if job.payload["name"] == "first" and job.attempts == 1:
                entered.set()
                await release.wait()
            return {}

        runner = JobRunner(store, {"index": index}, FixedClock())
        try:
            await runner.submit("a", "index", {"name": "first"})
            await asyncio.wait_for(entered.wait(), 5)
            pending = await runner.submit("a", "index", {"name": "pending"})
            store.broken = True
            release.set()
            await asyncio.wait_for(runner.drain(), 5)
            assert "pending" not in calls
            other = await runner.submit("b", "index", {"name": "other"}, wait_s=5)
            assert other.status == JobState.DONE
            await runner.submit("a", "index", {"name": "later"})
            await runner.drain()
            assert "later" not in calls
            stored = await job_store.get_job(pending.job_id)
            assert stored is not None and stored.status == JobState.QUEUED
            store.broken = False
            assert await runner.start() >= 2
            await asyncio.wait_for(runner.drain(), 5)
            assert calls.count("pending") == 1
            assert calls.count("later") == 1
        finally:
            release.set()
            await runner.stop()

    @pytest.mark.parametrize("max_attempts", [0, -1])
    async def test_attempt_limit_must_be_positive(
        self, job_store: StateStore, max_attempts: int
    ) -> None:
        """A nonpositive attempt budget is refused before any job can be queued."""
        with pytest.raises(ValueError, match="max_attempts"):
            JobRunner(job_store, {}, FixedClock(), max_attempts=max_attempts)
