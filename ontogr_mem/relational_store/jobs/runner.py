"""Execute persistent jobs in serial tenant lanes on one event loop."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from typing import Any
from uuid import uuid4

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.jobs import Job, JobState
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.errors import AgentMemoryError
from ontogr_mem.observability.ids import accept_trace_id
from ontogr_mem.observability.redact import redact_text
from ontogr_mem.relational_store.jobs.errors import LeaseLostError, UnknownJobOpError
from ontogr_mem.relational_store.jobs.ports import JobStore, LeaseHolder


log = logging.getLogger(__name__)
Handler = Callable[[Job], Awaitable[dict[str, Any]]]


def _request_hash(op: str, payload: Mapping[str, Any]) -> str:
    """Hash the operation and payload as canonical UTF-8 JSON."""
    encoded = json.dumps(
        {"op": op, "payload": dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error_record(exc: Exception) -> dict[str, str]:
    """Return the caller-visible type and message of a handler error."""
    return {"type": type(exc).__name__, "message": redact_text(str(exc))}


class JobRunner:
    """Run one job at a time per Tenant, with independent lanes across Tenants.

    Use one runner per store on one event loop. Handlers must tolerate replay
    after interruption. Store failures stop the affected lane until recovery.
    The runner does not own or close the store.
    """

    def __init__(
        self,
        store: JobStore,
        handlers: Mapping[str, Handler],
        clock: Clock,
        *,
        max_attempts: int = 3,
        heartbeat_seconds: float | None = None,
        lease_holders: Sequence[LeaseHolder] = (),
        can_run: Callable[[Job], bool] | None = None,
        prepare_tenant: Callable[[str], Awaitable[object]] | None = None,
        recover_tenant: Callable[[str], Awaitable[object]] | None = None,
    ) -> None:
        """Configure persistence, operations, time and the attempt limit.

        Args:
            store: Persistence shared by submission and recovery.
            handlers: Operation names and their async handlers.
            clock: Time source for submission and state changes.
            max_attempts: Positive limit including attempts before a restart.
            heartbeat_seconds: Optional interval for renewing hosted leases.
            lease_holders: Stores whose leases this runner renews and releases.
            can_run: Filter for jobs this process can execute.
            prepare_tenant: Open a tenant runtime before claiming its jobs.
            recover_tenant: Recover tenant state during heartbeat ticks.

        Raises:
            ValueError: If max_attempts is not positive.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._store = store
        self._handlers = dict(handlers)
        self._clock = clock
        self._max_attempts = max_attempts
        self._heartbeat_seconds = heartbeat_seconds
        self._lease_holders = tuple(lease_holders)
        self._can_run = can_run or (lambda job: True)
        self._prepare_tenant = prepare_tenant
        self._recover_tenant = recover_tenant
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._finished: dict[str, asyncio.Event] = {}
        self._blocked: set[str] = set()
        self._lifecycle = asyncio.Lock()
        self._orphans: set[str] = set()
        self._started = False
        self._stopping = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    def set_recovery_callback(
        self, callback: Callable[[str], Awaitable[object]]
    ) -> None:
        """Set tenant recovery after service collaborators are assembled."""
        self._recover_tenant = callback

    async def start(self) -> int:
        """Recover queued and interrupted jobs and return the number enqueued.

        If lanes are still active, return zero without changing running rows.
        Drain or stop active lanes before recovering lanes with store failures.
        Repeated calls never enqueue work already owned by an active lane.
        """
        async with self._lifecycle:
            if self._tasks:
                return 0
            await self.start_heartbeat()
            self._started = True
            self._blocked.clear()
            await self._store.requeue_expired(self._clock.now())
            queued = await self._store.queued_jobs()
            for tenant_id in dict.fromkeys(job.tenant_id for job in queued):
                self._start_lane(tenant_id)
            return len(queued)

    async def submit(
        self,
        tenant_id: str,
        op: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        wait_s: float = 0.0,
        trace_id: str | None = None,
    ) -> Job:
        """Persist a submission, enqueue new work and optionally wait for its result.

        Args:
            tenant_id: Tenant that owns the job.
            op: Registered operation name.
            payload: JSON-compatible arguments for the handler.
            idempotency_key: Optional request key scoped to the Tenant.
            wait_s: Maximum seconds to wait; nonpositive values return immediately.
            trace_id: Accepted request trace id, or a fresh id when absent/invalid.

        Returns:
            The latest job when finished or timed out. Idempotent requests set
            replayed on the returned copy only and never enqueue duplicate work.

        Raises:
            InvalidTenantError: If tenant_id is invalid.
            UnknownJobOpError: If op has no handler.
            IdempotencyConflictError: If the key names a different request.
        """
        validate_tenant_id(tenant_id)
        if op not in self._handlers:
            raise UnknownJobOpError(f"no handler for op {op!r}")
        arguments = deepcopy(dict(payload))
        now = self._clock.now()
        candidate = Job(
            job_id=uuid4().hex,
            tenant_id=tenant_id,
            op=op,
            payload=arguments,
            request_hash=_request_hash(op, arguments),
            status=JobState.QUEUED,
            attempts=0,
            created_at=now,
            updated_at=now,
            trace_id=accept_trace_id(trace_id),
            idempotency_key=idempotency_key,
        )
        async with self._lifecycle:
            job, replayed = await self._store.create_job(candidate)
            if not replayed:
                self._finished.setdefault(job.job_id, asyncio.Event())
                await self.start_heartbeat()
                self._start_lane(job.tenant_id)
            event = self._finished.get(job.job_id)
        if wait_s > 0 and not job.status.is_terminal:
            if event is not None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), wait_s)
            latest = await self._store.get_job(job.job_id)
            if latest is not None:
                job = latest
            if event is not None:
                self._finished.pop(job.job_id, None)
        return replace(job, replayed=replayed)

    async def get(self, job_id: str) -> Job | None:
        """Return the stored job, or None when the id is unknown."""
        return await self._store.get_job(job_id)

    async def drain(self) -> None:
        """Wait for active lanes to end, including lanes stopped by store errors."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()))

    async def stop(self) -> None:
        """Cancel active lanes without changing persisted job states.

        A subsequent start recovers interrupted and queued work. Stop does not
        close the store or prohibit future submissions.
        """
        async with self._lifecycle:
            self._stopping = True
            try:
                if self._heartbeat_task is not None:
                    self._heartbeat_task.cancel()
                    await asyncio.gather(self._heartbeat_task, return_exceptions=True)
                    self._heartbeat_task = None
                tasks = list(self._tasks.values())
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._tasks.clear()
                if self._heartbeat_seconds is not None:
                    for holder in self._lease_holders:
                        await holder.release_leases()
            finally:
                self._stopping = False

    async def start_heartbeat(self) -> None:
        """Start lease renewal and queued-work sweeps once."""
        if self._heartbeat_seconds is None or self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat(), name="job-heartbeat"
        )

    async def _heartbeat(self) -> None:
        heartbeat_seconds = self._heartbeat_seconds
        if heartbeat_seconds is None:
            return
        while True:
            await asyncio.sleep(heartbeat_seconds)
            try:
                for job_id in tuple(self._orphans):
                    await self._store.release_job(job_id)
                    self._orphans.discard(job_id)
                now = self._clock.now()
                for holder in self._lease_holders:
                    await holder.renew_leases(now)
                if self._started and not self._stopping:
                    if self._recover_tenant is not None:
                        for tenant_id in await self._store.tenant_ids():
                            await self._recover_tenant(tenant_id)
                    await self._sweep()
            except Exception:
                log.exception("job heartbeat failed")

    async def _sweep(self) -> None:
        await self._store.requeue_expired(self._clock.now())
        for tenant_id in await self._store.tenant_ids():
            if tenant_id not in self._blocked:
                self._start_lane(tenant_id)

    def _start_lane(self, tenant_id: str) -> None:
        if tenant_id not in self._tasks and tenant_id not in self._blocked:
            self._tasks[tenant_id] = asyncio.create_task(
                self._run_lane(tenant_id), name=f"job-lane:{tenant_id}"
            )

    async def _run_lane(self, tenant_id: str) -> None:
        claimed_id: str | None = None
        try:
            while True:
                if self._prepare_tenant is not None:
                    try:
                        await self._prepare_tenant(tenant_id)
                    except Exception:
                        log.exception("tenant preparation failed for %s", tenant_id)
                        return
                job = await self._store.claim_next(
                    tenant_id, self._clock.now(), self._can_run
                )
                if job is None:
                    return
                claimed_id = job.job_id
                await self._run_claimed(job)
                claimed_id = None
        except Exception:
            if claimed_id is not None:
                self._orphans.add(claimed_id)
            self._blocked.add(tenant_id)
            log.exception(
                "job lane for tenant %s stopped after a store failure", tenant_id
            )
        finally:
            # No await separates the empty check from removal, so enqueue cannot
            # append work to a lane that has already decided to exit.
            self._tasks.pop(tenant_id, None)

    async def _run_claimed(self, job: Job) -> None:
        job_id = job.job_id
        handler = self._handlers.get(job.op)
        if handler is None:
            await self._fail(
                job, _error_record(UnknownJobOpError(f"no handler for op {job.op!r}"))
            )
            return
        error = {
            "type": "Interrupted",
            "message": "the job was interrupted before it finished",
        }
        while job.attempts < self._max_attempts:
            job = replace(
                job,
                status=JobState.RUNNING,
                attempts=job.attempts + 1,
                updated_at=self._clock.now(),
                result=None,
                error=None,
            )
            if not await self._store.save_job(job):
                self._finish(job_id)
                return
            try:
                result = await handler(job)
            except LeaseLostError:
                self._finish(job_id)
                return
            except AgentMemoryError as exc:
                error = _error_record(exc)
                break
            except Exception as exc:
                log.exception("job %s attempt %d failed", job_id, job.attempts)
                error = _error_record(exc)
                continue
            saved = await self._store.save_job(
                replace(
                    job,
                    status=JobState.DONE,
                    result=result,
                    error=None,
                    updated_at=self._clock.now(),
                )
            )
            if not saved:
                self._finish(job_id)
                return
            self._finish(job_id)
            return
        await self._fail(job, error)

    async def _fail(self, job: Job, error: dict[str, str]) -> None:
        saved = await self._store.save_job(
            replace(
                job,
                status=JobState.FAILED,
                result=None,
                error=error,
                updated_at=self._clock.now(),
            )
        )
        if not saved:
            self._finish(job.job_id)
            return
        self._finish(job.job_id)

    def _finish(self, job_id: str) -> None:
        event = self._finished.pop(job_id, None)
        if event is not None:
            event.set()
