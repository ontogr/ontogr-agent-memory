"""Execute persistent jobs in serial tenant lanes on one event loop."""

import asyncio
import hashlib
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
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
from ontogr_mem.relational_store.jobs.errors import UnknownJobOpError
from ontogr_mem.relational_store.jobs.ports import JobStore


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
    return {"type": type(exc).__name__, "message": str(exc)}


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
    ) -> None:
        """Configure persistence, operations, time and the attempt limit.

        Args:
            store: Persistence shared by submission and recovery.
            handlers: Operation names and their async handlers.
            clock: Time source for submission and state changes.
            max_attempts: Positive limit including attempts before a restart.

        Raises:
            ValueError: If max_attempts is not positive.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._store = store
        self._handlers = dict(handlers)
        self._clock = clock
        self._max_attempts = max_attempts
        self._queues: dict[str, deque[str]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._finished: dict[str, asyncio.Event] = {}
        self._blocked: set[str] = set()
        self._lifecycle = asyncio.Lock()

    async def start(self) -> int:
        """Recover queued and interrupted jobs and return the number enqueued.

        If lanes are still active, return zero without changing running rows.
        Drain or stop active lanes before recovering lanes with store failures.
        Repeated calls never enqueue work already owned by an active lane.
        """
        async with self._lifecycle:
            if self._tasks:
                return 0
            queued = await self._store.requeue_running()
            self._blocked.clear()
            for job in queued:
                self._enqueue(job)
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
                self._enqueue(job)
            event = self._finished.get(job.job_id)
        if wait_s > 0 and not job.status.is_terminal:
            if event is not None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), wait_s)
            latest = await self._store.get_job(job.job_id)
            if latest is not None:
                job = latest
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
            tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.clear()
            self._queues.clear()

    def _enqueue(self, job: Job) -> None:
        self._finished.setdefault(job.job_id, asyncio.Event())
        if job.tenant_id in self._blocked:
            return
        queue = self._queues.setdefault(job.tenant_id, deque())
        queue.append(job.job_id)
        if job.tenant_id not in self._tasks:
            self._tasks[job.tenant_id] = asyncio.create_task(
                self._run_lane(job.tenant_id), name=f"job-lane:{job.tenant_id}"
            )

    async def _run_lane(self, tenant_id: str) -> None:
        queue = self._queues[tenant_id]
        try:
            while queue:
                await self._run_one(queue.popleft())
        except Exception:
            self._blocked.add(tenant_id)
            log.exception(
                "job lane for tenant %s stopped after a store failure", tenant_id
            )
        finally:
            # No await separates the empty check from removal, so enqueue cannot
            # append work to a lane that has already decided to exit.
            self._queues.pop(tenant_id, None)
            self._tasks.pop(tenant_id, None)

    async def _run_one(self, job_id: str) -> None:
        job = await self._store.get_job(job_id)
        if job is None or job.status.is_terminal:
            self._finish(job_id)
            return
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
            await self._store.save_job(job)
            try:
                result = await handler(job)
            except AgentMemoryError as exc:
                error = _error_record(exc)
                break
            except Exception as exc:
                log.exception("job %s attempt %d failed", job_id, job.attempts)
                error = _error_record(exc)
                continue
            await self._store.save_job(
                replace(
                    job,
                    status=JobState.DONE,
                    result=result,
                    error=None,
                    updated_at=self._clock.now(),
                )
            )
            self._finish(job_id)
            return
        await self._fail(job, error)

    async def _fail(self, job: Job, error: dict[str, str]) -> None:
        await self._store.save_job(
            replace(
                job,
                status=JobState.FAILED,
                result=None,
                error=error,
                updated_at=self._clock.now(),
            )
        )
        self._finish(job.job_id)

    def _finish(self, job_id: str) -> None:
        event = self._finished.pop(job_id, None)
        if event is not None:
            event.set()
