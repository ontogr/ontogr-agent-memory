"""Queued memory writes and their operation contexts."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ontogr_mem.data_models.jobs import Job
from ontogr_mem.errors import AgentMemoryError
from ontogr_mem.relational_store import (
    IdempotencyConflictError,
    OperationKind,
)
from ontogr_mem.relational_store.jobs.runner import JobRunner
from ontogr_mem.request_context import RequestContext
from ontogr_mem.service.contracts import (
    AddMemory,
    ConsolidateMemory,
    ErrorInfo,
    InvalidateMemory,
    JobView,
    UpdateMemory,
    WriteReceipt,
)
from ontogr_mem.service.dependencies import ServiceDeps
from ontogr_mem.service.errors import (
    ConflictError,
    ErrorCode,
    InvalidArgumentError,
    LLMUnavailableError,
    ObjectNotFoundError,
)
from ontogr_mem.service.operation_context import OperationRecorder
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.service.writes import add_memory as write_memory
from ontogr_mem.service.writes import invalidate_memory as write_invalidation
from ontogr_mem.service.writes import run_consolidation
from ontogr_mem.service.writes import update_memory as write_update


async def _consolidate(
    runtime: TenantRuntime, job: Job, payload: dict[str, Any]
) -> dict[str, Any]:
    """Run consolidation through the shared job execution boundary."""
    return await run_consolidation(runtime)


def _job_view(job: Job) -> JobView:
    """Convert a stored job to its public response."""
    return JobView(
        job_id=job.job_id,
        tenant=job.tenant_id,
        op=job.op,
        status=job.status,
        attempts=job.attempts,
        result=job.result,
        error=_error_info(job.error),
        trace_id=job.trace_id,
        progress=job.progress,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _error_info(error: Mapping[str, str] | None) -> ErrorInfo | None:
    """Map a stored exception record to a stable service error response."""
    if error is None:
        return None
    code_by_type = {
        "InvalidArgumentError": ErrorCode.INVALID_ARGUMENT,
        "ObjectNotFoundError": ErrorCode.NOT_FOUND,
        "ConflictError": ErrorCode.CONFLICT,
        "LLMUnavailableError": ErrorCode.LLM_UNAVAILABLE,
        "RateLimitedError": ErrorCode.RATE_LIMITED,
        "PayloadTooLargeError": ErrorCode.PAYLOAD_TOO_LARGE,
        "ConfirmationRequiredError": ErrorCode.CONFIRMATION_REQUIRED,
    }
    hints = {"LLMUnavailableError": "Retry, or write a structured fact with kind=fact."}
    return ErrorInfo(
        code=code_by_type.get(error.get("type", ""), ErrorCode.INTERNAL),
        message=error.get("message", "job failed"),
        hint=hints.get(error.get("type", ""), ""),
    )


@dataclass(frozen=True, slots=True)
class JobContext:
    """Submission context retained while a job is running or being retried."""

    parent_id: str | None
    trace_id: str
    session_id: str | None


class ServiceJobs:
    """Authorize queued writes and own their execution and retry contexts."""

    def __init__(
        self, deps: ServiceDeps, recorder: OperationRecorder, write_lock: asyncio.Lock
    ) -> None:
        """Bind service dependencies and the lock shared with tenant clearing."""
        self._deps = deps
        self._recorder = recorder
        self._write_lock = write_lock
        self._job_contexts: dict[str, JobContext] = {}
        self._runner = JobRunner(
            deps.state,
            {
                "add_memory": self._run_add_memory,
                "update_memory": self._run_update_memory,
                "invalidate_memory": self._run_invalidate_memory,
                "consolidate_memory": self._run_consolidate_memory,
            },
            deps.clock,
            max_attempts=deps.max_attempts,
        )

    async def start(self) -> int:
        """Start durable tenant job lanes after recovery has completed."""
        return await self._runner.start()

    async def add_memory(
        self,
        principal: Principal,
        request: AddMemory,
        *,
        trace_id: str | None = None,
    ) -> WriteReceipt:
        """Authorize and queue a structured, text, or message memory write.

        A text or message write requires an injected language model. A positive
        ``wait_seconds`` waits up to that long for the job to finish.
        """
        self._deps.limiter.check(principal.id, "write")
        tenant_id = request.tenant or principal.default_tenant
        async with self._recorder.track(
            OperationKind.WRITE,
            principal.id,
            tenant_id,
            session_id=request.session_id,
            trace_id=trace_id,
        ) as context:
            tenant_id = await self._deps.tenants.require(
                principal, tenant_id, Grant.WRITE, "memory.add"
            )
            if request.kind.value != "fact" and self._deps.llm is None:
                raise LLMUnavailableError(
                    "a language model is required for text memory",
                    hint="Retry, or write a structured fact with kind=fact.",
                )
            await self._deps.tenants.get(tenant_id, create=True)
            return await self.submit(
                "add_memory", principal, request, tenant_id, context
            )

    async def update_memory(
        self,
        principal: Principal,
        request: UpdateMemory,
        *,
        trace_id: str | None = None,
    ) -> WriteReceipt:
        """Authorize and queue a replacement for one current fact.

        The job records a stated end for the old fact and asserts its new value
        in one episode.
        """
        self._deps.limiter.check(principal.id, "write")
        tenant_id = request.tenant or principal.default_tenant
        async with self._recorder.track(
            OperationKind.WRITE, principal.id, tenant_id, trace_id=trace_id
        ) as context:
            tenant_id = await self._deps.tenants.require(
                principal,
                tenant_id,
                Grant.WRITE,
                "memory.update",
                target=request.fact_id,
            )
            if not request.object:
                raise InvalidArgumentError("update needs a new object")
            await self._deps.tenants.get(tenant_id, create=True)
            return await self.submit(
                "update_memory", principal, request, tenant_id, context
            )

    async def invalidate(
        self,
        principal: Principal,
        request: InvalidateMemory,
        *,
        trace_id: str | None = None,
    ) -> WriteReceipt:
        """Close a current fact while preserving its historical version."""
        self._deps.limiter.check(principal.id, "write")
        tenant_id = request.tenant or principal.default_tenant
        async with self._recorder.track(
            OperationKind.INVALIDATE, principal.id, tenant_id, trace_id=trace_id
        ) as context:
            tenant_id = await self._deps.tenants.require(
                principal,
                tenant_id,
                Grant.FORGET,
                "memory.invalidate",
                target=request.fact_id,
            )
            await self._deps.tenants.get(tenant_id, create=True)
            return await self.submit(
                "invalidate_memory", principal, request, tenant_id, context
            )

    async def consolidate_memory(
        self, principal: Principal, request: ConsolidateMemory
    ) -> WriteReceipt:
        """Authorize and queue a consolidation run for one tenant."""
        self._deps.limiter.check(principal.id, "write")
        tenant_id = request.tenant or principal.default_tenant
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.WRITE, "memory.consolidate"
        )
        async with self._recorder.track(
            OperationKind.WRITE, principal.id, tenant_id
        ) as context:
            await self._deps.tenants.get(tenant_id, create=True)
            async with self._write_lock:
                job = await self._runner.submit(
                    tenant_id,
                    "consolidate_memory",
                    {"principal_id": principal.id},
                    wait_s=0.0,
                    trace_id=context.trace_id,
                )
            self._job_contexts[job.job_id] = JobContext(
                context.operation_id,
                context.trace_id,
                None,
            )
            job = await self._wait_for_job(job, request.wait_seconds)
            return WriteReceipt(
                job_id=job.job_id,
                status=job.status,
                tenant=tenant_id,
                idempotent_replay=job.replayed,
                trace_id=context.trace_id,
                result=job.result,
                error=_error_info(job.error),
            )

    async def get_job(
        self, principal: Principal, job_id: str, *, trace_id: str | None = None
    ) -> JobView:
        """Read one job after checking its tenant grant.

        Raises:
            ObjectNotFoundError: If the job is missing or not visible.
        """
        self._deps.limiter.check(principal.id, "read")
        job = await self._runner.get(job_id)
        tenant_id = job.tenant_id if job is not None else None
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, tenant_id, trace_id=trace_id
        ):
            if job is None:
                raise ObjectNotFoundError("job not found")
            await self._deps.tenants.require(
                principal,
                job.tenant_id,
                Grant.READ,
                "memory.job",
                target=job_id,
            )
            return _job_view(job)

    async def drain(self) -> None:
        """Wait until all queued tenant lanes finish."""
        await self._runner.drain()

    async def stop(self) -> None:
        """Stop active job lanes without closing injected stores."""
        await self._runner.stop()

    async def submit(
        self,
        op: str,
        principal: Principal,
        request: AddMemory | UpdateMemory | InvalidateMemory,
        tenant_id: str,
        context: RequestContext,
    ) -> WriteReceipt:
        """Submit one request and shape the immediate receipt."""
        async with self._write_lock:
            try:
                job = await self._runner.submit(
                    tenant_id,
                    op,
                    {
                        "principal_id": principal.id,
                        "request": request.model_dump(mode="json"),
                    },
                    idempotency_key=request.idempotency_key,
                    wait_s=0.0,
                    trace_id=context.trace_id,
                )
            except IdempotencyConflictError as exc:
                raise ConflictError(str(exc)) from exc
        replayed = job.replayed
        if not job.replayed:
            self._job_contexts[job.job_id] = JobContext(
                context.operation_id,
                context.trace_id,
                request.session_id if isinstance(request, AddMemory) else None,
            )
        job = await self._wait_for_job(job, request.wait_seconds)
        return WriteReceipt(
            job_id=job.job_id,
            status=job.status,
            tenant=tenant_id,
            idempotent_replay=replayed,
            trace_id=context.trace_id,
            result=job.result,
            error=_error_info(job.error),
        )

    async def _wait_for_job(self, job: Job, wait_seconds: float) -> Job:
        """Return the latest job state, waiting no longer than requested."""
        if wait_seconds <= 0 or job.status.is_terminal:
            return job
        deadline = asyncio.get_running_loop().time() + wait_seconds
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(min(0.01, deadline - asyncio.get_running_loop().time()))
            latest = await self._runner.get(job.job_id)
            if latest is not None:
                job = latest
            if job.status.is_terminal:
                break
        return job

    @asynccontextmanager
    async def _job_operation(self, job: Job) -> AsyncGenerator[RequestContext, None]:
        """Bind the job's operation to the submission row when available."""
        submission = self._job_contexts.get(
            job.job_id, JobContext(None, job.trace_id, None)
        )
        kind = (
            OperationKind.INVALIDATE
            if job.op == "invalidate_memory"
            else OperationKind.WRITE
        )
        async with self._recorder.track(
            kind,
            str(job.payload["principal_id"]),
            job.tenant_id,
            parent_id=submission.parent_id,
            session_id=submission.session_id,
            trace_id=submission.trace_id,
        ) as context:
            yield context

    async def _run_job(
        self,
        job: Job,
        handler: Callable[
            [TenantRuntime, Job, dict[str, Any]], Awaitable[dict[str, Any]]
        ],
    ) -> dict[str, Any]:
        """Run a child operation and keep its context only for retryable failures."""
        try:
            async with self._job_operation(job):
                runtime = await self._deps.tenants.get_or_create(job.tenant_id)
                result = await handler(runtime, job, job.payload)
        except AgentMemoryError:
            self._job_contexts.pop(job.job_id, None)
            raise
        except Exception:
            if job.attempts >= self._deps.max_attempts:
                self._job_contexts.pop(job.job_id, None)
            raise
        self._job_contexts.pop(job.job_id, None)
        return result

    async def _run_add_memory(self, job: Job) -> dict[str, Any]:
        """Run the queued add_memory operation."""
        return await self._run_job(job, write_memory)

    async def _run_update_memory(self, job: Job) -> dict[str, Any]:
        """Run the queued update_memory operation."""
        return await self._run_job(job, write_update)

    async def _run_invalidate_memory(self, job: Job) -> dict[str, Any]:
        """Run the queued invalidate_memory operation."""
        return await self._run_job(job, write_invalidation)

    async def _run_consolidate_memory(self, job: Job) -> dict[str, Any]:
        """Run the queued consolidate_memory operation."""
        return await self._run_job(job, _consolidate)
