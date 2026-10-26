"""The async service boundary for tenant-scoped memory verbs."""

import asyncio
from collections.abc import Sequence
from datetime import datetime

from ontogr_mem.export import TenantExport
from ontogr_mem.relational_store import (
    OperationRecord,
)
from ontogr_mem.relational_store.audit import (
    AuditRow,
)
from ontogr_mem.retrieval.neighbourhood import Neighbourhood
from ontogr_mem.service.contracts import (
    AddMemory,
    ClearTenant,
    ConsolidateMemory,
    EntityView,
    EpisodePage,
    ForgetMemory,
    ForgetMode,
    GetEntity,
    InvalidateMemory,
    JobView,
    ListEpisodes,
    MemoryContext,
    MemoryFeedback,
    SearchMemory,
    StatusOut,
    UpdateMemory,
    WriteReceipt,
)
from ontogr_mem.service.contracts import (
    ForgetResult as ForgetContractResult,
)
from ontogr_mem.service.dependencies import ServiceDeps
from ontogr_mem.service.forget.results import (
    ClearTenantPreview,
    ForgetPreview,
    ForgetResult,
)
from ontogr_mem.service.forget.selector import ForgetSelector
from ontogr_mem.service.forget.service import ForgetService, RecoveryReport
from ontogr_mem.service.jobs import ServiceJobs
from ontogr_mem.service.operation_context import OperationRecorder
from ontogr_mem.service.principal import Principal
from ontogr_mem.service.queries import ServiceQueries
from ontogr_mem.summaries import EntityCard
from ontogr_mem.temporal.lineage import FactHistory


class MemoryService:
    """Async facade over tenant reads, queued writes, and resumable removal."""

    def __init__(self, deps: ServiceDeps) -> None:
        """Use dependencies assembled by the service bootstrap."""
        self._write_lock = asyncio.Lock()
        recorder = OperationRecorder(deps.operations, deps.tracer)
        self._jobs = ServiceJobs(deps, recorder, self._write_lock)
        self._queries = ServiceQueries(deps, recorder)
        self._forget = ForgetService(deps, recorder, self._jobs, self._write_lock)

    async def add_memory(
        self,
        principal: Principal,
        request: AddMemory,
        *,
        trace_id: str | None = None,
    ) -> WriteReceipt:
        """Authorize and queue a structured, text, or message memory write."""
        return await self._jobs.add_memory(principal, request, trace_id=trace_id)

    async def update_memory(
        self,
        principal: Principal,
        request: UpdateMemory,
        *,
        trace_id: str | None = None,
    ) -> WriteReceipt:
        """Authorize and queue a replacement for one current fact."""
        return await self._jobs.update_memory(principal, request, trace_id=trace_id)

    async def invalidate(
        self,
        principal: Principal,
        request: InvalidateMemory,
        *,
        trace_id: str | None = None,
    ) -> WriteReceipt:
        """Close a current fact while preserving its historical version."""
        return await self._jobs.invalidate(principal, request, trace_id=trace_id)

    async def preview_forget(
        self,
        principal: Principal,
        *,
        tenant: str,
        selector: ForgetSelector,
        mode: ForgetMode,
        trace_id: str | None = None,
    ) -> ForgetPreview:
        """Resolve a forget target and return its short-lived confirmation."""
        return await self._forget.preview_forget(
            principal, tenant=tenant, selector=selector, mode=mode, trace_id=trace_id
        )

    async def forget(
        self,
        principal: Principal,
        *,
        tenant: str,
        selector: ForgetSelector,
        mode: ForgetMode,
        confirm_token: str | None = None,
        reason: str = "",
        wait_seconds: float = 0.0,
        trace_id: str | None = None,
    ) -> ForgetResult | WriteReceipt:
        """Invalidate a fact, or verify the confirmation token and run a hard forget."""
        return await self._forget.forget(
            principal,
            tenant=tenant,
            selector=selector,
            mode=mode,
            confirm_token=confirm_token,
            reason=reason,
            wait_seconds=wait_seconds,
            trace_id=trace_id,
        )

    async def forget_memory(
        self,
        principal: Principal,
        request: ForgetMemory,
        *,
        trace_id: str | None = None,
    ) -> ForgetContractResult:
        """Run a forget request through the selector-based forget operations."""
        return await self._forget.forget_memory(principal, request, trace_id=trace_id)

    async def preview_clear_tenant(
        self,
        principal: Principal,
        tenant: str,
        *,
        trace_id: str | None = None,
    ) -> ClearTenantPreview:
        """Preview tenant-wide removal under the admin grant."""
        return await self._forget.preview_clear_tenant(
            principal, tenant, trace_id=trace_id
        )

    async def clear_tenant(
        self,
        principal: Principal,
        request: ClearTenant,
        *,
        trace_id: str | None = None,
    ) -> ForgetContractResult:
        """Confirm a tenant-wide clear and remove memory, vectors, and jobs."""
        return await self._forget.clear_tenant(principal, request, trace_id=trace_id)

    async def search_memory(
        self,
        principal: Principal,
        request: SearchMemory,
        *,
        trace_id: str | None = None,
    ) -> MemoryContext:
        """Search one or more authorized tenants inline."""
        return await self._queries.search_memory(principal, request, trace_id=trace_id)

    async def get_entity(
        self,
        principal: Principal,
        request: GetEntity,
        *,
        trace_id: str | None = None,
    ) -> EntityView:
        """Read one canonical entity and its current or historical facts."""
        return await self._queries.get_entity(principal, request, trace_id=trace_id)

    async def list_episodes(
        self,
        principal: Principal,
        request: ListEpisodes,
        *,
        trace_id: str | None = None,
    ) -> EpisodePage:
        """Read one newest-first page of episode metadata."""
        return await self._queries.list_episodes(principal, request, trace_id=trace_id)

    async def memory_feedback(
        self, principal: Principal, request: MemoryFeedback
    ) -> dict[str, str | float | int]:
        """Record a caller rating for an entity or a fact's endpoints."""
        return await self._queries.memory_feedback(principal, request)

    async def consolidate_memory(
        self, principal: Principal, request: ConsolidateMemory
    ) -> WriteReceipt:
        """Authorize and queue a consolidation run for one tenant."""
        return await self._jobs.consolidate_memory(principal, request)

    async def history(
        self,
        principal: Principal,
        tenant_id: str,
        fact_id: str,
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
    ) -> FactHistory:
        """Read every visible version of a fact's temporal slot."""
        return await self._queries.history(
            principal, tenant_id, fact_id, valid_at=valid_at, known_at=known_at
        )

    async def entity(
        self,
        principal: Principal,
        tenant_id: str,
        *,
        entity_id: str | None = None,
        name: str | None = None,
        include_history: bool = False,
        max_facts: int = 25,
    ) -> EntityCard:
        """Read an entity card and its latest visible fact versions."""
        return await self._queries.entity(
            principal,
            tenant_id,
            entity_id=entity_id,
            name=name,
            include_history=include_history,
            max_facts=max_facts,
        )

    async def graph(
        self,
        principal: Principal,
        tenant_id: str,
        *,
        seeds: Sequence[str] = (),
        query: str = "",
        depth: int = 2,
        max_nodes: int = 100,
        include_history: bool = False,
    ) -> Neighbourhood:
        """Read a bounded neighbourhood of facts visible to the tenant."""
        return await self._queries.graph(
            principal,
            tenant_id,
            seeds=seeds,
            query=query,
            depth=depth,
            max_nodes=max_nodes,
            include_history=include_history,
        )

    async def stats(self, principal: Principal, tenant_id: str) -> dict[str, object]:
        """Return tenant data counts and operation usage without creating a row."""
        return await self._queries.stats(principal, tenant_id)

    async def operations(
        self, principal: Principal, tenant_id: str, *, limit: int = 50
    ) -> list[OperationRecord]:
        """List recent operation rows without creating another row."""
        return await self._queries.operations(principal, tenant_id, limit=limit)

    async def audit_log(
        self, principal: Principal, tenant_id: str, *, limit: int = 100
    ) -> list[AuditRow]:
        """List recent tenant audit rows without creating another row."""
        return await self._queries.audit_log(principal, tenant_id, limit=limit)

    async def export(self, principal: Principal, tenant_id: str) -> TenantExport:
        """Export one tenant's memory and record its row counts in the audit log."""
        return await self._queries.export(principal, tenant_id)

    async def get_job(
        self, principal: Principal, job_id: str, *, trace_id: str | None = None
    ) -> JobView:
        """Read one job after checking its tenant grant."""
        return await self._jobs.get_job(principal, job_id, trace_id=trace_id)

    async def get_status(
        self, principal: Principal, *, trace_id: str | None = None
    ) -> StatusOut:
        """Report service availability and visible tenant job counts."""
        return await self._queries.get_status(principal, trace_id=trace_id)

    async def recover_all(self) -> RecoveryReport:
        """Finish pending commits and erasures for every known tenant."""
        return await self._forget.recover_all()

    async def drain(self) -> None:
        """Wait until all queued tenant lanes finish."""
        return await self._jobs.drain()

    async def stop(self) -> None:
        """Stop active job lanes without closing injected stores."""
        return await self._jobs.stop()

    async def start(self) -> int:
        """Recover pending work, then start the job runner."""
        async with self._write_lock:
            await self._forget.recover_all()
            return await self._jobs.start()
