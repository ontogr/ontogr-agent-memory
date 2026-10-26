"""Inline memory reads, feedback, and service metadata."""

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

from ontogr_mem.consolidation.signals import record_feedback
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.export import TenantExport, export_tenant
from ontogr_mem.relational_store import (
    OperationKind,
    OperationRecord,
)
from ontogr_mem.relational_store.audit import (
    AuditAction,
    AuditEntry,
    AuditOutcome,
    AuditRow,
)
from ontogr_mem.retrieval.errors import KeywordSearchError, RetrievalError
from ontogr_mem.retrieval.neighbourhood import Neighbourhood, neighbourhood
from ontogr_mem.service.contracts import (
    EntityView,
    EpisodePage,
    GetEntity,
    ListEpisodes,
    MemoryContext,
    MemoryFeedback,
    SearchMemory,
    StatusOut,
)
from ontogr_mem.service.dependencies import ServiceDeps
from ontogr_mem.service.errors import (
    InvalidArgumentError,
    ObjectNotFoundError,
)
from ontogr_mem.service.operation_context import OperationRecorder
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.service.reads import get_entity as read_entity
from ontogr_mem.service.reads import list_episodes as read_episodes
from ontogr_mem.service.reads import search as read_search
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.summaries import EntityCard, entity_card, tenant_stats
from ontogr_mem.temporal.lineage import FactHistory, fact_history
from ontogr_mem.temporal.query import TemporalQuery


def _search_tenants(principal: Principal, request: SearchMemory) -> list[str]:
    """Resolve the requested search scope without broadening it."""
    if request.tenant is not None and request.tenants:
        raise InvalidArgumentError("name either tenant or tenants, not both")
    requested = request.tenants or [request.tenant or principal.default_tenant]
    return list(dict.fromkeys(requested))


class ServiceQueries:
    """Authorize reads, feedback, exports, and service metadata queries."""

    def __init__(self, deps: ServiceDeps, recorder: OperationRecorder) -> None:
        """Bind tenant runtimes, operation recording, and service policy."""
        self._deps = deps
        self._recorder = recorder

    async def search_memory(
        self,
        principal: Principal,
        request: SearchMemory,
        *,
        trace_id: str | None = None,
    ) -> MemoryContext:
        """Search one or more authorized tenants inline.

        Every requested tenant must be readable by ``principal``. The returned
        trace id identifies the retrieval operation.
        """
        self._deps.limiter.check(principal.id, "read")
        tenant_ids = _search_tenants(principal, request)
        async with self._recorder.track(
            OperationKind.RETRIEVE,
            principal.id,
            None,
            session_id=request.session_id,
            trace_id=trace_id,
        ) as context:
            runtimes = []
            for tenant_id in tenant_ids:
                selected = await self._deps.tenants.require(
                    principal, tenant_id, Grant.READ, "memory.search"
                )
                runtimes.append(
                    (selected, await self._deps.tenants.require_existing(selected))
                )
            result = await read_search(
                runtimes, request, self._deps.clock.now(), self._deps.state
            )
            return result.model_copy(update={"trace_id": context.trace_id})

    async def get_entity(
        self,
        principal: Principal,
        request: GetEntity,
        *,
        trace_id: str | None = None,
    ) -> EntityView:
        """Read one canonical entity and its current or historical facts.

        Args:
            principal: Caller that must hold read access to the tenant.
            request: Entity id or strict name and history options.
            trace_id: Optional trace id shared with the calling transport.

        Returns:
            The canonical entity and the latest version of each matching fact.
        """
        self._deps.limiter.check(principal.id, "read")
        tenant_id = request.tenant or principal.default_tenant
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, tenant_id, trace_id=trace_id
        ):
            tenant_id = await self._deps.tenants.require(
                principal,
                tenant_id,
                Grant.READ,
                "memory.entity",
                target=request.entity_id or request.name or "",
            )
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            return await read_entity(
                tenant_id, runtime, request, self._deps.clock.now()
            )

    async def list_episodes(
        self,
        principal: Principal,
        request: ListEpisodes,
        *,
        trace_id: str | None = None,
    ) -> EpisodePage:
        """Read one newest-first page of episode metadata.

        Args:
            principal: Caller that must hold read access to the tenant.
            request: Page size, optional cursor and optional session filter.
            trace_id: Optional trace id shared with the calling transport.

        Returns:
            The page and a cursor when more episodes remain.
        """
        self._deps.limiter.check(principal.id, "read")
        tenant_id = request.tenant or principal.default_tenant
        async with self._recorder.track(
            OperationKind.RETRIEVE,
            principal.id,
            tenant_id,
            session_id=request.session_id,
            trace_id=trace_id,
        ):
            tenant_id = await self._deps.tenants.require(
                principal, tenant_id, Grant.READ, "memory.episodes"
            )
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            return await read_episodes(tenant_id, runtime, request)

    async def memory_feedback(
        self, principal: Principal, request: MemoryFeedback
    ) -> dict[str, str | float | int]:
        """Record a caller rating for an entity or a fact's endpoints."""
        self._deps.limiter.check(principal.id, "write")
        tenant_id = request.tenant or principal.default_tenant
        tenant_id = await self._deps.tenants.require(
            principal,
            tenant_id,
            Grant.WRITE,
            "memory.feedback",
            target=request.entity_id or request.fact_id or "",
        )
        async with self._recorder.track(OperationKind.WRITE, principal.id, tenant_id):
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            targets = await self._feedback_targets(runtime, request)
            rating = (request.score + 1.0) / 2.0
            for target_id in targets:
                await record_feedback(
                    runtime.stores.cons_store, target_id, rating, self._deps.clock.now()
                )
            return {
                "recorded": len(targets),
                "rating": rating,
                "tenant_id": tenant_id,
            }

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
        self._deps.limiter.check(principal.id, "read")
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.READ, "memory.history", target=fact_id
        )
        async with self._recorder.track(OperationKind.HISTORY, principal.id, tenant_id):
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            history = await fact_history(
                runtime.stores.temporal_store,
                runtime.reader,
                fact_id,
                now=self._deps.clock.now(),
                valid_at=valid_at,
                known_at=known_at,
            )
            if history is None:
                raise ObjectNotFoundError("tenant or object not found")
            return history

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
        self._deps.limiter.check(principal.id, "read")
        tenant_id = await self._deps.tenants.require(
            principal,
            tenant_id,
            Grant.READ,
            "memory.entity",
            target=entity_id or name or "",
        )
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, tenant_id
        ):
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            try:
                card = await entity_card(
                    runtime.stores.temporal_store,
                    runtime.stores.resolver_store,
                    runtime.reader,
                    now=self._deps.clock.now(),
                    entity_id=entity_id,
                    name=name,
                    include_history=include_history,
                    max_facts=max_facts,
                )
            except ValueError as exc:
                raise InvalidArgumentError(str(exc)) from exc
            if card is None:
                raise ObjectNotFoundError("tenant or object not found")
            return card

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
        self._deps.limiter.check(principal.id, "read")
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.READ, "memory.graph"
        )
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, tenant_id
        ):
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            try:
                return await neighbourhood(
                    runtime.stores.temporal_store,
                    runtime.stores.resolver_store,
                    runtime.stores.keywords,
                    runtime.reader,
                    now=self._deps.clock.now(),
                    seeds=seeds,
                    query=query,
                    depth=depth,
                    max_nodes=max_nodes,
                    include_history=include_history,
                )
            # A keyword failure is a server fault, though it subclasses RetrievalError.
            except KeywordSearchError:
                raise
            except RetrievalError as exc:
                raise InvalidArgumentError(str(exc)) from exc

    async def stats(self, principal: Principal, tenant_id: str) -> dict[str, object]:
        """Return tenant data counts and operation usage without creating a row."""
        self._deps.limiter.check(principal.id, "read")
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.READ, "memory.stats"
        )
        runtime = await self._deps.tenants.get(tenant_id, create=False)
        stats = asdict(
            await tenant_stats(
                runtime.stores.temporal_store,
                runtime.stores.resolver_store,
                runtime.stores.episodes,
                now=self._deps.clock.now(),
            )
        )
        stats["usage"] = asdict(await self._deps.operations.usage(tenant_id))
        return stats

    async def operations(
        self, principal: Principal, tenant_id: str, *, limit: int = 50
    ) -> list[OperationRecord]:
        """List recent operation rows without creating another row."""
        self._deps.limiter.check(principal.id, "read")
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.READ, "memory.operations"
        )
        if not 1 <= limit <= 500:
            raise InvalidArgumentError("limit must be between 1 and 500")
        return await self._deps.operations.list_operations(
            tenant=tenant_id, limit=limit
        )

    async def audit_log(
        self, principal: Principal, tenant_id: str, *, limit: int = 100
    ) -> list[AuditRow]:
        """List recent tenant audit rows without creating another row."""
        self._deps.limiter.check(principal.id, "read")
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.ADMIN, "memory.audit_log"
        )
        if not 1 <= limit <= 1000:
            raise InvalidArgumentError("limit must be between 1 and 1000")
        return await self._deps.audit_log.list_rows(tenant=tenant_id, limit=limit)

    async def export(self, principal: Principal, tenant_id: str) -> TenantExport:
        """Export one tenant's memory and record its row counts in the audit log."""
        tenant_id = await self._deps.tenants.require(
            principal, tenant_id, Grant.EXPORT, "memory.export"
        )
        async with self._recorder.track(
            OperationKind.EXPORT, principal.id, tenant_id
        ) as context:
            runtime = await self._deps.tenants.get(tenant_id, create=False)
            result = await export_tenant(
                runtime.stores.temporal_store,
                runtime.stores.resolver_store,
                runtime.stores.episodes,
                runtime.reader,
                tenant_id=tenant_id,
                now=self._deps.clock.now(),
            )
            await self._deps.audit_log.append(
                AuditEntry(
                    action=AuditAction.EXPORT,
                    outcome=AuditOutcome.OK,
                    principal_id=principal.id,
                    tenant=tenant_id,
                    target=tenant_id,
                    trace_id=context.trace_id,
                    detail={
                        f"count_{row_type}": result.counts.get(row_type, 0)
                        for row_type in ("meta", "entity", "fact", "episode")
                    },
                )
            )
            return result

    async def get_status(
        self, principal: Principal, *, trace_id: str | None = None
    ) -> StatusOut:
        """Report service availability and visible tenant job counts.

        Only tenants with a readable grant are included in the response.
        """
        self._deps.limiter.check(principal.id, "read")
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, None, trace_id=trace_id
        ):
            tenants = await self._deps.tenants.visible(principal)
            for tenant_id in tenants:
                await self._deps.tenants.require(
                    principal, tenant_id, Grant.READ, "memory.status"
                )
            queued = await self._deps.state.count_jobs(
                tenants, [JobState.QUEUED, JobState.RUNNING]
            )
            relation_counts: dict[str, int] = {}
            for tenant_id in tenants:
                runtime = await self._deps.tenants.get(tenant_id, create=False)
                ontology = await runtime.ontology.catalog()
                relation_counts[tenant_id] = len(ontology.relations)
            return StatusOut(
                status="ok",
                backend=self._deps.backend,
                llm_available=self._deps.llm is not None,
                tenants=tenants,
                queued_jobs=queued,
                detail={"relation_counts": relation_counts},
            )

    async def _feedback_targets(
        self, runtime: TenantRuntime, request: MemoryFeedback
    ) -> list[str]:
        """Resolve feedback to one entity or the two endpoints of a fact."""
        if (request.entity_id is None) == (request.fact_id is None):
            raise InvalidArgumentError("name exactly one of entity_id and fact_id")
        if request.entity_id is not None:
            entities = await runtime.resolver.get_entities(
                [request.entity_id], follow_merged=True
            )
            if not entities:
                raise ObjectNotFoundError("tenant or object not found")
            return [entities[0].entity_id]
        views = await runtime.reader.by_ids(
            [request.fact_id or ""], TemporalQuery(now=self._deps.clock.now())
        )
        if not views:
            raise ObjectNotFoundError("tenant or object not found")
        return sorted({views[0].subject_id, views[0].object_id})
