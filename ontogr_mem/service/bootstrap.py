"""Settings and construction helpers for the memory service."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.data_models.clock import Clock, SystemClock
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.llm import LLMClient
from ontogr_mem.llm.cache.port import ResponseCache
from ontogr_mem.observability import NoOpTracer, Tracer
from ontogr_mem.relational_store import (
    MemoryOperationStore,
    OperationStore,
    SqliteOperationStore,
    StateSettings,
    StateStore,
    open_state_store,
)
from ontogr_mem.relational_store.audit import AuditLog
from ontogr_mem.relational_store.engine import pg_pool, sqlite_shared_connection
from ontogr_mem.relational_store.memory.audit import MemoryAuditLog
from ontogr_mem.relational_store.postgres.audit import PostgresAuditLog
from ontogr_mem.relational_store.postgres.operations import (
    PostgresOperationStore,
)
from ontogr_mem.relational_store.schema.runner import (
    ensure_postgres_schema,
)
from ontogr_mem.relational_store.sqlite.audit import SqliteAuditLog
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.access.audit_sink import (
    AuditSink,
    ChainedAuditSink,
    InMemoryAuditSink,
)
from ontogr_mem.service.access.authz import Authorizer
from ontogr_mem.service.dependencies import ServiceDeps
from ontogr_mem.service.forget.confirm import ConfirmTokens
from ontogr_mem.service.forget.workflow import EraseWorkflow
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.memory_service import MemoryService
from ontogr_mem.service.runtime import TenantRuntime, build_tenant_runtime
from ontogr_mem.service.tenants import StoresFactory, TenantRuntimeCache
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder


@dataclass(slots=True)
class ServiceRuntime:
    """The service owned by ``open_service``."""

    service: MemoryService


class ServiceSettings(BaseSettings):
    """Configure the service graph backend.

    Environment variables use the ``MEMORY_SERVICE_`` prefix.

    Attributes:
        backend: Tenant graph backend. Env: ``MEMORY_SERVICE_BACKEND``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_SERVICE_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    backend: Literal["falkordb", "neo4j"] = "falkordb"


async def build_service(
    settings: ServiceSettings,
    *,
    state: StateStore,
    stores_factory: StoresFactory,
    authorizer: Authorizer,
    operations: OperationStore,
    clock: Clock,
    llm: LLMClient | None,
    embedder: Embedder,
    policy: ResolutionPolicy,
    resolver_settings: ResolverSettings,
    retrieval_settings: RetrievalSettings,
    limiter: CallerRateLimiter,
    max_attempts: int = 3,
    tracer: Tracer | None = None,
    audit_log: AuditLog | None = None,
    confirm_tokens: ConfirmTokens | None = None,
    drop_graph: Callable[[str], Awaitable[None]] | None = None,
    erase_cache: ResponseCache | None = None,
) -> MemoryService:
    """Build the service from settings and caller-owned dependencies.

    The caller must inject ``authorizer`` and ``operations``. The service needs
    access checks and operation records, and transient defaults for them here
    could discard state.

    Args:
        settings: Tenant graph backend.
        state: Shared job and erase state store.
        stores_factory: Opens the configured backend's stores for each Tenant.
        authorizer: Checks grants and records access decisions.
        operations: Persists service operation rows.
        clock: Time source shared by service components.
        llm: Optional language model client.
        embedder: Client used by Tenant vector stores.
        policy: Entity resolution policy.
        resolver_settings: Entity resolver configuration.
        retrieval_settings: Retrieval configuration.
        limiter: Per-principal read and write limiter.
        max_attempts: Maximum attempts for each queued job.
        tracer: Optional tracer for service and Tenant spans.
        audit_log: Optional audit chain for service operations.
        confirm_tokens: Optional signer for forget confirmation tokens.
        drop_graph: Graph cleanup callback that erases one Tenant's graph.
        erase_cache: Optional cache to clean during erase operations.

    Returns:
        The configured memory service.

    Raises:
        ValueError: If no graph cleanup callback is given.
    """
    if drop_graph is None:
        raise ValueError("a graph cleanup callback is required")

    selected_tracer = tracer or NoOpTracer()

    async def runtime_factory(tenant_id: str) -> TenantRuntime:
        stores = await stores_factory(tenant_id, embedder, tracer=selected_tracer)
        return build_tenant_runtime(
            stores,
            llm=llm,
            embedder=embedder,
            clock=clock,
            policy=policy,
            resolver_settings=resolver_settings,
            retrieval_settings=retrieval_settings,
            tracer=selected_tracer,
        )

    tenants = TenantRuntimeCache(runtime_factory, state, authorizer)
    workflow = EraseWorkflow(
        state,
        tenants.get_or_create,
        clock,
        cache=erase_cache,
        drop_graph=drop_graph,
        delete_tenant_jobs=state.delete_tenant,
    )
    return MemoryService(
        ServiceDeps(
            tenants=tenants,
            state=state,
            operations=operations,
            clock=clock,
            llm=llm,
            backend=settings.backend,
            limiter=limiter,
            max_attempts=max_attempts,
            tracer=selected_tracer,
            audit_log=audit_log or MemoryAuditLog(clock),
            confirm_tokens=confirm_tokens or ConfirmTokens(None, clock),
            workflow=workflow,
        )
    )


@asynccontextmanager
async def open_service(
    settings: ServiceSettings | None = None,
    *,
    state_settings: StateSettings | None = None,
    stores_factory: StoresFactory | None = None,
    clock: Clock | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder,
    policy: ResolutionPolicy | None = None,
    resolver_settings: ResolverSettings | None = None,
    retrieval_settings: RetrievalSettings | None = None,
    limiter: CallerRateLimiter | None = None,
    max_attempts: int = 3,
    tracer: Tracer | None = None,
    erase_cache: ResponseCache | None = None,
) -> AsyncGenerator[ServiceRuntime, None]:
    """Open, start, and own the service resources for one process lifetime.

    Args:
        settings: Tenant graph backend.
        state_settings: Durable job and erase-journal configuration.
        stores_factory: Optional tenant-store factory override.
        clock: Shared time source; defaults to the system clock.
        llm: Optional client for text extraction and model-backed operations.
        embedder: Client used by tenant vector stores.
        policy: Entity resolution policy.
        resolver_settings: Entity resolution settings.
        retrieval_settings: Retrieval settings.
        limiter: Request limiter; defaults to the standard service limits.
        max_attempts: Maximum attempts for queued jobs.
        tracer: Optional tracer shared by service components.
        erase_cache: Optional cache to clean during erase operations.

    Yields:
        A service runtime. The runtime and its stores are closed
        when the context exits.
    """
    service_settings = settings or ServiceSettings()
    selected_state = state_settings or StateSettings()
    selected_clock = clock or SystemClock()

    async with AsyncExitStack() as resources:
        state = await open_state_store(selected_state)
        resources.push_async_callback(state.close)

        selected_stores_factory = stores_factory
        drop_graph: Callable[[str], Awaitable[None]] | None = None
        if selected_stores_factory is None:
            graph_store, selected_stores_factory = _BACKENDS[service_settings.backend]()
            resources.push_async_callback(graph_store.close)
            drop_graph = graph_store.erase_tenant

        operations, audit_log, audit_sink = await _STATE_AUDITS[selected_state.backend](
            selected_state, resources, selected_clock
        )

        authorizer = Authorizer(audit_sink, selected_clock, tracer=tracer)
        service = await build_service(
            service_settings,
            state=state,
            stores_factory=selected_stores_factory,
            authorizer=authorizer,
            operations=operations,
            clock=selected_clock,
            llm=llm,
            embedder=embedder,
            policy=policy or ResolutionPolicy(),
            resolver_settings=resolver_settings or ResolverSettings(),
            retrieval_settings=retrieval_settings or RetrievalSettings(),
            limiter=limiter or CallerRateLimiter(),
            max_attempts=max_attempts,
            tracer=tracer,
            audit_log=audit_log,
            confirm_tokens=ConfirmTokens(None, selected_clock),
            drop_graph=drop_graph,
            erase_cache=erase_cache,
        )
        resources.push_async_callback(service.stop)
        await service.start()
        yield ServiceRuntime(service=service)


def _graph_stores_factory(
    store: GraphStore,
    *,
    tenant_keyed: bool,
) -> tuple[GraphStore, StoresFactory]:
    """Build tenant stores over a process-owned graph adapter."""
    from ontogr_mem.graphstore.tenant_stacks import graph_stores

    async def create_stores(
        tenant_id: str,
        embedder: Embedder,
        *,
        tracer: Tracer | None = None,
    ) -> TenantStores:
        return await graph_stores(
            store,
            tenant_id,
            embedder,
            tenant_keyed=tenant_keyed,
            tracer=tracer,
        )

    return store, create_stores


def _falkordb_stores() -> tuple[GraphStore, StoresFactory]:
    """Open the FalkorDB adapter with one graph per tenant."""
    from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore

    return _graph_stores_factory(FalkorDBGraphStore(), tenant_keyed=False)


def _neo4j_stores() -> tuple[GraphStore, StoresFactory]:
    """Open the Neo4j adapter with tenant-keyed graph rows."""
    from ontogr_mem.graphstore.neo4j import Neo4jGraphStore

    return _graph_stores_factory(Neo4jGraphStore(), tenant_keyed=True)


_BACKENDS = {
    "falkordb": _falkordb_stores,
    "neo4j": _neo4j_stores,
}


async def _memory_audit(
    settings: StateSettings, resources: AsyncExitStack, clock: Clock
) -> tuple[OperationStore, AuditLog, AuditSink]:
    """Keep operation and access audit records in process memory."""
    return MemoryOperationStore(clock), MemoryAuditLog(clock), InMemoryAuditSink()


async def _sqlite_audit(
    settings: StateSettings, resources: AsyncExitStack, clock: Clock
) -> tuple[OperationStore, AuditLog, AuditSink]:
    """Share one process-wide SQLite connection for operation and audit rows."""
    connection = sqlite_shared_connection(settings.sqlite_path)
    audit_log = SqliteAuditLog(connection, clock)
    return (
        SqliteOperationStore(connection, clock),
        audit_log,
        ChainedAuditSink(audit_log),
    )


async def _postgres_audit(
    settings: StateSettings, resources: AsyncExitStack, clock: Clock
) -> tuple[OperationStore, AuditLog, AuditSink]:
    """Share one process-wide pool for operation and chained audit rows."""
    pool = await pg_pool(settings.dsn)
    await ensure_postgres_schema(pool)
    operations = PostgresOperationStore(pool, clock)
    await operations.mark_interrupted_runs()
    audit_log = PostgresAuditLog(pool, clock)
    return operations, audit_log, ChainedAuditSink(audit_log)


_STATE_AUDITS = {
    "memory": _memory_audit,
    "sqlite": _sqlite_audit,
    "postgres": _postgres_audit,
}
