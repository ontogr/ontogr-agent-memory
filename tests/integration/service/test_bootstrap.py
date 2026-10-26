"""Build a service with explicit storage, access and operation dependencies."""

import asyncio

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.observability import InMemoryTracer, Tracer
from ontogr_mem.relational_store import MemoryOperationStore
from ontogr_mem.relational_store.memory import MemoryJobStore
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.access.audit_sink import InMemoryAuditSink
from ontogr_mem.service.access.authz import Authorizer
from ontogr_mem.service.bootstrap import ServiceSettings, build_service
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.memory_service import MemoryService
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.service.tenants import StoresFactory
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.graph_backend import GraphBackend


class TestBuildService:
    """Check bootstrap policy and caller-visible service construction."""

    async def test_status_uses_configured_backend_and_injected_services(
        self, graph_backend: GraphBackend
    ):
        """Status and tracing use the explicitly supplied dependencies."""
        tracer = InMemoryTracer()
        settings = ServiceSettings(backend=graph_backend.name, _env_file=None)
        service, operations = await _build_service(
            graph_backend, settings, tracer=tracer
        )
        principal = Principal(
            id="bootstrap-test",
            grants={"tenant-a": frozenset({Grant.READ})},
            default_tenant="tenant-a",
        )

        status = await service.get_status(principal)
        rows = await operations.list_operations(tenant=None)

        assert status.backend == graph_backend.name
        assert status.llm_available is False
        assert status.tenants == ["tenant-a"]
        assert [(row.kind.value, row.outcome.value) for row in rows] == [
            ("retrieve", "ok")
        ]
        assert {span.name for span in tracer.finished()} >= {
            "memory.retrieve",
            "auth.check",
        }

    async def test_tenants_initialize_concurrently_without_duplicate_runtime(
        self, graph_backend: GraphBackend
    ):
        """Cold initialization for one tenant does not block another tenant."""
        factory_release = asyncio.Event()
        both_tenants_started = asyncio.Event()
        factory_tenants: list[str] = []

        async def stores_factory(
            tenant_id: str,
            embedder: Embedder,
            *,
            tracer: Tracer | None = None,
        ) -> TenantStores:
            factory_tenants.append(tenant_id)
            if {"tenant-a", "tenant-b"}.issubset(factory_tenants):
                both_tenants_started.set()
            await factory_release.wait()
            return await graph_backend.stores_for(tenant_id, embedder, tracer=tracer)

        service, _ = await _build_service(
            graph_backend,
            ServiceSettings(backend=graph_backend.name, _env_file=None),
            stores_factory=stores_factory,
        )
        principal = Principal(
            id="parallel-initialization-test",
            grants={
                "tenant-a": frozenset({Grant.READ}),
                "tenant-b": frozenset({Grant.READ}),
            },
            default_tenant="tenant-a",
        )
        stats = asyncio.gather(
            service.stats(principal, "tenant-a"),
            service.stats(principal, "tenant-a"),
            service.stats(principal, "tenant-b"),
        )

        concurrent = True
        try:
            await asyncio.wait_for(both_tenants_started.wait(), timeout=1)
        except TimeoutError:
            concurrent = False
        finally:
            factory_release.set()
        results = await stats

        assert concurrent
        assert len(results) == 3
        assert sorted(factory_tenants) == ["tenant-a", "tenant-b"]


async def _build_service(
    graph_backend: GraphBackend,
    settings: ServiceSettings,
    *,
    tracer: Tracer | None = None,
    stores_factory: StoresFactory | None = None,
) -> tuple[MemoryService, MemoryOperationStore]:
    """Create the explicit test dependencies and call the public builder."""
    clock: Clock = FixedClock()
    embedder: Embedder = MockEmbedder()
    state = MemoryJobStore()
    operations = MemoryOperationStore(clock)
    authorizer = Authorizer(
        InMemoryAuditSink(),
        clock,
        tracer=tracer,
    )

    async def graph_stores_factory(
        tenant_id: str,
        embedder: Embedder,
        *,
        tracer: Tracer | None = None,
    ) -> TenantStores:
        return await graph_backend.stores_for(tenant_id, embedder, tracer=tracer)

    service = await build_service(
        settings,
        state=state,
        authorizer=authorizer,
        operations=operations,
        clock=clock,
        llm=None,
        embedder=embedder,
        stores_factory=stores_factory or graph_stores_factory,
        drop_graph=graph_backend.store.erase_tenant,
        policy=ResolutionPolicy(),
        resolver_settings=ResolverSettings(),
        retrieval_settings=RetrievalSettings(),
        limiter=CallerRateLimiter(clock=lambda: 0.0),
        tracer=tracer,
    )
    return service, operations
