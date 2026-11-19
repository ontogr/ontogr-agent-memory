"""Build the service and public ports over a live graph backend."""

from dataclasses import dataclass, replace

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.llm import LLMConfig, RoleModels, build_models
from ontogr_mem.observability import Tracer
from ontogr_mem.relational_store import MemoryOperationStore
from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.relational_store.memory import MemoryJobStore
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionMode, ResolutionPolicy, TypePolicy
from ontogr_mem.retrieval.keyword import KeywordSearch
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.access.audit_sink import InMemoryAuditSink
from ontogr_mem.service.access.authz import Authorizer
from ontogr_mem.service.bootstrap import ServiceSettings, build_service
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.memory_service import MemoryService
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.falkordb import falkordb_test_url
from tests.support.graph_backend import GraphBackend
from tests.support.neo4j import neo4j_test_url


@dataclass
class MemoryServiceHarness:
    """A service and the stores used to inspect its observable behavior."""

    service: MemoryService
    principal: Principal
    clock: Clock
    operations: MemoryOperationStore
    state: StateStore


async def memory_service_harness(
    backend: GraphBackend,
    *,
    models: RoleModels | None = None,
    tracer: Tracer | None = None,
    keywords: KeywordSearch | None = None,
    state: StateStore | None = None,
) -> MemoryServiceHarness:
    """Build an isolated service with a grant on one Tenant.

    Args:
        backend: The live graph backend that tenant stores are opened on.
        models: Optional caller-owned model pool passed to the service. When
            omitted, the service gets an unconfigured pool with no clients.
        state: Optional job store passed to the service. When omitted, the
            service gets an in-memory job store.
        tracer: Optional tracer for service and tenant spans.
        keywords: Replaces the tenant's keyword search when given.
    """
    clock = FixedClock()
    embedder = MockEmbedder()
    selected_state = state if state is not None else MemoryJobStore()
    operations = MemoryOperationStore(clock)
    authorizer = Authorizer(
        InMemoryAuditSink(),
        clock,
        tracer=tracer,
    )

    async def stores_factory(
        tenant_id: str,
        embedder: Embedder,
        *,
        tracer: Tracer | None = None,
    ) -> TenantStores:
        stores = await backend.stores_for(tenant_id, embedder, tracer=tracer)
        if keywords is None:
            return stores
        return replace(stores, keywords=keywords)

    graph_url = falkordb_test_url() if backend.name == "falkordb" else neo4j_test_url()
    service = await build_service(
        ServiceSettings(graph_url=graph_url, _env_file=None),
        stores_factory=stores_factory,
        state=selected_state,
        authorizer=authorizer,
        operations=operations,
        clock=clock,
        models=(
            models if models is not None else build_models(LLMConfig.model_construct())
        ),
        embedder=embedder,
        policy=ResolutionPolicy(
            types={
                "Person": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Organization": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Location": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Object": TypePolicy(mode=ResolutionMode.AUTO_KEY),
            }
        ),
        resolver_settings=ResolverSettings(),
        retrieval_settings=RetrievalSettings(),
        limiter=CallerRateLimiter(clock=lambda: 0.0),
        tracer=tracer,
        drop_graph=backend.store.erase_tenant,
    )
    principal = Principal(
        id="service-test-caller",
        grants={"t1": frozenset({Grant.READ, Grant.WRITE})},
        default_tenant="t1",
    )
    return MemoryServiceHarness(service, principal, clock, operations, selected_state)
