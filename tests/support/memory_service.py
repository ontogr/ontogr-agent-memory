"""Build the service and public ports over a live graph backend."""

from dataclasses import dataclass, replace

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.llm import LLMClient
from ontogr_mem.observability import Tracer
from ontogr_mem.relational_store import MemoryOperationStore
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
from tests.support.graph_backend import GraphBackend


@dataclass
class MemoryServiceHarness:
    """A service and the stores used to inspect its observable behavior."""

    service: MemoryService
    principal: Principal
    clock: Clock
    operations: MemoryOperationStore
    state: MemoryJobStore


async def memory_service_harness(
    backend: GraphBackend,
    *,
    llm: LLMClient | None = None,
    tracer: Tracer | None = None,
    keywords: KeywordSearch | None = None,
) -> MemoryServiceHarness:
    """Build an isolated service with a grant on one Tenant.

    Args:
        backend: The live graph backend that tenant stores are opened on.
        llm: Optional language model client.
        tracer: Optional tracer for service and tenant spans.
        keywords: Replaces the tenant's keyword search when given.
    """
    clock = FixedClock()
    embedder = MockEmbedder()
    state = MemoryJobStore()
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

    service = await build_service(
        ServiceSettings(backend=backend.name),
        stores_factory=stores_factory,
        state=state,
        authorizer=authorizer,
        operations=operations,
        clock=clock,
        llm=llm,
        embedder=embedder,
        policy=ResolutionPolicy(
            types={
                "Person": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Organization": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Place": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Thing": TypePolicy(mode=ResolutionMode.AUTO_KEY),
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
    return MemoryServiceHarness(service, principal, clock, operations, state)
