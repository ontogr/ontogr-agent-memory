"""Public service runtime construction with explicitly ephemeral test stores."""

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.graphstore.memory.tenant_stacks import memory_stores
from ontogr_mem.observability import Tracer
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings, open_service
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder


class TestOpenService:
    """The managed runtime starts a service with caller-supplied principals."""

    async def test_yields_a_started_service_runtime(self):
        """The context manager wires the injected memory adapters."""
        clock: Clock = FixedClock()
        embedder: Embedder = MockEmbedder()

        async def stores_factory(
            tenant_id: str,
            selected_embedder: Embedder,
            *,
            tracer: Tracer | None = None,
        ) -> TenantStores:
            return await memory_stores(tenant_id, selected_embedder, tracer=tracer)

        settings = ServiceSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        )
        state_settings = StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        )
        async with open_service(
            settings,
            state_settings=state_settings,
            stores_factory=stores_factory,
            clock=clock,
            embedder=embedder,
            limiter=CallerRateLimiter(clock=lambda: 0.0),
        ) as runtime:
            status = await runtime.service.get_status(
                Principal(
                    id="reader",
                    grants={"default": frozenset({Grant.READ})},
                )
            )
            assert status.backend == "memory"
            assert status.tenants == ["default"]
