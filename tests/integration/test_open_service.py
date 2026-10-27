"""Public service runtime construction with explicitly ephemeral state."""

import pytest

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.observability import Tracer
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings, open_service
from ontogr_mem.service.contracts import ClearTenant
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.graph_backend import GraphBackend


class TestOpenService:
    """The managed runtime starts a service with caller-supplied principals."""

    @pytest.mark.parametrize("backend", ["falkordb", "neo4j"])
    async def test_yields_a_started_service_runtime(self, backend):
        """The context manager wires the configured graph backend."""
        clock: Clock = FixedClock()
        embedder: Embedder = MockEmbedder()

        settings = ServiceSettings(backend=backend, _env_file=None)
        state_settings = StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        )
        async with open_service(
            settings,
            state_settings=state_settings,
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
            assert status.backend == backend
            assert status.tenants == ["default"]

    async def test_tenant_clear_calls_the_injected_graph_cleanup(
        self, graph_backend: GraphBackend
    ):
        """Clearing a tenant drops its graph through the injected drop_graph."""
        dropped: list[str] = []

        async def drop_graph(tenant_id: str) -> None:
            dropped.append(tenant_id)

        async def stores_factory(
            tenant_id: str,
            selected_embedder: Embedder,
            *,
            tracer: Tracer | None = None,
        ) -> TenantStores:
            return await graph_backend.stores_for(
                tenant_id, selected_embedder, tracer=tracer
            )

        state_settings = StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        )
        async with open_service(
            ServiceSettings(backend=graph_backend.name, _env_file=None),
            state_settings=state_settings,
            stores_factory=stores_factory,
            drop_graph=drop_graph,
            clock=FixedClock(),
            embedder=MockEmbedder(),
            limiter=CallerRateLimiter(clock=lambda: 0.0),
        ) as runtime:
            admin = Principal(
                id="admin",
                grants={"t1": frozenset({Grant.ADMIN})},
            )
            preview = await runtime.service.preview_clear_tenant(admin, "t1")

            await runtime.service.clear_tenant(
                admin,
                ClearTenant(tenant="t1", confirm_token=preview.confirm_token),
            )

        assert dropped == ["t1"]

    async def test_custom_stores_factory_without_graph_cleanup_is_refused(
        self, graph_backend: GraphBackend
    ):
        """A custom stores factory needs a graph cleanup callback."""

        async def stores_factory(
            tenant_id: str,
            selected_embedder: Embedder,
            *,
            tracer: Tracer | None = None,
        ) -> TenantStores:
            return await graph_backend.stores_for(
                tenant_id, selected_embedder, tracer=tracer
            )

        state_settings = StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        )
        with pytest.raises(ValueError, match="graph cleanup callback"):
            async with open_service(
                ServiceSettings(backend=graph_backend.name, _env_file=None),
                state_settings=state_settings,
                stores_factory=stores_factory,
                clock=FixedClock(),
                embedder=MockEmbedder(),
                limiter=CallerRateLimiter(clock=lambda: 0.0),
            ):
                pytest.fail("the service must not open")
