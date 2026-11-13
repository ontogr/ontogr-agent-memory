"""Public service runtime construction with explicitly ephemeral state."""

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models
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


def _judge_model(models: RoleModels) -> Any:
    """Return the provider model behind the judge role."""
    wrapped = models.model_for(ModelRole.JUDGE)
    assert isinstance(wrapped, ConcurrencyLimitedModel)
    return wrapped.wrapped


def _record_built_pools(
    monkeypatch: pytest.MonkeyPatch, built: list[RoleModels]
) -> None:
    """Record each model pool that open_service builds from the environment."""

    def record(*args: Any, **kwargs: Any) -> RoleModels:
        pool = build_models(*args, **kwargs)
        built.append(pool)
        return pool

    monkeypatch.setattr("ontogr_mem.service.bootstrap.build_models", record)


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

    async def test_caller_owned_model_pool_enables_llm_status_and_stays_open(self):
        """A pool passed in reports an LLM and its HTTP client stays open."""
        models = build_models(
            LLMConfig(
                model="m",
                base_url="https://llm.invalid/v1",
                api_key="k",
                _env_file=None,
            )
        )
        reader = Principal(id="reader", grants={"default": frozenset({Grant.READ})})
        try:
            async with open_service(
                ServiceSettings(backend="falkordb", _env_file=None),
                state_settings=StateSettings(
                    backend="memory", allow_ephemeral=True, _env_file=None
                ),
                models=models,
                clock=FixedClock(),
                embedder=MockEmbedder(),
                limiter=CallerRateLimiter(clock=lambda: 0.0),
            ) as runtime:
                status = await runtime.service.get_status(reader)
                assert status.llm_available is True

            assert not _judge_model(models).client._client.is_closed
        finally:
            await models.aclose()

    async def test_builds_model_pool_from_environment_when_none_is_passed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Model settings in the environment enable LLM status without a request."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MEMORY_LLM_MODEL", "openai-chat:test-model")
        monkeypatch.setenv("MEMORY_LLM_BASE_URL", "https://llm.invalid/v1")
        monkeypatch.setenv("MEMORY_LLM_API_KEY", "test-key")
        reader = Principal(id="reader", grants={"default": frozenset({Grant.READ})})
        built: list[RoleModels] = []
        _record_built_pools(monkeypatch, built)

        async with open_service(
            ServiceSettings(backend="falkordb", _env_file=None),
            state_settings=StateSettings(
                backend="memory", allow_ephemeral=True, _env_file=None
            ),
            clock=FixedClock(),
            embedder=MockEmbedder(),
            limiter=CallerRateLimiter(clock=lambda: 0.0),
        ) as runtime:
            status = await runtime.service.get_status(reader)
            assert len(built) == 1
            assert not _judge_model(built[0]).client._client.is_closed

        assert status.llm_available is True
        assert _judge_model(built[0]).client._client.is_closed

    async def test_startup_failure_after_model_build_closes_owned_clients(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        graph_backend: GraphBackend,
    ):
        """A startup failure still closes clients of the environment-built pool."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MEMORY_LLM_MODEL", "openai-chat:test-model")
        monkeypatch.setenv("MEMORY_LLM_BASE_URL", "https://llm.invalid/v1")
        monkeypatch.setenv("MEMORY_LLM_API_KEY", "test-key")
        built: list[RoleModels] = []
        _record_built_pools(monkeypatch, built)

        async def stores_factory(
            tenant_id: str,
            selected_embedder: Embedder,
            *,
            tracer: Tracer | None = None,
        ) -> TenantStores:
            return await graph_backend.stores_for(
                tenant_id, selected_embedder, tracer=tracer
            )

        with pytest.raises(ValueError, match="graph cleanup callback"):
            async with open_service(
                ServiceSettings(backend=graph_backend.name, _env_file=None),
                state_settings=StateSettings(
                    backend="memory", allow_ephemeral=True, _env_file=None
                ),
                stores_factory=stores_factory,
                clock=FixedClock(),
                embedder=MockEmbedder(),
                limiter=CallerRateLimiter(clock=lambda: 0.0),
            ):
                pytest.fail("the service must not open")

        assert len(built) == 1
        assert _judge_model(built[0]).client._client.is_closed

    async def test_reports_no_llm_without_models_or_model_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """No model pool and no MEMORY_LLM_ settings leave LLM status unavailable."""
        monkeypatch.chdir(tmp_path)
        for name in list(os.environ):
            if name.startswith("MEMORY_LLM_"):
                monkeypatch.delenv(name)
        reader = Principal(id="reader", grants={"default": frozenset({Grant.READ})})

        async with open_service(
            ServiceSettings(backend="falkordb", _env_file=None),
            state_settings=StateSettings(
                backend="memory", allow_ephemeral=True, _env_file=None
            ),
            clock=FixedClock(),
            embedder=MockEmbedder(),
            limiter=CallerRateLimiter(clock=lambda: 0.0),
        ) as runtime:
            status = await runtime.service.get_status(reader)

        assert status.llm_available is False
