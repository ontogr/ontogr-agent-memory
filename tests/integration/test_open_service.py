"""Public service runtime construction with durable SQLite state."""

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models
from ontogr_mem.observability import Tracer
from ontogr_mem.ontology import EntityDeclaration, OntologyDeclaration, OntologyManager
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings, open_service
from ontogr_mem.service.contracts import ClearTenant
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.store_urls import sqlite_url
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder, embedder_source
from tests.support.falkordb import falkordb_test_url
from tests.support.graph_backend import GraphBackend
from tests.support.neo4j import neo4j_test_url
from tests.support.tenants import new_tenant


def _backend_url(backend: GraphBackend) -> str:
    """Return the test graph URL that matches a live backend's name."""
    if backend.name == "falkordb":
        return falkordb_test_url()
    return neo4j_test_url()


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
    async def test_yields_a_started_service_runtime(self, backend, tmp_path: Path):
        """The context manager wires the configured graph backend."""
        clock: Clock = FixedClock()
        embedder: Embedder = MockEmbedder()

        graph_url = falkordb_test_url() if backend == "falkordb" else neo4j_test_url()
        settings = ServiceSettings(graph_url=graph_url, _env_file=None)
        state_settings = StateSettings(
            url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
        )
        async with open_service(
            settings,
            state_settings=state_settings,
            clock=clock,
            embedder=embedder_source(embedder),
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
        self, graph_backend: GraphBackend, tmp_path: Path
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
            url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
        )
        async with open_service(
            ServiceSettings(graph_url=_backend_url(graph_backend), _env_file=None),
            state_settings=state_settings,
            stores_factory=stores_factory,
            drop_graph=drop_graph,
            clock=FixedClock(),
            embedder=embedder_source(MockEmbedder()),
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
        self, graph_backend: GraphBackend, tmp_path: Path
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
            url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
        )
        with pytest.raises(ValueError, match="graph cleanup callback"):
            async with open_service(
                ServiceSettings(graph_url=_backend_url(graph_backend), _env_file=None),
                state_settings=state_settings,
                stores_factory=stores_factory,
                clock=FixedClock(),
                embedder=embedder_source(MockEmbedder()),
                limiter=CallerRateLimiter(clock=lambda: 0.0),
            ):
                pytest.fail("the service must not open")

    async def test_caller_owned_model_pool_enables_llm_status_and_stays_open(
        self, tmp_path: Path
    ):
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
                ServiceSettings(graph_url=falkordb_test_url(), _env_file=None),
                state_settings=StateSettings(
                    url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
                ),
                models=models,
                clock=FixedClock(),
                embedder=embedder_source(MockEmbedder()),
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
            ServiceSettings(graph_url=falkordb_test_url(), _env_file=None),
            state_settings=StateSettings(
                url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
            ),
            clock=FixedClock(),
            embedder=embedder_source(MockEmbedder()),
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
                ServiceSettings(graph_url=_backend_url(graph_backend), _env_file=None),
                state_settings=StateSettings(
                    url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
                ),
                stores_factory=stores_factory,
                clock=FixedClock(),
                embedder=embedder_source(MockEmbedder()),
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
            ServiceSettings(graph_url=falkordb_test_url(), _env_file=None),
            state_settings=StateSettings(
                url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
            ),
            clock=FixedClock(),
            embedder=embedder_source(MockEmbedder()),
            limiter=CallerRateLimiter(clock=lambda: 0.0),
        ) as runtime:
            status = await runtime.service.get_status(reader)

        assert status.llm_available is False

    async def test_ontology_declaration_is_scoped_to_the_selected_tenant(
        self, graph_backend: GraphBackend, tmp_path: Path
    ):
        """Initializing another tenant preserves its existing ontology."""
        selected = new_tenant()
        other = new_tenant()
        other_stores = await graph_backend.stores_for(other)
        existing = OntologyDeclaration(entities=[EntityDeclaration(label="Widget")])
        clock = FixedClock()
        other_manager = OntologyManager(other_stores.ontology_store, clock)
        before = await other_manager.declare(existing)
        async with open_service(
            ServiceSettings(graph_url=_backend_url(graph_backend), _env_file=None),
            state_settings=StateSettings(
                url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None
            ),
            stores_factory=graph_backend.stores_for,
            drop_graph=graph_backend.store.erase_tenant,
            clock=clock,
            embedder=embedder_source(MockEmbedder()),
            ontology=OntologyDeclaration(entities=[EntityDeclaration(label="Gadget")]),
            ontology_tenant=selected,
        ) as runtime:
            await runtime.service.get_status(
                Principal(
                    id="reader",
                    default_tenant=selected,
                    grants={
                        selected: frozenset({Grant.READ}),
                        other: frozenset({Grant.READ}),
                    },
                )
            )
        assert await other_manager.catalog() == before
        selected_stores = await graph_backend.stores_for(selected)
        selected_manager = OntologyManager(selected_stores.ontology_store, clock)
        assert [
            entity.label for entity in (await selected_manager.catalog()).entities
        ] == ["Unknown", "Gadget"]
