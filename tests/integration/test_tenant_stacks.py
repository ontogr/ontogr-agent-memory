"""Tenant stack and runtime isolation on the live graph backends."""

from ontogr_mem.data_models.clock import SystemClock
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.runtime import build_tenant_runtime
from tests.support.embedder import MockEmbedder
from tests.support.graph_backend import GraphBackend
from tests.support.llm import ScriptedLLM


class TestTenantStacks:
    """A tenant stack starts empty, reports model availability and stays isolated."""

    async def test_runtime_starts_empty_and_reports_model_availability(
        self, graph_backend: GraphBackend
    ):
        """A fresh stack has genesis ontology and client availability is explicit."""
        embedder = MockEmbedder()
        stores = await graph_backend.open_stores(embedder)
        assert (await stores.ontology_store.get()).version == 0
        assert await stores.episodes.list_episodes(limit=10) == []

        arguments = {
            "stores": stores,
            "embedder": embedder,
            "clock": SystemClock(),
            "policy": ResolutionPolicy(),
            "resolver_settings": ResolverSettings(),
            "retrieval_settings": RetrievalSettings(),
        }
        assert not build_tenant_runtime(llm=None, **arguments).llm_available
        assert build_tenant_runtime(
            llm=ScriptedLLM(replies=[]), **arguments
        ).llm_available

    async def test_stacks_do_not_share_entities_or_episodes(
        self, graph_backend: GraphBackend
    ):
        """Two tenants on one backend cannot see each other's data."""
        first = await graph_backend.open_stores()
        second = await graph_backend.open_stores()
        now = SystemClock().now()
        await first.resolver_store.save_entity(
            Entity(
                entity_id="only-first",
                type_label="Thing",
                canonical_name="Only first",
                strict_key="only first",
                loose_key="only first",
                created_at=now,
                updated_at=now,
            )
        )

        assert await first.episodes.list_episodes(limit=10) == []
        assert await second.episodes.list_episodes(limit=10) == []
        assert await first.resolver_store.has_entity("only-first")
        assert not await second.resolver_store.has_entity("only-first")
