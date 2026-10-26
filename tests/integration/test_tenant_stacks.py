"""Tenant stack and runtime isolation on the in-memory adapters."""

from ontogr_mem.data_models.clock import SystemClock
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore.memory.tenant_stacks import memory_stores
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.runtime import build_tenant_runtime
from tests.support.embedder import MockEmbedder
from tests.support.llm import ScriptedLLM


async def test_memory_runtime_starts_empty_and_reports_model_availability():
    """A fresh stack has genesis ontology and client availability is explicit."""
    embedder = MockEmbedder()
    stores = await memory_stores("stack-one", embedder)
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
    assert build_tenant_runtime(llm=ScriptedLLM(replies=[]), **arguments).llm_available


async def test_memory_stacks_do_not_share_entities_or_episodes():
    """Two tenant factories return independent stores."""
    embedder = MockEmbedder()
    first = await memory_stores("stack-a", embedder)
    second = await memory_stores("stack-b", embedder)
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
