"""Build storage ports for one in-memory Tenant."""

from ontogr_mem.graphstore.memory import (
    MemoryConsStore,
    MemoryEpisodeStore,
    MemoryKeywordSearch,
    MemoryOntologyStore,
    MemoryResolverStore,
    MemoryTemporalStore,
)
from ontogr_mem.observability import Tracer
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder, EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.memory import InMemoryBackend


async def memory_stores(
    tenant_id: str, embedder: Embedder, *, tracer: Tracer | None = None
) -> TenantStores:
    """Create isolated in-memory stores and a ready vector store."""
    episodes = MemoryEpisodeStore()
    resolver = MemoryResolverStore()
    temporal = MemoryTemporalStore(episodes, resolver)
    vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(embedder, tracer=tracer))
    await vectors.ensure_ready()
    return TenantStores(
        tenant_id=tenant_id,
        episodes=episodes,
        resolver_store=resolver,
        temporal_store=temporal,
        ontology_store=MemoryOntologyStore(episodes.ingested_ids),
        cons_store=MemoryConsStore(),
        keywords=MemoryKeywordSearch(resolver, temporal, episodes),
        vectors=vectors,
    )
