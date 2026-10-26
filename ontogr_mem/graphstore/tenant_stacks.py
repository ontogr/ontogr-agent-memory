"""Build graph-backed storage ports for one Tenant."""

from ontogr_mem.graphstore.cons_store import GraphConsStore
from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.keyword_search import GraphKeywordSearch
from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.schemas import (
    CONSOLIDATION,
    EPISODES,
    KEYWORD,
    ONTOLOGY,
    RESOLUTION,
    TEMPORAL,
    VECTOR,
)
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.observability import Tracer
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder, EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.graph.falkordb import FalkorDBVectorBackend
from ontogr_mem.vectorstore.graph.neo4j import Neo4jVectorBackend


async def graph_stores(
    store: GraphStore,
    tenant_id: str,
    embedder: Embedder,
    *,
    tenant_keyed: bool,
    tracer: Tracer | None = None,
) -> TenantStores:
    """Open the tenant schema and bind all graph adapters to its runner."""
    handle = await store.open(
        tenant_id,
        [EPISODES, ONTOLOGY, RESOLUTION, TEMPORAL, CONSOLIDATION, KEYWORD, VECTOR],
    )
    runner = store.runner(handle)

    vector_backend = (
        Neo4jVectorBackend(store, handle)
        if tenant_keyed
        else FalkorDBVectorBackend(store, handle)
    )
    vectors = VectorStore(vector_backend, EmbeddingGateway(embedder, tracer=tracer))
    await vectors.ensure_ready()
    resolver = GraphResolverStore(runner, tenant_keyed=tenant_keyed)
    episodes = GraphEpisodeStore(runner, tenant_keyed=tenant_keyed)
    temporal = GraphTemporalStore(store, handle, tenant_keyed=tenant_keyed)
    return TenantStores(
        tenant_id=tenant_id,
        episodes=episodes,
        resolver_store=resolver,
        temporal_store=temporal,
        ontology_store=GraphOntologyStore(store, handle, tenant_keyed=tenant_keyed),
        cons_store=GraphConsStore(store, handle, tenant_keyed=tenant_keyed),
        keywords=GraphKeywordSearch(runner),
        vectors=vectors,
    )
