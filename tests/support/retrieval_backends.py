"""Graph-backed ports and fixed embeddings for the retrieval scenario."""

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.keyword_search import GraphKeywordSearch
from ontogr_mem.graphstore.memory import MemoryOntologyStore
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.schemas import (
    EPISODES,
    KEYWORD,
    RESOLUTION,
    TEMPORAL,
    VECTOR,
)
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.vectorstore import EmbeddingGateway, VectorKind, VectorStore
from ontogr_mem.vectorstore.graph.falkordb import FalkorDBVectorBackend
from ontogr_mem.vectorstore.graph.neo4j import Neo4jVectorBackend
from tests.support.retrieval_world import RetrievalBackend, new_retrieval_embedder
from tests.support.tenants import new_tenant


@asynccontextmanager
async def graph_retrieval_backend(
    store: GraphStore, *, tenant_keyed: bool
) -> AsyncGenerator[RetrievalBackend, None]:
    """Bind the retrieval scenario ports to a fresh graph tenant."""
    handle: GraphHandle = await store.open(
        new_tenant(), [EPISODES, RESOLUTION, TEMPORAL, KEYWORD, VECTOR]
    )
    try:
        runner = store.runner(handle)
        embedder = new_retrieval_embedder()
        vector_backend = (
            Neo4jVectorBackend(store, handle)
            if tenant_keyed
            else FalkorDBVectorBackend(store, handle)
        )
        vectors = VectorStore(vector_backend, EmbeddingGateway(embedder))
        await vectors.ensure_ready()

        async def prune_orphan_vectors(ids: Sequence[str]) -> None:
            await vectors.delete(VectorKind.EDGE, ids)

        yield RetrievalBackend(
            vectors=vectors,
            embedder=embedder,
            episodes=GraphEpisodeStore(runner, tenant_keyed=tenant_keyed),
            resolver_store=GraphResolverStore(runner, tenant_keyed=tenant_keyed),
            temporal_store=GraphTemporalStore(store, handle, tenant_keyed=tenant_keyed),
            ontology_store=MemoryOntologyStore(set[str]),
            keywords=GraphKeywordSearch(runner),
            prune_orphan_vectors=prune_orphan_vectors,
        )
    finally:
        await store.erase_tenant(handle.tenant)
