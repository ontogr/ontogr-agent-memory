"""Graph-backed ports and fixed embeddings for the retrieval scenario."""

from collections.abc import Sequence

from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.keyword_search import GraphKeywordSearch
from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.vectorstore import EmbeddingGateway, VectorKind, VectorStore
from ontogr_mem.vectorstore.graph.falkordb import FalkorDBVectorBackend
from ontogr_mem.vectorstore.graph.neo4j import Neo4jVectorBackend
from tests.support.graph_backend import GraphBackend
from tests.support.retrieval_world import RetrievalBackend, new_retrieval_embedder


async def graph_retrieval_backend(backend: GraphBackend) -> RetrievalBackend:
    """Bind the retrieval scenario ports to a fresh graph tenant.

    The tenant is erased when the backend's fixture tears down.
    """
    handle = await backend.open_tenant()
    store = backend.store
    tenant_keyed = backend.tenant_keyed
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

    return RetrievalBackend(
        vectors=vectors,
        embedder=embedder,
        episodes=GraphEpisodeStore(runner, tenant_keyed=tenant_keyed),
        resolver_store=GraphResolverStore(runner, tenant_keyed=tenant_keyed),
        temporal_store=GraphTemporalStore(store, handle, tenant_keyed=tenant_keyed),
        ontology_store=GraphOntologyStore(store, handle, tenant_keyed=tenant_keyed),
        keywords=GraphKeywordSearch(runner),
        prune_orphan_vectors=prune_orphan_vectors,
    )
