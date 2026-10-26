"""A provisioned pgvector tenant with a fake graph, store and outbox."""

from dataclasses import dataclass

from ontogr_mem.vectorstore import SpaceSpec, VectorKind, VectorStore
from ontogr_mem.vectorstore.pgvector import (
    PgBackend,
    PgOutbox,
    PgVectorSettings,
    provision,
)
from ontogr_mem.vectorstore.text import TEMPLATES
from tests.support.embedder import MockEmbedder, gateway
from tests.support.pg_graph import FakeGraph
from tests.support.tenants import new_tenant
from tests.support.vectors import fixed_embedder


@dataclass
class PgEnv:
    """One tenant's pgvector pieces plus the graph they mirror."""

    tenant: str
    graph: FakeGraph
    backend: PgBackend
    store: VectorStore
    outbox: PgOutbox


def specs_for(embedder: MockEmbedder) -> dict[VectorKind, SpaceSpec]:
    """Return the embedding space of every kind for `embedder`."""
    return {
        kind: SpaceSpec(
            kind,
            embedder.model_id,
            embedder.dim,
            TEMPLATES[kind].id,
            TEMPLATES[kind].version,
        )
        for kind in VectorKind
    }


async def open_pg_env(
    pool,
    embedder: MockEmbedder | None = None,
    *,
    tenant: str | None = None,
    settings: PgVectorSettings | None = None,
) -> PgEnv:
    """Provision a tenant and return a ready store, its backend and its outbox."""
    chosen = embedder or fixed_embedder()
    chosen_tenant = tenant or new_tenant()
    chosen_settings = settings or PgVectorSettings()
    specs = specs_for(chosen)
    await provision(pool, chosen_tenant, specs)
    graph = FakeGraph()
    backend = PgBackend(pool, chosen_tenant, specs, chosen_settings)
    store = VectorStore(backend, gateway(chosen), graph=graph)
    await store.ensure_ready()
    return PgEnv(
        chosen_tenant,
        graph,
        backend,
        store,
        PgOutbox(pool, chosen_tenant, chosen_settings),
    )
