"""The vector store contract on Neo4j, plus the index behavior it relies on."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.graphstore import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.graphstore.schemas import VECTOR
from ontogr_mem.vectorstore import (
    Validity,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.graph.base import LABELS
from ontogr_mem.vectorstore.graph.neo4j import Neo4jVectorBackend, index_name
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.contracts.vector_store import (
    GraphVectorStoreContract,
    edge_item,
    node_item,
    seed_edges,
)
from tests.support.embedder import MockEmbedder, gateway
from tests.support.tenants import new_tenant
from tests.support.vector_graph import seed_graph
from tests.support.vectors import (
    A,
    Factory,
    VectorHarness,
    fixed_embedder,
    ready,
)


class TestNeo4jVectorStore(GraphVectorStoreContract):
    """Neo4jVectorBackend satisfies the graph vector store contract."""

    @pytest.fixture
    async def make_harness(
        self, neo4j_store: Neo4jGraphStore
    ) -> AsyncGenerator[Factory, None]:
        """Return a factory for stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make(embedder: MockEmbedder | None) -> VectorHarness:
            handle = await neo4j_store.open(new_tenant(), [VECTOR])
            tenants.append(handle.tenant)

            def store_for(other: MockEmbedder) -> VectorStore:
                backend = Neo4jVectorBackend(neo4j_store, handle)
                return VectorStore(backend, gateway(other))

            async def seed(kind: VectorKind, ids: list[str], **props: object) -> None:
                await seed_graph(neo4j_store, handle, kind, ids, **props)

            async def drop(kind: VectorKind) -> None:
                await neo4j_store.runner(handle).run(
                    QuerySpec(
                        f"DROP INDEX {quote(index_name(LABELS[kind]))} IF EXISTS",
                        idempotent=True,
                        auto_commit=True,
                    )
                )

            chosen = embedder or fixed_embedder()
            return VectorHarness(store_for(chosen), seed, store_for, drop)

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)

    async def test_tenant_filter_is_inside_the_index_query(self, make_harness):
        """Identical vectors in two tenants never cross, even past the first k."""
        one = await ready(make_harness)
        two = await ready(make_harness)
        one_ids = [f"one-{i}" for i in range(2)]
        two_ids = [f"two-{i}" for i in range(10)]
        await one.seed(VectorKind.NODE, one_ids)
        await two.seed(VectorKind.NODE, two_ids)
        await one.store.upsert(VectorKind.NODE, [node_item(i, "a") for i in one_ids])
        await two.store.upsert(VectorKind.NODE, [node_item(i, "a") for i in two_ids])

        seen_one = await one.store.search(VectorKind.NODE, A, k=10)
        seen_two = await two.store.search(VectorKind.NODE, A, k=10)

        assert [hit.id for hit in seen_one] == one_ids
        assert [hit.id for hit in seen_two] == two_ids

    async def test_float_list_is_what_is_stored(self, neo4j_store: Neo4jGraphStore):
        """The embedding property is a list of floats, and KNN finds it."""
        handle = await neo4j_store.open(new_tenant(), [VECTOR])
        try:
            await seed_graph(neo4j_store, handle, VectorKind.NODE, ["n1"])
            store = VectorStore(
                Neo4jVectorBackend(neo4j_store, handle), gateway(fixed_embedder())
            )
            await store.ensure_ready()
            await store.upsert(VectorKind.NODE, [node_item("n1", "a")])

            (row,) = await neo4j_store.get_nodes(handle, "Entity", ["n1"])
            stored = row.props["embedding"]

            assert isinstance(stored, list)
            assert all(isinstance(value, float) for value in stored)
            assert stored == pytest.approx(A, abs=1e-6)
            hits = await store.search(VectorKind.NODE, A, k=1)
            assert [hit.id for hit in hits] == ["n1"]
        finally:
            await neo4j_store.erase_tenant(handle.tenant)

    async def test_quantization_none_gives_exact_scores(self, make_harness):
        """Scores match the exact live graph tenant to a tight tolerance."""
        h = await ready(make_harness)
        items = [
            node_item("i-a", "a"),
            node_item("i-half", "half"),
            node_item("i-orth", "orthogonal"),
            node_item("i-opp", "opposite"),
        ]
        await h.seed(VectorKind.NODE, [item.id for item in items])
        await h.store.upsert(VectorKind.NODE, items)
        twin = VectorStore(InMemoryBackend(), gateway(fixed_embedder()))
        await twin.ensure_ready()
        await twin.upsert(VectorKind.NODE, items)

        native = await h.store.search(VectorKind.NODE, A, k=10)
        exact = await twin.search(VectorKind.NODE, A, k=10)

        assert [hit.id for hit in native] == [hit.id for hit in exact]
        for left, right in zip(native, exact, strict=True):
            assert left.score == pytest.approx(right.score, abs=1e-6)

    async def test_many_expired_facts_do_not_hide_the_current_ones(self, make_harness):
        """Closer expired facts do not push the k current facts out of reach."""
        h = await ready(make_harness)
        expired = [f"expired-{i}" for i in range(20)]
        current = [f"current-{i}" for i in range(3)]
        old = Validity("active", 0, 0, expired_at_us=10)
        await seed_edges(h, expired, validity=old)
        await seed_edges(h, current)
        await h.store.upsert(
            VectorKind.EDGE,
            [edge_item(fact_id, "a", validity=old) for fact_id in expired],
        )
        await h.store.upsert(
            VectorKind.EDGE,
            [edge_item(fact_id, "half") for fact_id in current],
        )

        hits = await h.store.search(VectorKind.EDGE, A, k=3)

        assert sorted(hit.id for hit in hits) == current
