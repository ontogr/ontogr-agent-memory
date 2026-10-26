"""The graph-native vector store contract on FalkorDB."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from ontogr_mem.graphstore import QuerySpec
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.falkordb.runner import GraphRunner
from ontogr_mem.graphstore.schemas import VECTOR
from ontogr_mem.vectorstore import (
    IndexDriftError,
    SearchFilters,
    Validity,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.graph.falkordb import FalkorDBVectorBackend
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


DROP_INDEX = {
    VectorKind.NODE: "DROP VECTOR INDEX FOR (n:Entity) ON (n.embedding)",
    VectorKind.EPISODE: "DROP VECTOR INDEX FOR (n:Episode) ON (n.embedding)",
    VectorKind.EDGE: "DROP VECTOR INDEX FOR ()-[e:FACT]->() ON (e.embedding)",
}
"""Index DDL the FalkorDB 4.22 server accepts for each kind."""


@dataclass
class FalkorHarness(VectorHarness):
    """A harness that can also read the tenant's graph directly."""

    runner: GraphRunner | None = None


class TestFalkorDBVectorStore(GraphVectorStoreContract):
    """FalkorDBVectorBackend satisfies the graph-native contract."""

    @pytest.fixture
    async def make_harness(
        self, falkordb_store: FalkorDBGraphStore
    ) -> AsyncGenerator[Factory]:
        """Return a factory for harnesses on fresh tenant graphs."""
        tenants: list[str] = []

        async def make(embedder: MockEmbedder | None) -> VectorHarness:
            chosen = embedder or fixed_embedder()
            handle = await falkordb_store.open(new_tenant(), [VECTOR])
            tenants.append(handle.tenant)
            backend = FalkorDBVectorBackend(falkordb_store, handle)
            store = VectorStore(backend, gateway(chosen))
            runner = falkordb_store.runner(handle)

            async def seed(kind: VectorKind, ids, **props: object) -> None:
                await seed_graph(falkordb_store, handle, kind, ids, **props)

            def reopen(other: MockEmbedder) -> VectorStore:
                return VectorStore(backend, gateway(other))

            async def drop_index(kind: VectorKind) -> None:
                await runner.run(
                    QuerySpec(
                        DROP_INDEX[kind],
                        idempotent=True,
                        tag="drop_vector_index",
                    )
                )

            return FalkorHarness(store, seed, reopen, drop_index, runner)

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)

    async def test_plain_list_embedding_is_never_written(self, make_harness):
        """A vector written through the store is a `vecf32` and is indexed."""
        h = await ready(make_harness)
        if h.runner is None:
            pytest.fail("harness has no graph runner")
        await h.seed(VectorKind.NODE, ["n1"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])

        hits = await h.store.search(VectorKind.NODE, A, k=5)
        stored = await h.runner.run(
            QuerySpec(
                "MATCH (n:Entity {id: $id}) RETURN typeof(n.embedding) AS kind",
                {"id": "n1"},
                read_only=True,
                idempotent=True,
                tag="read_vector_type",
            )
        )

        assert [hit.id for hit in hits] == ["n1"]
        assert stored.scalar("kind") == "Vectorf32"

    async def test_search_right_after_index_creation_raises_index_drift_then_succeeds(
        self, make_harness
    ):
        """A missing index is an error until `ensure_ready` recreates it."""
        h = await ready(make_harness)
        if h.drop_index is None:
            pytest.fail("harness has no index to drop")
        await h.seed(VectorKind.NODE, ["n1"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])
        await h.drop_index(VectorKind.NODE)

        with pytest.raises(IndexDriftError):
            await h.store.search(VectorKind.NODE, A, k=5)

        await h.store.ensure_ready()
        hits = await h.store.search(VectorKind.NODE, A, k=5)

        assert [hit.id for hit in hits] == ["n1"]

    async def test_status_filter_over_fetches_and_never_returns_fewer_than_k(
        self, make_harness
    ):
        """A closer crowd of closed nodes cannot crowd out the active ones."""
        h = await ready(make_harness)
        k = 3
        active = [f"active-{i}" for i in range(k)]
        closed = [f"closed-{i}" for i in range(40)]
        await h.seed(VectorKind.NODE, active, status="active")
        await h.seed(VectorKind.NODE, closed, status="closed")
        await h.store.upsert(
            VectorKind.NODE,
            [node_item(i, "half", status="active") for i in active]
            + [node_item(i, "a", status="closed") for i in closed],
        )

        hits = await h.store.search(
            VectorKind.NODE,
            A,
            k=k,
            filters=SearchFilters(statuses=frozenset({"active"})),
        )

        assert [hit.id for hit in hits] == active
        assert [hit.score for hit in hits] == pytest.approx([0.75] * k)

    async def test_expired_edges_do_not_crowd_out_current_ones(self, make_harness):
        """A closer crowd of expired facts cannot crowd out current facts."""
        h = await ready(make_harness)
        k = 3
        current = Validity("active", 0, 0)
        expired = Validity("active", 0, 0, expired_at_us=10)
        current_ids = [f"current-{i}" for i in range(k)]
        expired_ids = [f"expired-{i}" for i in range(40)]
        await seed_edges(h, current_ids, validity=current)
        await seed_edges(h, expired_ids, validity=expired)
        await h.store.upsert(
            VectorKind.EDGE,
            [edge_item(i, "half", validity=current) for i in current_ids]
            + [edge_item(i, "a", validity=expired) for i in expired_ids],
        )

        hits = await h.store.search(VectorKind.EDGE, A, k=k)

        assert [hit.id for hit in hits] == current_ids
        assert [hit.score for hit in hits] == pytest.approx([0.75] * k)
