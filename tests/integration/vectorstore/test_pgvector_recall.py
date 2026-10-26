"""Recall self-test: pass, fail and exact fallback against Postgres."""

from collections.abc import Sequence

import pytest

from ontogr_mem.vectorstore import (
    EmbedItem,
    NodeExtras,
    SearchFilters,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.pgvector import (
    PgBackend,
    PgVectorSettings,
    RecallBelowThresholdError,
    drop_tenant,
    provision,
    recall_selftest,
    require_recall,
)
from ontogr_mem.vectorstore.protocol import RawHit
from tests.contracts.vector_store import node_item
from tests.support.embedder import gateway
from tests.support.pg_graph import FakeGraph
from tests.support.pgvector import specs_for
from tests.support.tenants import new_tenant
from tests.support.vectors import HALF, OPPOSITE, ORTHOGONAL, A, fixed_embedder


NODE = VectorKind.NODE
QUERIES = [A, HALF, ORTHOGONAL, OPPOSITE, [0.3, 0.4, 0.5]]


class LossyBackend(PgBackend):
    """An ANN path that drops the last `drop` hits; the exact path is whole."""

    def __init__(self, *args, drop: int, **kwargs) -> None:
        """Create the backend and record how many hits the index loses."""
        super().__init__(*args, **kwargs)
        self.drop = drop
        self.exact_flags: list[bool] = []

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
        *,
        exact: bool = False,
    ) -> list[RawHit]:
        """Search, then lose the tail unless the exact scan ran."""
        effective = exact or kind in self.force_exact
        self.exact_flags.append(effective)
        hits = await super().search(kind, vector, k, filters, min_score, exact=exact)
        if effective:
            return hits
        return hits[: max(len(hits) - self.drop, 0)]


async def lossy_store(pool, drop: int) -> tuple[LossyBackend, VectorStore, str]:
    """Return a backend, a ready store with 20 nodes, and its tenant."""
    embedder = fixed_embedder()
    tenant = new_tenant()
    specs = specs_for(embedder)
    await provision(pool, tenant, specs)
    graph = FakeGraph()
    backend = LossyBackend(pool, tenant, specs, PgVectorSettings(), drop=drop)
    store = VectorStore(backend, gateway(embedder), graph=graph)
    await store.ensure_ready()
    graph.add(
        NODE,
        *(EmbedItem(f"n{i}", f"name {i}", NodeExtras()) for i in range(20)),
    )
    await store.upsert(NODE, [node_item(f"n{i}", f"name {i}") for i in range(20)])
    return backend, store, tenant


async def test_perfect_index_passes_and_keeps_the_ann_path(pg_pool):
    """A whole index reaches full recall and is not switched to exact."""
    backend, store, tenant = await lossy_store(pg_pool, drop=0)
    try:
        report = await recall_selftest(store, NODE, QUERIES, k=10, min_recall=0.95)

        assert report.passed
        assert report.recall == pytest.approx(1.0)
        assert report.shortfalls == 0
        assert NODE not in backend.force_exact

        backend.exact_flags.clear()
        await store.search(NODE, QUERIES[0], k=5)

        assert backend.exact_flags == [False]
    finally:
        await drop_tenant(pg_pool, tenant)


async def test_lossy_index_fails_and_switches_the_backend_to_exact(pg_pool):
    """A short index fails the test and subsequent searches use the exact scan."""
    backend, store, tenant = await lossy_store(pg_pool, drop=4)
    try:
        report = await recall_selftest(store, NODE, QUERIES, k=10, min_recall=0.95)

        assert not report.passed
        assert report.recall == pytest.approx(0.6)
        assert report.shortfalls == len(QUERIES)
        assert NODE in backend.force_exact

        backend.exact_flags.clear()
        hits = await store.search(NODE, QUERIES[0], k=5)

        assert backend.exact_flags == [True]
        assert len(hits) == 5
        with pytest.raises(RecallBelowThresholdError):
            require_recall(report)
    finally:
        await drop_tenant(pg_pool, tenant)


async def test_failure_can_leave_the_backend_alone(pg_pool):
    """A caller that reports the failure itself can keep the index path."""
    backend, store, tenant = await lossy_store(pg_pool, drop=4)
    try:
        await recall_selftest(
            store,
            NODE,
            QUERIES,
            k=10,
            min_recall=0.95,
            force_exact_on_failure=False,
        )

        assert NODE not in backend.force_exact
    finally:
        await drop_tenant(pg_pool, tenant)


async def test_no_queries_is_a_pass(pg_pool):
    """An empty sample cannot lower recall."""
    _, store, tenant = await lossy_store(pg_pool, drop=0)
    try:
        report = await recall_selftest(store, NODE, [], k=10, min_recall=0.9)

        assert report.passed
    finally:
        await drop_tenant(pg_pool, tenant)
