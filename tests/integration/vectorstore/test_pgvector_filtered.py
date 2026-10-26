"""Filtered ANN on a table large enough for the index to matter."""

import random

from ontogr_mem.vectorstore import (
    NodeExtras,
    SearchFilters,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.models import Row
from ontogr_mem.vectorstore.pgvector import (
    PgBackend,
    PgVectorSettings,
    drop_tenant,
    provision,
    recall_selftest,
    schema_name,
)
from ontogr_mem.vectorstore.pgvector.schema import quote_ident
from ontogr_mem.vectorstore.vecmath import to_unit
from tests.support.embedder import MockEmbedder, gateway
from tests.support.pg_graph import FakeGraph
from tests.support.pgvector import specs_for
from tests.support.tenants import new_tenant


NODE = VectorKind.NODE
ROWS = 20_000
TYPES = 100
K = 10


def rand_unit(rng: random.Random) -> list[float]:
    """Return a random unit vector of length three."""
    values = [rng.gauss(0, 1) for _ in range(3)]
    length = sum(value * value for value in values) ** 0.5
    return [value / length for value in values]


async def test_filtered_recall_is_restored_by_iterative_scan(pg_pool):
    """Without iterative scan a selective filter falls short; with it, recall holds."""
    embedder = MockEmbedder()
    tenant = new_tenant()
    specs = specs_for(embedder)
    await provision(pg_pool, tenant, specs)
    try:
        rng = random.Random(7)
        rows = [
            Row(
                f"n{i:06d}",
                f"h{i}",
                tuple(rand_unit(rng)),
                NodeExtras(i % TYPES),
            )
            for i in range(ROWS)
        ]
        writer = PgBackend(
            pg_pool, tenant, specs, PgVectorSettings(exact_threshold=1000)
        )
        for start in range(0, ROWS, 1000):
            await writer.write(NODE, rows[start : start + 1000])
        async with pg_pool.acquire() as conn:
            await conn.execute(f"ANALYZE {quote_ident(schema_name(tenant))}.vec_node")
        filters = SearchFilters(entity_types=frozenset({7}))
        query = to_unit(rand_unit(rng), 3)

        off = PgBackend(
            pg_pool,
            tenant,
            specs,
            PgVectorSettings(exact_threshold=1000, iterative_scan="off"),
        )
        strict = PgBackend(
            pg_pool,
            tenant,
            specs,
            PgVectorSettings(exact_threshold=1000, iterative_scan="strict_order"),
        )
        for backend in (off, strict):
            await backend.ensure_ready(list(specs.values()))
        short = await off.search(NODE, query, K, filters, None)
        full = await strict.search(NODE, query, K, filters, None)

        assert len(short) < K, short
        assert len(full) == K
        assert all(int(hit[0][1:]) % TYPES == 7 for hit in full)

        store = VectorStore(strict, gateway(embedder), graph=FakeGraph())
        await store.ensure_ready()
        queries = [rand_unit(rng) for _ in range(20)]
        report = await recall_selftest(
            store,
            NODE,
            queries,
            k=K,
            min_recall=0.95,
            filters=filters,
        )

        assert report.passed, report
        assert report.shortfalls == 0
    finally:
        await drop_tenant(pg_pool, tenant)
