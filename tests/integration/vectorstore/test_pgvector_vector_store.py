"""The pgvector store: shared contract, validity mirror and live Postgres behaviour."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from ontogr_mem.vectorstore import (
    CapabilityError,
    EmbedItem,
    FailReason,
    IndexDriftError,
    NodeExtras,
    QueryTimeoutError,
    SearchFilters,
    SpaceMismatchError,
    TemporalFilter,
    Validity,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.models import Row
from ontogr_mem.vectorstore.pgvector import (
    InvalidIdentifierError,
    PgBackend,
    PgVectorSettings,
    TenantNotProvisionedError,
    build_manifest,
    create_pool,
    drop_tenant,
    provision,
    schema_name,
)
from ontogr_mem.vectorstore.pgvector.queries import build_search
from ontogr_mem.vectorstore.pgvector.schema import quote_ident
from tests.contracts.vector_store import (
    GraphVectorStoreContract,
    edge_item,
    node_item,
)
from tests.contracts.vector_store_validity import ValidityMirrorContract
from tests.support.embedder import MockEmbedder, gateway
from tests.support.pg_graph import FakeGraph
from tests.support.pgvector import open_pg_env, specs_for
from tests.support.tenants import new_tenant
from tests.support.vectors import A, Factory, VectorHarness, fixed_embedder, ready


NODE, EDGE = VectorKind.NODE, VectorKind.EDGE
DIM = 3


@dataclass
class PgHarness(VectorHarness):
    """A harness with the fake graph and the pgvector backend behind the store."""

    graph: FakeGraph | None = None
    backend: PgBackend | None = None


class TestPgVectorStore(GraphVectorStoreContract, ValidityMirrorContract):
    """`VectorStore` over PostgreSQL satisfies the graph contract and mirror."""

    @pytest.fixture
    async def make_harness(self, pg_pool) -> AsyncGenerator[Factory]:
        """Return a factory for harnesses on fresh tenant schemas."""
        tenants: list[str] = []

        async def make(embedder: MockEmbedder | None) -> VectorHarness:
            chosen = embedder or fixed_embedder()
            tenant = new_tenant()
            tenants.append(tenant)
            specs = specs_for(chosen)
            await provision(pg_pool, tenant, specs)
            quoted = quote_ident(schema_name(tenant))
            async with pg_pool.acquire() as conn:
                await conn.execute(f"DELETE FROM {quoted}.vec_space")
            graph = FakeGraph()
            backend = PgBackend(pg_pool, tenant, specs, PgVectorSettings())
            store = VectorStore(backend, gateway(chosen), graph=graph)

            async def seed(kind: VectorKind, ids, **props: object) -> None:
                items: list[EmbedItem] = []
                for object_id in ids:
                    if kind is NODE:
                        items.append(
                            EmbedItem(
                                object_id,
                                object_id,
                                NodeExtras(
                                    props.get("entity_type"),
                                    props.get("status"),
                                ),
                            )
                        )
                    elif kind is EDGE:
                        validity = props.get("validity") or Validity("active", 0, 0)
                        items.append(
                            edge_item(
                                object_id,
                                object_id,
                                validity=validity,
                                subject=props.get("subject_id"),
                                target=props.get("object_id"),
                                relation=props.get("relation_type_id"),
                            )
                        )
                    else:
                        items.append(EmbedItem(object_id, object_id))
                graph.add(kind, *items)

            def reopen(other: MockEmbedder) -> VectorStore:
                other_backend = PgBackend(
                    pg_pool,
                    tenant,
                    specs_for(other),
                    PgVectorSettings(),
                )
                return VectorStore(other_backend, gateway(other), graph=graph)

            async def drop_index(kind: VectorKind) -> None:
                names = {
                    NODE: ["vec_node_hnsw"],
                    EDGE: ["vec_edge_hnsw_cur", "vec_edge_hnsw_all"],
                }[kind]
                quoted = quote_ident(schema_name(tenant))
                async with pg_pool.acquire() as conn:
                    for name in names:
                        await conn.execute(f"DROP INDEX {quoted}.{name}")

            return PgHarness(
                store, seed, reopen, drop_index, graph=graph, backend=backend
            )

        yield make
        for tenant in tenants:
            await drop_tenant(pg_pool, tenant)

    async def test_known_at_search_raises_capability_error(self, make_harness):
        """A store without a graph refuses past-belief search."""
        h = await ready(make_harness)
        h.store.graph = None

        with pytest.raises(CapabilityError):
            await h.store.search(
                EDGE,
                A,
                k=5,
                filters=SearchFilters(temporal=TemporalFilter(known_at_us=5)),
            )


async def open_tenant(
    pool,
    embedder: MockEmbedder | None = None,
    *,
    tenant: str | None = None,
    **settings: object,
) -> tuple[str, FakeGraph, PgBackend, VectorStore]:
    """Provision a fresh tenant and return its graph, backend and ready store."""
    env = await open_pg_env(
        pool,
        embedder,
        tenant=tenant,
        settings=PgVectorSettings(**settings),
    )
    return env.tenant, env.graph, env.backend, env.store


async def _plan(pool, tenant: str, kind: VectorKind, filters: SearchFilters) -> str:
    """Return the EXPLAIN plan of a search under a forced generic plan."""
    sql, params = build_search(
        schema_name(tenant),
        kind,
        DIM,
        vector=A,
        k=5,
        filters=filters,
        min_score=None,
    )
    decl = ",".join(["text", "int"] + ["bigint"] * (len(params) - 2))
    literals = [
        "'" + value + "'" if isinstance(value, str) else str(value) for value in params
    ]
    name = "probe_" + new_tenant()
    async with pool.acquire() as conn:
        try:
            await conn.execute("SET plan_cache_mode = force_generic_plan")
            await conn.execute("SET enable_seqscan = off")
            await conn.execute("SET enable_bitmapscan = off")
            await conn.execute(f"PREPARE {name}({decl}) AS {sql}")
            rows = await conn.fetch(f"EXPLAIN EXECUTE {name}({','.join(literals)})")
            return "\n".join(row[0] for row in rows)
        finally:
            await conn.execute(f"DEALLOCATE {name}")
            for setting in ("plan_cache_mode", "enable_seqscan", "enable_bitmapscan"):
                await conn.execute(f"RESET {setting}")


async def hold_table_lock(pool, tenant: str):
    """Lock `vec_node` until the returned release callback runs."""
    manager = pool.acquire()
    conn = await manager.__aenter__()
    transaction = conn.transaction()
    await transaction.__aenter__()
    await conn.execute(
        f"LOCK TABLE {quote_ident(schema_name(tenant))}.vec_node "
        "IN ACCESS EXCLUSIVE MODE"
    )

    async def release() -> None:
        await transaction.__aexit__(None, None, None)
        await manager.__aexit__(None, None, None)

    return release


class TestPgVectorProvisioning:
    """Schema, tenancy and drift against a live server."""

    async def test_concurrent_provisioning_converges(self, pg_pool):
        """Eight callers create one schema and one set of indexes."""
        tenant = new_tenant()
        specs = specs_for(fixed_embedder())
        try:
            await asyncio.gather(*(provision(pg_pool, tenant, specs) for _ in range(8)))
            await PgBackend(pg_pool, tenant, specs).ensure_ready(list(specs.values()))
            async with pg_pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                    "AND indexname LIKE 'vec_%_hnsw%'",
                    schema_name(tenant),
                )
            assert count == 4
        finally:
            await drop_tenant(pg_pool, tenant)

    async def test_provisioning_is_idempotent(self, pg_pool):
        """Provisioning twice leaves a usable tenant."""
        tenant = new_tenant()
        specs = specs_for(fixed_embedder())
        try:
            await provision(pg_pool, tenant, specs)
            await provision(pg_pool, tenant, specs)
            await PgBackend(pg_pool, tenant, specs).ensure_ready(list(specs.values()))
        finally:
            await drop_tenant(pg_pool, tenant)

    async def test_hostile_tenant_ids_are_refused(self):
        """A tenant id that could break the schema name is rejected."""
        for tenant in ("", "a.b", 'a"b', "a" * 61, "../x"):
            with pytest.raises(InvalidIdentifierError):
                schema_name(tenant)

    async def test_tenant_isolation_through_one_connection(self, pg_pool):
        """Two tenants share a pool but never each other's rows."""
        pool = await create_pool(PgVectorSettings(min_size=1, max_size=1))
        one, two = new_tenant(), new_tenant()
        specs = specs_for(fixed_embedder())
        try:
            for tenant in (one, two):
                await provision(pool, tenant, specs)
            _, graph_one, _, store_one = await open_tenant(pool, tenant=one)
            _, graph_two, _, store_two = await open_tenant(pool, tenant=two)
            graph_one.add(NODE, node_item("only-in-one", "a"))
            graph_two.add(NODE, node_item("only-in-two", "a"))
            await store_one.upsert(NODE, [node_item("only-in-one", "a")])
            await store_two.upsert(NODE, [node_item("only-in-two", "a")])

            assert [hit.id for hit in await store_one.search(NODE, A, k=5)] == [
                "only-in-one"
            ]
            assert [hit.id for hit in await store_two.search(NODE, A, k=5)] == [
                "only-in-two"
            ]
        finally:
            await drop_tenant(pool, one)
            await drop_tenant(pool, two)
            await pool.close()

    async def test_unprovisioned_tenant_is_an_error_not_an_empty_result(self, pg_pool):
        """A missing schema raises instead of returning no hits."""
        specs = specs_for(fixed_embedder())
        backend = PgBackend(pg_pool, new_tenant(), specs, PgVectorSettings())

        with pytest.raises(TenantNotProvisionedError):
            await backend.ensure_ready(list(specs.values()))

    async def test_search_after_tenant_drop_raises(self, pg_pool):
        """A dropped schema is reported as not provisioned."""
        tenant, _, _, store = await open_tenant(pg_pool)
        await drop_tenant(pg_pool, tenant)

        with pytest.raises(TenantNotProvisionedError):
            await store.search(NODE, A, k=1)

    async def test_dimension_drift_against_the_stored_space(self, pg_pool):
        """A stored space with another dimension is refused."""
        tenant = new_tenant()
        specs = specs_for(fixed_embedder())
        try:
            await provision(pg_pool, tenant, specs)
            other = MockEmbedder(dim=4)
            backend = PgBackend(pg_pool, tenant, specs_for(other))
            store = VectorStore(backend, gateway(other), graph=FakeGraph())
            with pytest.raises(SpaceMismatchError):
                await store.ensure_ready()
        finally:
            await drop_tenant(pg_pool, tenant)

    async def test_changed_index_definition_is_drift(self, pg_pool):
        """An index built with other parameters is not silently accepted."""
        tenant, _, _, _ = await open_tenant(pg_pool)
        quoted = quote_ident(schema_name(tenant))
        async with pg_pool.acquire() as conn:
            await conn.execute(f"DROP INDEX {quoted}.vec_node_hnsw")
            await conn.execute(
                f"CREATE INDEX vec_node_hnsw ON {quoted}.vec_node "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 8, ef_construction = 32)"
            )
        specs = specs_for(fixed_embedder())
        backend = PgBackend(pg_pool, tenant, specs, PgVectorSettings())

        with pytest.raises(IndexDriftError):
            await backend.ensure_ready(list(specs.values()))


class TestPgVectorPlans:
    """Planner behaviour that only a real server can show."""

    async def test_current_facts_query_uses_the_partial_index(self, pg_pool):
        """Under a generic plan the current-facts predicate proves the partial index."""
        tenant, graph, _, store = await open_tenant(pg_pool)
        graph.add(EDGE, *(edge_item(f"f{i}", f"fact {i}") for i in range(200)))
        await store.upsert(EDGE, [edge_item(f"f{i}", f"fact {i}") for i in range(200)])
        quoted = quote_ident(schema_name(tenant))
        async with pg_pool.acquire() as conn:
            await conn.execute(f"DROP INDEX {quoted}.vec_edge_time")
            await conn.execute(f"DROP INDEX {quoted}.vec_edge_hnsw_all")
            await conn.execute(f"ANALYZE {quoted}.vec_edge")
        plan = await _plan(pg_pool, tenant, EDGE, SearchFilters())

        assert "vec_edge_hnsw_cur" in plan, plan

        async with pg_pool.acquire() as conn:
            await conn.execute(f"DROP INDEX {quoted}.vec_edge_hnsw_cur")
        control = await _plan(pg_pool, tenant, EDGE, SearchFilters())

        assert "Seq Scan" in control and "hnsw" not in control, control

    async def test_text_cast_query_uses_the_hnsw_index(self, pg_pool):
        """Casting the parameter text to a vector keeps the index usable."""
        tenant, graph, _, store = await open_tenant(pg_pool)
        graph.add(NODE, *(EmbedItem(f"n{i}", f"name {i}") for i in range(200)))
        await store.upsert(NODE, [node_item(f"n{i}", f"name {i}") for i in range(200)])
        async with pg_pool.acquire() as conn:
            await conn.execute(f"ANALYZE {quote_ident(schema_name(tenant))}.vec_node")

        plan = await _plan(pg_pool, tenant, NODE, SearchFilters())

        assert "vec_node_hnsw" in plan, plan


class TestPgVectorWrites:
    """Write paths that differ from the graph-native backends."""

    async def test_concurrent_identical_upserts_write_one_row(self, pg_pool):
        """Ten writers of the same text converge on one stored vector."""
        _, graph, _, store = await open_tenant(pg_pool)
        graph.add(NODE, EmbedItem("n1", "same"))

        results = await asyncio.gather(
            *(store.upsert(NODE, [node_item("n1", "same")]) for _ in range(10))
        )

        assert sum(result.written for result in results) == 1
        assert await store.exists(NODE, ["n1"]) == {"n1"}

    async def test_bad_row_is_reported_and_good_rows_survive(self, pg_pool):
        """A row the server rejects does not take the batch down."""
        _, graph, backend, _ = await open_tenant(pg_pool)
        graph.add(
            NODE,
            EmbedItem("ok1", "a"),
            EmbedItem("bad", "a"),
            EmbedItem("ok2", "a"),
        )
        rows = [
            Row("ok1", "h1", tuple(A)),
            Row("bad", "h2", (1.0, 0.0)),
            Row("ok2", "h3", tuple(A)),
        ]

        result = await backend.write(NODE, rows)

        assert result.written == 2
        assert result.missing == ()
        assert [(item.id, item.reason) for item in result.rejected] == [
            ("bad", FailReason.REJECTED)
        ]
        assert await backend.exists(NODE, ["ok1", "bad", "ok2"]) == {"ok1", "ok2"}

    async def test_halfvec_is_used_above_2000_dimensions(self, pg_pool):
        """A 2,500-dimensional space uses a halfvec column and index."""
        embedder = MockEmbedder(dim=2500, model_id="mock")
        tenant, graph, backend, store = await open_tenant(pg_pool, embedder)
        graph.add(NODE, EmbedItem("a", "alpha"), EmbedItem("b", "beta"))
        await store.upsert(NODE, [node_item("a", "alpha"), node_item("b", "beta")])
        async with pg_pool.acquire() as conn:
            column_type = await conn.fetchval(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = to_regclass($1) AND attname = 'embedding'",
                f"{quote_ident(schema_name(tenant))}.vec_node",
            )
        (query,) = await embedder.embed_batch(["alpha"])

        hits = await store.search(NODE, query, k=2)

        assert column_type == "halfvec(2500)"
        assert hits[0].id == "a"
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    async def test_manifest_builds_four_hnsw_indexes(self, pg_pool):
        """The manifest and the server agree on the HNSW index count."""
        tenant, _, _, _ = await open_tenant(pg_pool)
        async with pg_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                "AND indexdef LIKE '%USING hnsw%'",
                schema_name(tenant),
            )

        assert count == 4
        manifest = build_manifest(schema_name(tenant), specs_for(fixed_embedder()))
        assert len([spec for spec in manifest if "hnsw" in spec.name]) == 4


class TestPgVectorTimeouts:
    """Statement time limits and cancellation safety."""

    async def test_blocked_search_hits_the_statement_timeout(self, pg_pool):
        """A search that waits on a lock raises `QueryTimeoutError`."""
        tenant, _, backend, store = await open_tenant(pg_pool)
        backend.settings = backend.settings.model_copy(update={"search_timeout_s": 0.3})
        release = await hold_table_lock(pg_pool, tenant)
        try:
            with pytest.raises(QueryTimeoutError):
                await store.search(NODE, A, k=1)
        finally:
            await release()

        assert await store.search(NODE, A, k=1) == []

    async def test_cancelled_search_does_not_poison_the_pool(self, pg_pool):
        """A cancelled search releases or terminates its connection."""
        tenant, graph, _, store = await open_tenant(pg_pool)
        release = await hold_table_lock(pg_pool, tenant)
        task = asyncio.create_task(store.search(NODE, A, k=1))
        await asyncio.sleep(0.4)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await release()
        for _ in range(20):
            async with pg_pool.acquire() as conn:
                waiting = await conn.fetchval(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock'"
                )
            if waiting == 0:
                break
            await asyncio.sleep(0.1)
        graph.add(NODE, EmbedItem("n1", "after cancel"))

        await store.upsert(NODE, [node_item("n1", "after cancel")])

        assert await store.exists(NODE, ["n1"]) == {"n1"}
