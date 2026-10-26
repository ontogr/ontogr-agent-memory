"""Server risks of Neo4j that the adapter must survive."""

import asyncio

import pytest

from ontogr_mem.graphstore import GraphHandle, Node, QuerySpec, QueryTimeoutError
from tests.support.graph_schema import TEST_PART
from tests.support.tenants import new_tenant


SLOW_READ = "UNWIND range(1, 6000) AS x UNWIND range(1, 6000) AS y RETURN count(y) AS c"


class TestNeo4jRisks:
    """Concurrency and timeout behavior."""

    async def test_one_pool_serves_400_concurrent_queries(
        self, neo4j_store, neo4j_handle
    ):
        """Mixed writes and reads over one 16 connection pool all succeed."""

        async def task(i: int) -> str:
            node_id = f"n{i:04d}"
            await neo4j_store.upsert_nodes(
                neo4j_handle, [Node(node_id, "Item", {"rank": i})]
            )
            (row,) = await neo4j_store.get_nodes(neo4j_handle, "Item", [node_id])
            assert row.props["rank"] == i
            return row.id

        ids = await asyncio.gather(*[task(i) for i in range(200)])

        assert ids == [f"n{i:04d}" for i in range(200)]
        assert (await neo4j_store.stats(neo4j_handle)).labels["Item"] == 200

    async def test_concurrent_tenants_do_not_cross(self, neo4j_store):
        """Four tenants written and read at the same time see only their own rows."""
        handles = [await neo4j_store.open(new_tenant(), [TEST_PART]) for _ in range(4)]
        try:

            async def task(handle: GraphHandle, i: int) -> None:
                node_id = f"{handle.tenant}-{i}"
                await neo4j_store.upsert_nodes(handle, [Node(node_id, "Item", {})])
                rows = await neo4j_store.get_nodes(handle, "Item", [node_id])
                assert [r.id for r in rows] == [node_id]

            await asyncio.gather(*[task(h, i) for h in handles for i in range(50)])

            for handle in handles:
                rows = await neo4j_store.runner(handle).run(
                    QuerySpec(
                        "MATCH (n:Item {_tenant: $tenant}) RETURN collect(n.id) AS ids",
                        read_only=True,
                        idempotent=True,
                    )
                )
                ids = rows.scalar("ids")
                assert len(ids) == 50
                assert all(i.startswith(handle.tenant) for i in ids)
        finally:
            for handle in handles:
                await neo4j_store.erase_tenant(handle.tenant)

    async def test_explicit_query_timeout_kills_a_slow_read(
        self, neo4j_store, neo4j_handle
    ):
        """A per-query timeout ends a slow read with a typed error."""
        with pytest.raises(QueryTimeoutError):
            await neo4j_store.runner(neo4j_handle).run(
                QuerySpec(SLOW_READ, read_only=True, timeout_ms=100)
            )
