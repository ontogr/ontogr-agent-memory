"""Server risks of FalkorDB that the adapter must survive."""

import asyncio

import pytest

from ontogr_mem.graphstore import (
    CircuitOpenError,
    DatabaseUnavailableError,
    GraphHandle,
    Node,
    QuerySpec,
    QueryTimeoutError,
)
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from tests.support.falkordb import falkordb_test_settings
from tests.support.graph_schema import TEST_PART
from tests.support.proxy import TcpProxy
from tests.support.tenants import new_tenant


SLOW_READ = "UNWIND range(1, 6000) AS x UNWIND range(1, 6000) AS y RETURN count(y) AS c"


class TestFalkorDBRisks:
    """Concurrency, timeout and outage behavior."""

    async def test_one_pool_serves_400_concurrent_queries(
        self, falkordb_store, falkordb_handle
    ):
        """Mixed writes and reads over one 16 connection pool all succeed."""

        async def task(i: int) -> str:
            node_id = f"n{i:04d}"
            await falkordb_store.upsert_nodes(
                falkordb_handle, [Node(node_id, "Item", {"rank": i})]
            )
            (row,) = await falkordb_store.get_nodes(falkordb_handle, "Item", [node_id])
            assert row.props["rank"] == i
            return row.id

        ids = await asyncio.gather(*[task(i) for i in range(200)])

        assert ids == [f"n{i:04d}" for i in range(200)]
        assert (await falkordb_store.stats(falkordb_handle)).labels["Item"] == 200

    async def test_concurrent_tenants_do_not_cross(self, falkordb_store):
        """Four tenants written and read at the same time see only their own rows."""
        handles = [
            await falkordb_store.open(new_tenant(), [TEST_PART]) for _ in range(4)
        ]
        try:

            async def task(handle: GraphHandle, i: int) -> None:
                node_id = f"{handle.tenant}-{i}"
                await falkordb_store.upsert_nodes(handle, [Node(node_id, "Item", {})])
                rows = await falkordb_store.get_nodes(handle, "Item", [node_id])
                assert [r.id for r in rows] == [node_id]

            await asyncio.gather(*[task(h, i) for h in handles for i in range(50)])

            for handle in handles:
                rows = await falkordb_store.runner(handle).run(
                    QuerySpec(
                        "MATCH (n:Item) RETURN collect(n.id) AS ids",
                        read_only=True,
                        idempotent=True,
                    )
                )
                ids = rows.scalar("ids")
                assert len(ids) == 50
                assert all(i.startswith(handle.tenant) for i in ids)
        finally:
            for handle in handles:
                await falkordb_store.erase_tenant(handle.tenant)

    async def test_id_list_reads_on_indexed_labels(
        self, falkordb_store, falkordb_handle
    ):
        """Reading 200 nodes by id list on an indexed label returns the right rows."""
        await falkordb_store.upsert_nodes(
            falkordb_handle,
            [Node(f"e{i:03d}", "Entity", {"name": f"name {i}"}) for i in range(200)],
        )
        wanted = [f"e{i:03d}" for i in range(0, 200, 2)]

        rows = await falkordb_store.get_nodes(falkordb_handle, "Entity", wanted)

        assert sorted(r.id for r in rows) == wanted
        assert all(r.props["name"] == f"name {int(r.id[1:])}" for r in rows)

    async def test_explicit_query_timeout_kills_a_slow_read(
        self, falkordb_store, falkordb_handle
    ):
        """A per-query timeout ends a slow read with a typed error."""
        with pytest.raises(QueryTimeoutError):
            await falkordb_store.runner(falkordb_handle).run(
                QuerySpec(SLOW_READ, read_only=True, timeout_ms=100)
            )

    async def test_unreachable_server_opens_the_breaker_and_it_recovers(
        self, falkordb_store
    ):
        """Five failures open the breaker; it closes again once the server is back."""
        direct = falkordb_test_settings()
        proxy = TcpProxy(direct.host, direct.port)
        await proxy.start()
        store = FalkorDBGraphStore(
            falkordb_test_settings(
                port=proxy.port,
                max_attempts=1,
                breaker_failure_threshold=5,
                breaker_recovery_s=0.5,
            )
        )
        tenant = new_tenant()
        handle = await store.open(tenant, [TEST_PART])
        spec = QuerySpec("RETURN 1 AS x", read_only=True, idempotent=True)
        try:
            await proxy.stop()
            for _ in range(5):
                with pytest.raises(DatabaseUnavailableError):
                    await store.runner(handle).run(spec)

            with pytest.raises(CircuitOpenError):
                await store.runner(handle).run(spec)
            assert (await store.health()).breaker == "open"

            await proxy.start(proxy.port)
            await asyncio.sleep(0.6)
            rows = await store.runner(handle).run(spec)

            assert rows.rows == [{"x": 1}]
            assert (await store.health()).breaker == "closed"
        finally:
            await falkordb_store.erase_tenant(tenant)
            await store.close()
            await proxy.stop()
