"""The graph store contract on FalkorDB."""

import pytest

from ontogr_mem.graphstore import (
    ConstraintViolationError,
    Edge,
    Node,
    NodeSelector,
    QuerySpec,
)
from tests.contracts.graph_store import GraphStoreContract
from tests.support.falkordb import drop_unique_constraint


class TestFalkorDBGraphStore(GraphStoreContract):
    """FalkorDBGraphStore satisfies the graph store contract."""

    @pytest.fixture
    def store(self, falkordb_store):
        """The store under test."""
        return falkordb_store

    @pytest.fixture
    def handle(self, falkordb_handle):
        """A fresh tenant graph."""
        return falkordb_handle

    @pytest.fixture
    def store_factory(self, falkordb_store_factory):
        """Creates more stores on the same server."""
        return falkordb_store_factory

    @pytest.fixture
    def unreachable_store(self, falkordb_unreachable_store):
        """A store pointed at a closed port."""
        return falkordb_unreachable_store

    async def drop_range_index(self, store, handle) -> None:
        """Drop the `Drift.name` index."""
        await store.runner(handle).run(QuerySpec("DROP INDEX ON :Drift(name)"))

    async def drop_unique_constraint(self, store, handle) -> None:
        """Drop the `Drift.email` constraint."""
        await drop_unique_constraint(handle.graph, "Drift", "email")

    async def test_refused_multi_node_delete_keeps_every_selected_node(
        self, store, handle
    ):
        """A connected node prevents deletion of the whole selection."""
        await store.upsert_nodes(
            handle, [Node("a", "Entity"), Node("b", "Entity"), Node("c", "Entity")]
        )
        await store.upsert_edges(
            handle, [Edge("Entity", "a", "Entity", "b", "MENTIONS")]
        )

        with pytest.raises(ConstraintViolationError):
            await store.delete(handle, NodeSelector("Entity", ["a", "c"]), detach=False)

        rows = await store.get_nodes(handle, "Entity", ["a", "c"])
        assert [row.id for row in rows] == ["a", "c"]
        assert (await store.stats(handle)).edge_count == 1

    async def test_get_nodes_returns_labels_in_a_stable_order(self, store, handle):
        """NodeRow labels do not depend on server label order."""
        await store.upsert_nodes(handle, [Node("a", "Entity", {}, ("Zebra", "Alpha"))])

        (row,) = await store.get_nodes(handle, "Entity", ["a"])

        assert row.labels == ("Alpha", "Entity", "Zebra")
