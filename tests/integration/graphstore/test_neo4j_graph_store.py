"""The graph store contract on Neo4j."""

import pytest

from ontogr_mem.graphstore import (
    ConstraintViolationError,
    Edge,
    Node,
    NodeSelector,
    QuerySpec,
)
from tests.contracts.graph_store import GraphStoreContract
from tests.contracts.graph_store_property_mode import PropertyModeContract


class TestNeo4jGraphStore(GraphStoreContract, PropertyModeContract):
    """Neo4jGraphStore satisfies the graph store contract."""

    @pytest.fixture
    def store(self, neo4j_store):
        """The store under test."""
        return neo4j_store

    @pytest.fixture
    def handle(self, neo4j_handle):
        """A fresh tenant."""
        return neo4j_handle

    @pytest.fixture
    def store_factory(self, neo4j_store_factory):
        """Creates more stores on the same server."""
        return neo4j_store_factory

    @pytest.fixture
    def unreachable_store(self, neo4j_unreachable_store):
        """A store pointed at a closed port."""
        return neo4j_unreachable_store

    def create_cypher(self, node_id: str) -> str:
        """Create an `Entity` that belongs to the runner's tenant."""
        return f"CREATE (:Entity {{_tenant: $tenant, id: '{node_id}'}})"

    def count_cypher(self) -> str:
        """Count the runner's tenant's `Entity` nodes."""
        return "MATCH (n:Entity {_tenant: $tenant}) RETURN count(n) AS c"

    def guard_props_cypher(self, node_id: str) -> str:
        """Read the guard node's stored properties, tenant and lock included."""
        return (
            f"MATCH (n:Slot {{_tenant: $tenant, id: '{node_id}'}}) "
            "RETURN properties(n) AS props"
        )

    async def drop_range_index(self, store, handle) -> None:
        """Drop the `Drift.name` index."""
        await store.runner(handle).run(
            QuerySpec("DROP INDEX am_ix_Drift_name", auto_commit=True)
        )

    async def drop_unique_constraint(self, store, handle) -> None:
        """Drop the `Drift.email` constraint."""
        await store.runner(handle).run(
            QuerySpec("DROP CONSTRAINT am_uq_Drift_email", auto_commit=True)
        )

    async def test_open_does_not_create_meta_nodes(self, neo4j_store, neo4j_handle):
        """Schema provisioning does not add bookkeeping labels to tenant data."""
        stats = await neo4j_store.stats(neo4j_handle)

        assert "__Meta" not in stats.labels

    async def test_get_nodes_returns_labels_in_a_stable_order(
        self, neo4j_store, neo4j_handle
    ):
        """NodeRow labels do not depend on server label order."""
        await neo4j_store.upsert_nodes(
            neo4j_handle, [Node("a", "Entity", {}, ("Zebra", "Alpha"))]
        )

        (row,) = await neo4j_store.get_nodes(neo4j_handle, "Entity", ["a"])

        assert row.labels == ("Alpha", "Entity", "Zebra")

    async def test_delete_without_detach_raises_constraint_violation(
        self, neo4j_store, neo4j_handle
    ):
        """A connected-node delete uses the public constraint error type."""
        await neo4j_store.upsert_nodes(
            neo4j_handle, [Node("a", "Entity"), Node("b", "Entity")]
        )
        await neo4j_store.upsert_edges(
            neo4j_handle, [Edge("Entity", "a", "Entity", "b", "MENTIONS")]
        )

        with pytest.raises(ConstraintViolationError):
            await neo4j_store.delete(
                neo4j_handle, NodeSelector("Entity", ["a"]), detach=False
            )
