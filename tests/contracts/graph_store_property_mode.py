"""Checks for stores that keep every tenant in one shared database.

A backend test module subclasses `PropertyModeContract` next to
`GraphStoreContract` and provides the same `store` and `handle` fixtures.
"""

import pytest

from ontogr_mem.graphstore import (
    Edge,
    EdgeSelector,
    Guard,
    Node,
    NodeSelector,
    ValidationError,
)
from tests.support.graph_schema import TEST_PART
from tests.support.tenants import new_tenant


class PropertyModeContract:
    """Tenant isolation and reserved property names."""

    async def test_one_tenant_cannot_touch_the_data_of_another(self, store, handle):
        """Reads, deletes and erase of one tenant leave the other's data intact."""
        other = await store.open(new_tenant(), [TEST_PART])
        nodes = [Node(i, "Entity", {"name": i}) for i in ("a", "b")]
        edge = Edge("Entity", "a", "Entity", "b", "MENTIONS")
        try:
            for h in (handle, other):
                await store.upsert_nodes(h, nodes)
                await store.upsert_edges(h, [edge])
            before = await store.stats(other)

            await store.delete(handle, NodeSelector("Entity", ["a", "b"]))
            await store.delete(
                handle, EdgeSelector("MENTIONS", "Entity", "Entity", [("a", "b")])
            )
            await store.erase_tenant(handle.tenant)

            after = await store.stats(other)
            assert (after.node_count, after.edge_count) == (
                before.node_count,
                before.edge_count,
            )
            assert after.edge_count == 1
            rows = await store.get_nodes(other, "Entity", ["a", "b"])
            assert sorted(r.id for r in rows) == ["a", "b"]
        finally:
            await store.erase_tenant(other.tenant)

    @pytest.mark.parametrize("name", ["_tenant", "_lock"])
    async def test_caller_property_with_a_reserved_name_is_refused(
        self, store, handle, name
    ):
        """Callers cannot write the properties the store keeps for itself."""
        with pytest.raises(ValidationError):
            await store.upsert_nodes(handle, [Node("a", "Entity", {name: "x"})])
        guard = Guard("Slot", "s", expect={}, set={name: "x"}, create_with={})
        with pytest.raises(ValidationError):
            await store.guarded_write(handle, guard)
        assert await store.get_nodes(handle, "Entity", ["a"]) == []

    async def test_reserved_properties_are_not_returned(self, store, handle):
        """Reads show only caller properties."""
        await store.upsert_nodes(handle, [Node("a", "Entity", {"name": "A"})])

        (row,) = await store.get_nodes(handle, "Entity", ["a"])

        assert dict(row.props) == {"id": "a", "name": "A"}
