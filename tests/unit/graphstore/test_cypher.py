"""Tests for backend Cypher dialect and tenant scoping."""

import pytest

from ontogr_mem.graphstore.cypher import (
    GRAPH_SCOPE,
    PROPERTY_SCOPE,
    GraphModeDialect,
    PropertyModeDialect,
    TenantScope,
    edge_merge_clause,
    node_merge_lines,
)
from ontogr_mem.graphstore.errors import ValidationError
from ontogr_mem.graphstore.indices import SchemaPart


class TestTenantScope:
    """Behavior of tenant-bound Cypher patterns."""

    def test_property_mode_scopes_nodes_and_edges_and_stamps_properties(self):
        """Shared-database patterns filter and write the tenant key."""
        scope = TenantScope(PropertyModeDialect())

        assert scope.pattern("id: $id") == "{_tenant: $tenant, id: $id}"
        assert scope.edge("id: $id") == "{_tenant: $tenant, id: $id}"
        assert scope.stamp_assignment("r") == "SET r._tenant = $tenant"
        assert scope.stamp({"id": "a"}, "tenant-a") == {
            "id": "a",
            "_tenant": "tenant-a",
        }

    def test_graph_mode_uses_unkeyed_patterns_and_keeps_properties_unchanged(self):
        """One-graph-per-tenant patterns do not store a redundant tenant key."""
        scope = TenantScope(GraphModeDialect())

        assert scope.pattern("id: $id") == "{id: $id}"
        assert scope.edge("id: $id") == "{id: $id}"
        assert scope.stamp_assignment("r") == ""
        assert scope.stamp({"id": "a"}, "tenant-a") == {"id": "a"}


class TestNodePattern:
    """The node pattern each mode writes."""

    def test_property_mode_names_the_tenant(self):
        """A node in property mode carries the tenant."""
        assert PROPERTY_SCOPE.node_pattern("n", "Entity", "item.id") == (
            "(n:`Entity` {_tenant: $tenant, id: item.id})"
        )

    def test_graph_mode_names_only_the_node(self):
        """A node in graph mode carries nothing but its id."""
        assert GRAPH_SCOPE.node_pattern("n", "Entity", "item.id") == (
            "(n:`Entity` {id: item.id})"
        )

    def test_an_unsafe_label_never_reaches_a_statement(self):
        """A label that could break out of the statement is refused."""
        with pytest.raises(ValidationError):
            GRAPH_SCOPE.node_pattern("n", "Entity` REMOVE n //", "item.id")

    def test_property_mode_hides_the_properties_the_store_keeps(self):
        """A caller does not see the tenant or lock property."""
        props = {"id": "a", "_tenant": "t", "_lock": "l"}

        assert PROPERTY_SCOPE.read_props(props) == {"id": "a"}

    def test_graph_mode_shows_every_property(self):
        """Graph mode keeps no property of its own."""
        props = {"id": "a", "_tenant": "t"}

        assert GRAPH_SCOPE.read_props(props) == props


class TestSharedStatements:
    """The statements both backends write."""

    def test_the_edge_merge_clause_is_the_same_for_both_modes(self):
        """The relationship clause does not depend on the tenant scoping."""
        schema = SchemaPart("s", edge_keys={"MENTIONS": ("k",)})
        key = ("Entity", "Entity", "MENTIONS")

        assert edge_merge_clause(key, schema) == (
            "MERGE (a)-[r:`MENTIONS` {`k`: item.key.`k`}]->(b)"
        )

    def test_a_replace_in_property_mode_keeps_the_tenant(self):
        """The node keeps its tenant after a replace."""
        lines = node_merge_lines(("Entity", ()), PROPERTY_SCOPE, replace=True)

        assert lines == [
            "MERGE (n:`Entity` {_tenant: $tenant, id: item.id})",
            "SET n = item.props",
            "SET n._tenant = $tenant",
        ]

    def test_a_replace_in_graph_mode_keeps_only_the_properties(self):
        """The node holds exactly the properties the write names."""
        lines = node_merge_lines(("Entity", ()), GRAPH_SCOPE, replace=True)

        assert lines == [
            "MERGE (n:`Entity` {id: item.id})",
            "SET n = item.props",
        ]

    def test_an_extra_label_is_quoted(self):
        """An extra label is written the same way as the primary label."""
        lines = node_merge_lines(("Entity", ("Org",)), GRAPH_SCOPE, replace=False)

        assert lines[-1] == "SET n:`Org`"
