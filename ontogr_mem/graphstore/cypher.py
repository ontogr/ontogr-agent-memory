"""Cypher dialects, tenant-scoped patterns and writes shared by graph backends.

Neo4j keeps every tenant of every component in one shared database and marks
each row with a tenant property. FalkorDB keeps one graph per tenant, so the
tenant is part of the graph and needs no property. The scope holds that
difference. The statements below are the same for both backends.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ontogr_mem.graphstore.batching import (
    BatchOutcome,
    BatchReply,
    Item,
    to_write_result,
    write_batches,
)
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.ident import ident, quote
from ontogr_mem.graphstore.indices import SchemaPart
from ontogr_mem.graphstore.prepare import EdgeKey, NodeKey, group_edges, group_nodes
from ontogr_mem.graphstore.reserved import TENANT, strip_reserved
from ontogr_mem.graphstore.types import (
    Edge,
    EdgeSelector,
    Node,
    NodeRow,
    WriteResult,
)


Send = Callable[[QuerySpec], Awaitable[Rows]]
"""Runs one statement and returns its rows."""


class Dialect(Protocol):
    """Backend differences used when building common Cypher."""

    @property
    def keyed(self) -> bool:
        """Whether every pattern needs a tenant property."""
        ...

    def stamp(self, properties: Mapping[str, Any], tenant: str) -> dict[str, Any]:
        """Add backend-required properties to a node or relationship row."""
        ...

    def stamp_clause(self, variable: str) -> str:
        """Return a property assignment for a matched node or relationship."""
        ...

    def stamp_assignment(self, variable: str) -> str:
        """Return a standalone property assignment, if the mode needs one."""
        ...


@dataclass(frozen=True)
class PropertyModeDialect:
    """Dialect for tenants that share a database and use `_tenant`."""

    keyed: bool = True

    def stamp(self, properties: Mapping[str, Any], tenant: str) -> dict[str, Any]:
        """Copy properties and add their tenant key."""
        return {**properties, TENANT: tenant}

    def stamp_clause(self, variable: str) -> str:
        """Return a clause that writes the tenant key."""
        return f", {variable}.{TENANT} = $tenant"

    def stamp_assignment(self, variable: str) -> str:
        """Return a standalone clause that writes the tenant key."""
        return f"SET {variable}.{TENANT} = $tenant"


@dataclass(frozen=True)
class GraphModeDialect:
    """Dialect for one graph per tenant."""

    keyed: bool = False

    def stamp(self, properties: Mapping[str, Any], tenant: str) -> dict[str, Any]:
        """Copy properties without a redundant tenant key."""
        return dict(properties)

    def stamp_clause(self, variable: str) -> str:
        """Return no clause because graph identity supplies the tenant scope."""
        return ""

    def stamp_assignment(self, variable: str) -> str:
        """Return no clause because graph identity supplies the tenant scope."""
        return ""


@dataclass(frozen=True)
class TenantScope:
    """Build node and edge patterns for one backend's tenant model."""

    dialect: Dialect = PropertyModeDialect()

    @classmethod
    def from_mode(cls, tenant_keyed: bool) -> "TenantScope":
        """Return the property-mode or graph-mode scope."""
        return cls(PropertyModeDialect() if tenant_keyed else GraphModeDialect())

    def pattern(self, properties: str = "") -> str:
        """Return a node property map that always includes tenant scope."""
        return self._property_map(properties)

    def edge(self, properties: str = "") -> str:
        """Return an edge property map that always includes tenant scope."""
        return self._property_map(properties)

    def stamp(self, properties: Mapping[str, Any], tenant: str) -> dict[str, Any]:
        """Return a copy of properties stamped for this backend."""
        return self.dialect.stamp(properties, tenant)

    def stamp_clause(self, variable: str) -> str:
        """Return the tenant assignment suffix for a Cypher `SET` clause."""
        return self.dialect.stamp_clause(variable)

    def stamp_assignment(self, variable: str) -> str:
        """Return a standalone tenant assignment clause when required."""
        return self.dialect.stamp_assignment(variable)

    def node_pattern(self, alias: str, label: str, id_expression: str) -> str:
        """Return the pattern that binds `alias` to the node of `label`.

        Args:
            alias: The variable the node is bound to.
            label: Label of the node.
            id_expression: The Cypher expression that holds the node id.

        Returns:
            The pattern, with the tenant scope when the mode needs one.
        """
        return (
            f"({alias}:{quote(label, 'label')} {self.pattern(f'id: {id_expression}')})"
        )

    def edge_pattern(self, rel: str, body: str = "") -> str:
        """Return the pattern for a relationship of type `rel`.

        Args:
            rel: Type of the relationship.
            body: The properties of the relationship, without braces.

        Returns:
            The pattern, bound to the variable `r`.
        """
        return f"[r:{quote(rel, 'relationship type')}{body}]"

    def read_props(self, props: Mapping[str, Any]) -> dict[str, Any]:
        """Return the properties a caller may see.

        Args:
            props: The properties the server sent.

        Returns:
            The properties without the ones the store keeps for itself.
        """
        if self.dialect.keyed:
            return strip_reserved(props)
        return dict(props)

    def _property_map(self, properties: str) -> str:
        parts = [f"{TENANT}: $tenant"] if self.dialect.keyed else []
        if properties:
            parts.append(properties)
        return "{" + ", ".join(parts) + "}" if parts else ""


PROPERTY_SCOPE = TenantScope(PropertyModeDialect())
GRAPH_SCOPE = TenantScope(GraphModeDialect())


def node_merge_lines(key: NodeKey, scope: TenantScope, *, replace: bool) -> list[str]:
    """Return the common clauses that merge one `item` node."""
    label, extras = key
    lines = [
        f"MERGE {scope.node_pattern('n', label, 'item.id')}",
        "SET n = item.props" if replace else "SET n += item.props",
    ]
    if replace:
        restore = scope.stamp_assignment("n")
        if restore:
            lines.append(restore)
    if extras:
        lines.append("SET n:" + ":".join(quote(name, "label") for name in extras))
    return lines


def edge_merge_clause(key: EdgeKey, schema: SchemaPart) -> str:
    """Return the common `MERGE` clause for one edge row."""
    rel = key[2]
    props = ", ".join(
        f"{quote(name, 'property key')}: item.key.{quote(name, 'property key')}"
        for name in schema.edge_keys.get(rel, ())
    )
    props = f" {{{props}}}" if props else ""
    return f"MERGE (a)-[r:{quote(rel, 'relationship type')}{props}]->(b)"


def edge_match_lines(key: EdgeKey, scope: TenantScope) -> list[str]:
    """Return the common endpoint lookups for one edge row."""
    src, dst, _ = key
    return [
        f"MATCH {scope.node_pattern('a', src, 'item.src')}",
        f"MATCH {scope.node_pattern('b', dst, 'item.dst')}",
    ]


def created_value(key: str, parameter: str) -> str:
    """Return the Cypher expression for a property on a newly made guard."""
    if key == "id":
        return "$gid"
    return f"${parameter}['{ident(key, 'property key')}']"


async def upsert_nodes(
    send: Send,
    scope: TenantScope,
    nodes: Sequence[Node],
    *,
    replace: bool,
    batch_size: int,
) -> WriteResult:
    """Write `nodes` and report what the server did.

    Args:
        send: Runs one statement and returns its rows.
        scope: The tenant scope of the backend that runs the statement.
        nodes: The nodes to write.
        replace: True to drop the properties the write does not name.
        batch_size: Number of nodes one statement carries.

    Returns:
        The outcome of the write.
    """
    outcome = BatchOutcome()
    for key, items in group_nodes(nodes).items():
        statement = "\n".join(
            [
                "UNWIND $batch AS item",
                *node_merge_lines(key, scope, replace=replace),
                "RETURN collect(item.i) AS ok",
            ]
        )

        async def send_chunk(
            chunk: list[Item], statement: str = statement
        ) -> BatchReply:
            rows = await send(
                QuerySpec(
                    statement, {"batch": chunk}, idempotent=True, tag="upsert_nodes"
                )
            )
            return BatchReply(rows.scalar("ok", []), rows.stats["nodes_created"])

        await write_batches(outcome, items, batch_size, send_chunk)
    return to_write_result(len(nodes), outcome)


async def upsert_edges(
    send: Send,
    scope: TenantScope,
    schema: SchemaPart,
    edges: Sequence[Edge],
    batch_size: int,
) -> WriteResult:
    """Write `edges` between existing nodes and report what the server did.

    Args:
        send: Runs one statement and returns its rows.
        scope: The tenant scope of the backend that runs the statement.
        schema: The schema part that declares the edge keys of the types.
        edges: The edges to write.
        batch_size: Number of edges one statement carries.

    Returns:
        The outcome of the write. Edges whose endpoint is missing are in
        `WriteResult.missing`.
    """
    outcome = BatchOutcome()
    for key, items in group_edges(edges, schema).items():
        statement = "\n".join(
            [
                "UNWIND $batch AS item",
                # Bind both endpoints first: MERGE on a path with an unbound
                # endpoint creates a ghost node.
                *edge_match_lines(key, scope),
                edge_merge_clause(key, schema),
                "SET r += item.props",
                "RETURN collect(item.i) AS ok",
            ]
        )

        async def send_chunk(
            chunk: list[Item], statement: str = statement
        ) -> BatchReply:
            rows = await send(
                QuerySpec(
                    statement, {"batch": chunk}, idempotent=True, tag="upsert_edges"
                )
            )
            return BatchReply(
                rows.scalar("ok", []), rows.stats["relationships_created"]
            )

        await write_batches(outcome, items, batch_size, send_chunk)
    return to_write_result(len(edges), outcome)


async def get_nodes(
    send: Send, scope: TenantScope, label: str, ids: Sequence[str]
) -> list[NodeRow]:
    """Read the nodes of `label` by id.

    The labels come back sorted, so both backends report them the same way.

    Args:
        send: Runs one statement and returns its rows.
        scope: The tenant scope of the backend that runs the statement.
        label: Label of the nodes to read.
        ids: Ids of the nodes to read.

    Returns:
        One row per node that exists.
    """
    if not ids:
        return []
    rows = await send(
        QuerySpec(
            f"UNWIND $ids AS i MATCH {scope.node_pattern('n', label, 'i')} "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props",
            {"ids": list(dict.fromkeys(ids))},
            read_only=True,
            idempotent=True,
            tag="get_nodes",
        )
    )
    return [
        NodeRow(r["id"], tuple(sorted(r["labels"])), scope.read_props(r["props"]))
        for r in rows.rows
    ]


async def delete_edges(send: Send, scope: TenantScope, selector: EdgeSelector) -> int:
    """Delete the selected edges and return how many were removed.

    Args:
        send: Runs one statement and returns its rows.
        scope: The tenant scope of the backend that runs the statement.
        selector: The edges to delete.

    Returns:
        The number of relationships the server removed.
    """
    if not selector.pairs:
        return 0
    rows = await send(
        QuerySpec(
            f"UNWIND $pairs AS p "
            f"MATCH {scope.node_pattern('a', selector.src_label, 'p[0]')}-"
            f"{scope.edge_pattern(selector.rel_type)}->"
            f"{scope.node_pattern('b', selector.dst_label, 'p[1]')} DELETE r",
            {"pairs": [list(p) for p in selector.pairs]},
            tag="delete_edges",
        )
    )
    return rows.stats["relationships_deleted"]
