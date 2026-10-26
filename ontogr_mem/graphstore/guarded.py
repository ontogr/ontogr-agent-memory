"""Shared guarded-write statement builder for graph backends."""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from ontogr_mem.graphstore.cypher import (
    TenantScope,
    created_value,
    edge_merge_clause,
    node_merge_lines,
)
from ontogr_mem.graphstore.errors import ValidationError
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.indices import SchemaPart
from ontogr_mem.graphstore.prepare import group_edges, group_nodes
from ontogr_mem.graphstore.reserved import LOCK, reject_reserved
from ontogr_mem.graphstore.types import Edge, Guard, Node
from ontogr_mem.graphstore.values import clean_properties


def _validate_guard(guard: Guard) -> None:
    """Reject changes to identity and backend-owned properties."""
    if "id" in guard.set or (
        guard.create_with is not None and "id" in guard.create_with
    ):
        raise ValidationError("a guard must not change the id of its node")
    for owner, props in (
        ("guard expect", guard.expect),
        ("guard set", guard.set),
        ("guard create_with", guard.create_with or {}),
    ):
        reject_reserved(props, owner)


def build_guarded_write(
    guard: Guard,
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    schema: SchemaPart,
    scope: TenantScope,
) -> QuerySpec:
    """Build one statement that checks the guard before writing anything."""
    _validate_guard(guard)
    params: dict[str, Any] = {"gid": guard.id}
    if scope.dialect.keyed:
        params["writer"] = uuid4().hex
    label = quote(guard.label, "label")
    guard_pattern = scope.pattern("id: $gid")
    conditions: list[str] = []
    created: list[str] = []
    for index, (key, value) in enumerate(guard.expect.items()):
        if isinstance(value, Mapping):
            msg = (
                f"guard expectation {key!r} must be a scalar; "
                "a mapping is stored under a different property name"
            )
            raise ValidationError(msg)
        prop = f"g.{quote(key, 'property key')}"
        target = created_value(key, "gcreate")
        if value is None:
            conditions.append(f"{prop} IS NULL")
            created.append(f"{target} IS NULL")
            continue
        params[f"x{index}"] = clean_properties({key: value})[key]
        conditions.append(f"{prop} = $x{index}")
        created.append(f"{target} = $x{index}")

    lines: list[str] = []
    if guard.create_with is not None:
        params["gcreate"] = clean_properties(guard.create_with)
        holds = " AND ".join(created) or "true"
        lines.append(
            f"FOREACH (_ IN CASE WHEN {holds} THEN [1] ELSE [] END | "
            f"MERGE (gc:{label} {guard_pattern}) ON CREATE SET gc += $gcreate)"
        )
        lines.append("WITH 1 AS guard_row")

    lines.append(f"MATCH (g:{label} {guard_pattern})")
    check = " AND ".join(conditions) or "true"
    if scope.dialect.keyed:
        lines += [
            f"WITH g, g.{LOCK} AS prev",
            f"SET g.{LOCK} = $writer",
            f"WITH g, prev, ({check}) AS ok",
            f"SET g.{LOCK} = prev",
            "WITH g WHERE ok",
        ]
    elif conditions:
        lines.append(f"WITH g WHERE {check}")

    if guard.set:
        params["gset"] = clean_properties(guard.set)
        lines.append("SET g += $gset")
    for index, (key, items) in enumerate(group_nodes(nodes).items()):
        params[f"n{index}"] = items
        lines += ["WITH DISTINCT g", f"UNWIND $n{index} AS item"]
        lines.extend(node_merge_lines(key, scope, replace=False))
    for index, (key, items) in enumerate(group_edges(edges, schema).items()):
        params[f"e{index}"] = items
        src, dst, _ = key
        lines += [
            "WITH DISTINCT g",
            f"UNWIND $e{index} AS item",
            f"OPTIONAL MATCH (a:{quote(src, 'label')} {scope.pattern('id: item.src')})",
            f"OPTIONAL MATCH (b:{quote(dst, 'label')} {scope.pattern('id: item.dst')})",
            "FOREACH (_ IN CASE WHEN a IS NOT NULL AND b IS NOT NULL "
            f"THEN [1] ELSE [] END | {edge_merge_clause(key, schema)} "
            "SET r += item.props"
            f"{scope.stamp_clause('r')})",
        ]
    lines += ["WITH DISTINCT g", "RETURN count(g) AS applied"]
    return QuerySpec("\n".join(lines), params, tag="guarded_write")
