"""Validation and batch-row building shared by every backend."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ontogr_mem.graphstore.batching import Item
from ontogr_mem.graphstore.errors import ValidationError
from ontogr_mem.graphstore.ident import ident
from ontogr_mem.graphstore.indices import SchemaPart
from ontogr_mem.graphstore.reserved import reject_reserved
from ontogr_mem.graphstore.types import Edge, Node
from ontogr_mem.graphstore.values import clean_properties


NodeKey = tuple[str, tuple[str, ...]]
EdgeKey = tuple[str, str, str]


def prepare_node(node: Node, index: int) -> tuple[NodeKey, dict[str, Any]]:
    """Validate `node` and return its group key and the batch row for `index`.

    Raises:
        ValidationError: If the id is empty or conflicts with a property, or a
            name or value is unsafe.
    """
    if not isinstance(node.id, str) or not node.id:
        msg = f"node of label {node.label!r} has no id"
        raise ValidationError(msg)
    label = ident(node.label, "label")
    extras = tuple(sorted(ident(x, "label") for x in node.extra_labels if x != label))
    props = clean_properties(node.props)
    reject_reserved(props, f"node {node.id!r}")
    if props.setdefault("id", node.id) != node.id:
        msg = f"node {node.id!r}: property 'id' differs from the node id"
        raise ValidationError(msg)
    return (label, extras), {"i": index, "id": node.id, "props": props}


def prepare_edge(
    edge: Edge, schema: SchemaPart, index: int
) -> tuple[EdgeKey, dict[str, Any]]:
    """Validate `edge` and return its group key and the batch row for `index`.

    Edge key properties missing from `props` default to an empty string,
    because a graph database cannot merge on a null value.
    """
    rel = ident(edge.rel_type, "relationship type")
    key = (ident(edge.src_label, "label"), ident(edge.dst_label, "label"), rel)
    if not edge.src_id or not edge.dst_id:
        msg = f"edge {rel!r} has an empty endpoint id"
        raise ValidationError(msg)
    props = clean_properties(edge.props)
    identity = {k: props.get(k, "") for k in schema.edge_keys.get(rel, ())}
    item = {
        "i": index,
        "src": edge.src_id,
        "dst": edge.dst_id,
        "props": props,
        "key": identity,
    }
    return key, item


def group_nodes(nodes: Sequence[Node]) -> dict[NodeKey, list[Item]]:
    """Validate `nodes` and group them by label set, each group sorted by id."""
    groups: dict[NodeKey, list[Item]] = defaultdict(list)
    for i, node in enumerate(nodes):
        key, item = prepare_node(node, i)
        groups[key].append(item)
    for items in groups.values():
        items.sort(key=lambda it: it["id"])
    return groups


def group_edges(edges: Sequence[Edge], schema: SchemaPart) -> dict[EdgeKey, list[Item]]:
    """Validate `edges` and group them by endpoint labels and type."""
    groups: dict[EdgeKey, list[Item]] = defaultdict(list)
    for i, edge in enumerate(edges):
        key, item = prepare_edge(edge, schema, i)
        groups[key].append(item)
    for items in groups.values():
        items.sort(key=lambda it: (it["src"], it["dst"]))
    return groups
