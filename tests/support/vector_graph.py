"""Graph objects that carry vectors, for the integration backends."""

from collections.abc import Sequence

from ontogr_mem.graphstore import (
    Edge,
    GraphHandle,
    GraphStore,
    Node,
)
from ontogr_mem.vectorstore import Validity, VectorKind


async def seed_graph(
    store: GraphStore,
    handle: GraphHandle,
    kind: VectorKind,
    ids: Sequence[str],
    *,
    status: str | None = "active",
    entity_type: int | None = None,
    validity: Validity | None = None,
    subject_id: str | None = None,
    object_id: str | None = None,
    relation_type_id: int | None = None,
) -> None:
    """Create the nodes or edges that `ids` name.

    Edges get a `FACT` edge between two `Entity` nodes, with the validity
    properties the graph-native search reads. Endpoints default to `s-<id>` and
    `o-<id>`.
    """
    if kind is VectorKind.EPISODE:
        await store.upsert_nodes(handle, [Node(i, "Episode") for i in ids])
        return
    if kind is VectorKind.NODE:
        props: dict[str, object] = {}
        if status is not None:
            props["status"] = status
        if entity_type is not None:
            props["entity_type"] = entity_type
        await store.upsert_nodes(handle, [Node(i, "Entity", props) for i in ids])
        return
    validity = validity or Validity("active", 0, 0)
    endpoints: dict[str, Node] = {}
    edges: list[Edge] = []
    for fact_id in ids:
        subject = subject_id or f"s-{fact_id}"
        target = object_id or f"o-{fact_id}"
        endpoints[subject] = Node(subject, "Entity")
        endpoints[target] = Node(target, "Entity")
        props: dict[str, object] = {
            "fact_id": fact_id,
            "status": validity.status,
            "expired_at": validity.expired_at_us,
            "valid_from": validity.valid_from_us,
            "valid_to": validity.valid_to_us,
            "recorded_at": validity.recorded_at_us,
            "subject_id": subject,
            "object_id": target,
        }
        if relation_type_id is not None:
            props["relation_type_id"] = relation_type_id
        edges.append(Edge("Entity", subject, "Entity", target, "FACT", props=props))
    await store.upsert_nodes(handle, list(endpoints.values()))
    await store.upsert_edges(handle, edges)
