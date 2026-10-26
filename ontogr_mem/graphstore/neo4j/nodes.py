"""Node writes, reads and deletes for Neo4j, in property mode."""

from collections.abc import Sequence
from functools import partial

from ontogr_mem.graphstore import cypher
from ontogr_mem.graphstore.cypher import PROPERTY_SCOPE
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.neo4j.runner import Neo4jRunner
from ontogr_mem.graphstore.types import (
    EdgeSelector,
    Node,
    NodeRow,
    NodeSelector,
    WriteResult,
)


async def upsert_nodes(
    runner: Neo4jRunner,
    tenant: str,
    nodes: Sequence[Node],
    *,
    replace: bool = False,
) -> WriteResult:
    """Idempotently write `nodes`, sent in chunks of the batch size.

    A rejected chunk is sent again item by item.

    Raises:
        ValidationError: If a node uses a reserved property name.
    """
    send = partial(runner.run, tenant=tenant)
    return await cypher.upsert_nodes(
        send,
        PROPERTY_SCOPE,
        nodes,
        replace=replace,
        batch_size=runner.settings.batch_size,
    )


async def get_nodes(
    runner: Neo4jRunner, tenant: str, label: str, ids: Sequence[str]
) -> list[NodeRow]:
    """Read the tenant's nodes of `label` by id."""
    send = partial(runner.run, tenant=tenant)
    return await cypher.get_nodes(send, PROPERTY_SCOPE, label, ids)


async def delete(
    runner: Neo4jRunner,
    tenant: str,
    selector: NodeSelector | EdgeSelector,
    *,
    detach: bool = True,
) -> int:
    """Delete the selection and return how many items were removed.

    Missing items are ignored. The server refuses to delete a node that still
    has relationships, and it removes nothing when it refuses.

    Raises:
        StoreQueryError: If `detach` is False and a selected node has edges.
            Nothing is removed.
    """
    send = partial(runner.run, tenant=tenant)
    if isinstance(selector, NodeSelector):
        if not selector.ids:
            return 0
        verb = "DETACH DELETE" if detach else "DELETE"
        rows = await send(
            QuerySpec(
                f"UNWIND $ids AS i "
                f"MATCH {PROPERTY_SCOPE.node_pattern('n', selector.label, 'i')} "
                f"{verb} n",
                {"ids": list(dict.fromkeys(selector.ids))},
                tag="delete_nodes",
            )
        )
        return rows.stats["nodes_deleted"]
    return await cypher.delete_edges(send, PROPERTY_SCOPE, selector)
