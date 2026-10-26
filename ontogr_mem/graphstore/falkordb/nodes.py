"""Node writes, reads and deletes for FalkorDB, in graph mode."""

from collections.abc import Sequence
from functools import partial

from ontogr_mem.graphstore import cypher
from ontogr_mem.graphstore.cypher import GRAPH_SCOPE, Send
from ontogr_mem.graphstore.errors import ConstraintViolationError
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.falkordb.runner import QueryRunner
from ontogr_mem.graphstore.types import (
    EdgeSelector,
    Node,
    NodeRow,
    NodeSelector,
    WriteResult,
)


async def upsert_nodes(
    runner: QueryRunner, graph: str, nodes: Sequence[Node], *, replace: bool = False
) -> WriteResult:
    """Idempotently write `nodes`.

    Items are grouped by label set so the labels are known when the statement
    is parsed, then sent in chunks of the batch size. A rejected chunk is sent
    again item by item.
    """
    send = partial(runner.run, graph)
    return await cypher.upsert_nodes(
        send,
        GRAPH_SCOPE,
        nodes,
        replace=replace,
        batch_size=runner.settings.batch_size,
    )


async def get_nodes(
    runner: QueryRunner, graph: str, label: str, ids: Sequence[str]
) -> list[NodeRow]:
    """Read nodes of `label` by id."""
    send = partial(runner.run, graph)
    return await cypher.get_nodes(send, GRAPH_SCOPE, label, ids)


async def delete(
    runner: QueryRunner,
    graph: str,
    selector: NodeSelector | EdgeSelector,
    *,
    detach: bool = True,
) -> int:
    """Delete the selection and return how many items were removed.

    Missing items are ignored. The count comes from the server's write
    statistics.

    Raises:
        ConstraintViolationError: If `detach` is False and a selected node has
            edges. No selected node is removed.
    """
    send = partial(runner.run, graph)
    if isinstance(selector, NodeSelector):
        return await _delete_nodes(send, selector, detach=detach)
    return await cypher.delete_edges(send, GRAPH_SCOPE, selector)


async def _delete_nodes(send: Send, selector: NodeSelector, *, detach: bool) -> int:
    """Delete the selected nodes and return how many were removed."""
    if not selector.ids:
        return 0
    pattern = GRAPH_SCOPE.node_pattern("n", selector.label, "i")
    ids = list(dict.fromkeys(selector.ids))
    if not detach:
        rows = await send(
            QuerySpec(
                f"UNWIND $ids AS i OPTIONAL MATCH {pattern} "
                "OPTIONAL MATCH (n)-[r]-() "
                "WITH collect(DISTINCT n) AS selected, "
                "collect(DISTINCT CASE WHEN r IS NOT NULL THEN n END) AS linked "
                "FOREACH (_ IN CASE WHEN size(linked) = 0 "
                "THEN selected ELSE [] END | DELETE _) "
                "RETURN size(linked) AS linked",
                {"ids": ids},
                tag="delete_nodes_without_detach",
            )
        )
        linked_count = int(rows.scalar("linked", 0))
        if linked_count:
            msg = (
                f"refusing detach=False delete: {linked_count} nodes have relationships"
            )
            raise ConstraintViolationError(msg)
        return rows.stats["nodes_deleted"]
    rows = await send(
        QuerySpec(
            f"UNWIND $ids AS i MATCH {pattern} DETACH DELETE n",
            {"ids": ids},
            tag="delete_nodes",
        )
    )
    return rows.stats["nodes_deleted"]
