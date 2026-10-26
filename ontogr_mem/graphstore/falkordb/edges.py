"""Edge writes for FalkorDB, in graph mode."""

from collections.abc import Sequence
from functools import partial

from ontogr_mem.graphstore import cypher
from ontogr_mem.graphstore.cypher import GRAPH_SCOPE
from ontogr_mem.graphstore.falkordb.runner import QueryRunner
from ontogr_mem.graphstore.indices import SchemaPart
from ontogr_mem.graphstore.types import Edge, WriteResult


async def upsert_edges(
    runner: QueryRunner, graph: str, schema: SchemaPart, edges: Sequence[Edge]
) -> WriteResult:
    """Idempotently write `edges` between existing nodes.

    The result counts edges that matched both endpoints. Indexes of edges that
    were skipped are in `WriteResult.missing`.
    """
    send = partial(runner.run, graph)
    return await cypher.upsert_edges(
        send, GRAPH_SCOPE, schema, edges, runner.settings.batch_size
    )
