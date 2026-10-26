"""Edge writes for Neo4j, in property mode."""

from collections.abc import Sequence
from functools import partial

from ontogr_mem.graphstore import cypher
from ontogr_mem.graphstore.cypher import PROPERTY_SCOPE
from ontogr_mem.graphstore.indices import SchemaPart
from ontogr_mem.graphstore.neo4j.runner import Neo4jRunner
from ontogr_mem.graphstore.types import Edge, WriteResult


async def upsert_edges(
    runner: Neo4jRunner, tenant: str, schema: SchemaPart, edges: Sequence[Edge]
) -> WriteResult:
    """Idempotently write `edges` between existing nodes.

    The result counts edges that matched both endpoints. Indexes of edges that
    were skipped are in `WriteResult.missing`.
    """
    send = partial(runner.run, tenant=tenant)
    return await cypher.upsert_edges(
        send, PROPERTY_SCOPE, schema, edges, runner.settings.batch_size
    )
