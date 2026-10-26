"""Count graph records through the backend's tenant scope."""

from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.graphstore.types import GraphStats


async def graph_stats(runner: ScopedRunner) -> GraphStats:
    """Return tenant-scoped counts without checking the backend schema."""
    pattern = runner.scope.pattern()
    labels = await runner.run(
        QuerySpec(
            f"MATCH (n {pattern}) UNWIND labels(n) AS l "
            "RETURN l AS name, count(n) AS c",
            read_only=True,
            idempotent=True,
            tag="stats_labels",
        )
    )
    relationships = await runner.run(
        QuerySpec(
            f"MATCH (a {pattern})-[r]->() RETURN type(r) AS name, count(r) AS c",
            read_only=True,
            idempotent=True,
            tag="stats_rels",
        )
    )
    nodes = await runner.run(
        QuerySpec(
            f"MATCH (n {pattern}) RETURN count(n) AS c",
            read_only=True,
            idempotent=True,
            tag="stats_nodes",
        )
    )
    by_relationship = {row["name"]: row["c"] for row in relationships.rows}
    return GraphStats(
        int(nodes.scalar("c", 0)),
        sum(by_relationship.values()),
        {row["name"]: row["c"] for row in labels.rows},
        by_relationship,
        (),
    )
