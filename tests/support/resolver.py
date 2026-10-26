"""Builders and graph hooks for resolver store tests."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ontogr_mem.graphstore import QuerySpec, ScopedRunner
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.resolution.resolver_store import ResolverStore


@dataclass
class ResolverHarness:
    """A resolver store with the hooks the contract needs."""

    store: ResolverStore
    seed_fact_edge: Callable[..., Awaitable[None]]
    fact_edge_count: Callable[..., Awaitable[int]]


def graph_resolver_harness(
    runner: ScopedRunner, *, tenant_keyed: bool
) -> ResolverHarness:
    """Build a harness for a graph backend from its tenant-bound runner."""
    scope = "{_tenant: $tenant, " if tenant_keyed else "{"
    edge_stamp = ", r._tenant = $tenant" if tenant_keyed else ""

    async def seed_fact_edge(
        store: ResolverStore, entity_id: str, *, live: bool
    ) -> None:
        await runner.run(
            QuerySpec(
                f"MATCH (e:Entity {scope}id: $id}}) "
                "MERGE (e)-[r:FACT]->(e) "
                f"SET r.status = $status{edge_stamp}",
                {"id": entity_id, "status": "active" if live else "withdrawn"},
                idempotent=True,
                tag="resolver_seed_fact",
            )
        )

    async def fact_edge_count(
        store: ResolverStore, entity_id: str, *, live: bool
    ) -> int:
        rows = await runner.run(
            QuerySpec(
                f"MATCH (e:Entity {scope}id: $id}})-[r:FACT]->() "
                "WHERE r.status = $status RETURN count(r) AS n",
                {"id": entity_id, "status": "active" if live else "withdrawn"},
                read_only=True,
                idempotent=True,
                tag="resolver_count_fact",
            )
        )
        return int(rows.scalar("n", 0))

    return ResolverHarness(
        GraphResolverStore(runner, tenant_keyed=tenant_keyed),
        seed_fact_edge,
        fact_edge_count,
    )
