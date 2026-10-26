"""`FalkorDBGraphStore`: the graph store on FalkorDB, one graph per tenant."""

import time
from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.graphstore.errors import DatabaseUnavailableError, SchemaDriftError
from ontogr_mem.graphstore.falkordb import edges as edge_ops
from ontogr_mem.graphstore.falkordb import nodes as node_ops
from ontogr_mem.graphstore.falkordb.client import FalkorClientPool
from ontogr_mem.graphstore.falkordb.runner import GraphRunner
from ontogr_mem.graphstore.falkordb.settings import FalkorDBSettings
from ontogr_mem.graphstore.falkordb.tenancy import TenantGraphs
from ontogr_mem.graphstore.guarded import build_guarded_write
from ontogr_mem.graphstore.indices import SchemaPart
from ontogr_mem.graphstore.stats import graph_stats
from ontogr_mem.graphstore.types import (
    Edge,
    EdgeSelector,
    GraphHandle,
    GraphStats,
    Guard,
    Health,
    Node,
    NodeRow,
    NodeSelector,
    WriteResult,
)


class FalkorDBGraphStore:
    """Async graph store on FalkorDB with one graph per tenant."""

    def __init__(
        self,
        settings: FalkorDBSettings | None = None,
        pool: FalkorClientPool | None = None,
    ) -> None:
        """Create a store; the connection opens on first use."""
        self.settings = settings or FalkorDBSettings()
        self._pool = pool or FalkorClientPool(self.settings)
        self._runner = self._pool.runner
        self._tenants = TenantGraphs(self._pool)

    async def open(
        self, tenant: str, schema: Sequence[SchemaPart] | None = None
    ) -> GraphHandle:
        """Provision the tenant's graph if needed and return its handle.

        Args:
            tenant: Tenant id.
            schema: Schema parts the graph needs. Defaults to every part.

        Raises:
            ValidationError: If the tenant id is invalid.
        """
        return await self._tenants.open(tenant, schema)

    async def provision(self, handle: GraphHandle) -> None:
        """Create any missing index or constraint of the handle's schema."""
        await self._tenants.reprovision(handle)

    async def verify(self, handle: GraphHandle) -> None:
        """Raise `SchemaDriftError` if the live schema differs from the declared."""
        report = await self._tenants.schema.verify(
            handle.graph, self._tenants.schema_of(handle)
        )
        if not report.ok:
            msg = f"{handle.graph}: " + "; ".join(report.lines())
            raise SchemaDriftError(msg)

    async def upsert_nodes(
        self, handle: GraphHandle, nodes: Sequence[Node], *, replace: bool = False
    ) -> WriteResult:
        """Idempotently write nodes; `replace` drops properties not given."""
        return await node_ops.upsert_nodes(
            self._runner, handle.graph, nodes, replace=replace
        )

    async def upsert_edges(
        self, handle: GraphHandle, edges: Sequence[Edge]
    ) -> WriteResult:
        """Idempotently write edges; edges with a missing endpoint are reported."""
        return await edge_ops.upsert_edges(
            self._runner, handle.graph, self._tenants.schema_of(handle), edges
        )

    async def get_nodes(
        self, handle: GraphHandle, label: str, ids: Sequence[str]
    ) -> list[NodeRow]:
        """Read nodes of `label` by id."""
        return await node_ops.get_nodes(self._runner, handle.graph, label, ids)

    async def delete(
        self,
        handle: GraphHandle,
        selector: NodeSelector | EdgeSelector,
        *,
        detach: bool = True,
    ) -> int:
        """Delete the selection; missing items are ignored. Returns the count."""
        return await node_ops.delete(
            self._runner, handle.graph, selector, detach=detach
        )

    async def guarded_write(
        self,
        handle: GraphHandle,
        guard: Guard,
        nodes: Sequence[Node] = (),
        edges: Sequence[Edge] = (),
    ) -> bool:
        """Write `nodes` and `edges` and update the guard node, all or nothing.

        Edges whose endpoints do not exist are skipped.

        Returns:
            True if the guard held and everything was written, False if the
            guard did not hold and nothing was written.
        """
        runner = self.runner(handle)
        spec = build_guarded_write(
            guard, nodes, edges, self._tenants.schema_of(handle), runner.scope
        )
        rows = await runner.run(spec)
        return bool(rows.scalar("applied", 0))

    def runner(self, handle: GraphHandle) -> GraphRunner:
        """Return a runner bound to the handle's graph."""
        return GraphRunner(self._runner, handle.graph)

    async def stats(self, handle: GraphHandle) -> GraphStats:
        """Return counts per label and relationship type, plus schema drift."""
        counts = await graph_stats(self.runner(handle))
        drift = await self._tenants.schema.verify(
            handle.graph, self._tenants.schema_of(handle)
        )
        return replace(counts, drift=drift.lines())

    async def health(self) -> Health:
        """PING the server through the runner; never raises."""
        started = time.perf_counter()
        try:
            ok = await self._pool.ping()
            detail = ""
        except DatabaseUnavailableError as exc:
            ok, detail = False, str(exc)
        return Health(
            ok,
            (time.perf_counter() - started) * 1000,
            self._runner.breaker.state.value,
            detail,
        )

    async def erase_tenant(self, tenant: str) -> None:
        """Delete the tenant's graph; erasing a missing graph does nothing."""
        await self._tenants.erase(tenant)

    async def close(self) -> None:
        """Release the connection pool."""
        await self._pool.close()
