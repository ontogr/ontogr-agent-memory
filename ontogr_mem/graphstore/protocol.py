"""Ports of the graph store."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.indices import SchemaPart
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


@runtime_checkable
class ScopedRunner(Protocol):
    """Runs a component adapter's own Cypher against one tenant's graph."""

    scope: TenantScope

    async def run(self, spec: QuerySpec) -> Rows:
        """Execute `spec` and return its rows."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Id-keyed graph storage, one graph per tenant."""

    async def open(
        self, tenant: str, schema: Sequence[SchemaPart] | None = None
    ) -> GraphHandle:
        """Provision (once) and return the tenant's graph handle."""
        ...

    async def provision(self, handle: GraphHandle) -> None:
        """Create any missing index or constraint of the handle's schema."""
        ...

    async def verify(self, handle: GraphHandle) -> None:
        """Raise `SchemaDriftError` if the live schema differs from the declared."""
        ...

    async def upsert_nodes(
        self, handle: GraphHandle, nodes: Sequence[Node], *, replace: bool = False
    ) -> WriteResult:
        """Idempotently write nodes."""
        ...

    async def upsert_edges(
        self, handle: GraphHandle, edges: Sequence[Edge]
    ) -> WriteResult:
        """Idempotently write edges; edges with a missing endpoint are reported."""
        ...

    async def get_nodes(
        self, handle: GraphHandle, label: str, ids: Sequence[str]
    ) -> list[NodeRow]:
        """Read nodes of `label` by id."""
        ...

    async def delete(
        self,
        handle: GraphHandle,
        selector: NodeSelector | EdgeSelector,
        *,
        detach: bool = True,
    ) -> int:
        """Delete the selection; missing items are ignored. Returns the count."""
        ...

    async def guarded_write(
        self,
        handle: GraphHandle,
        guard: Guard,
        nodes: Sequence[Node] = (),
        edges: Sequence[Edge] = (),
    ) -> bool:
        """Write `nodes` and `edges` and update the guard node, all or nothing.

        Returns:
            True if the guard held and everything was written, False if the
            guard did not hold and nothing was written.
        """
        ...

    def runner(self, handle: GraphHandle) -> ScopedRunner:
        """Return a runner bound to the handle's graph."""
        ...

    async def stats(self, handle: GraphHandle) -> GraphStats:
        """Counts plus schema drift."""
        ...

    async def health(self) -> Health:
        """Check the server."""
        ...

    async def erase_tenant(self, tenant: str) -> None:
        """Delete the tenant's graph."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
