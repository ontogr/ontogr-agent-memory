"""Graph-per-tenant naming, provisioning and erasure."""

import asyncio
from collections.abc import Sequence

from ontogr_mem.graphstore.falkordb.client import FalkorClientPool
from ontogr_mem.graphstore.falkordb.schema import SchemaManager
from ontogr_mem.graphstore.ident import tenant_id
from ontogr_mem.graphstore.indices import SchemaPart, merge_parts
from ontogr_mem.graphstore.schemas import ALL, KERNEL
from ontogr_mem.graphstore.types import GraphHandle


def graph_name(tenant: str) -> str:
    """Return the name of the graph that holds `tenant`'s data."""
    return f"m_{tenant_id(tenant)}"


class TenantGraphs:
    """Opens tenant graphs, provisioning each once per process and schema."""

    def __init__(self, pool: FalkorClientPool) -> None:
        """Bind to a pool."""
        self._pool = pool
        self.schema = SchemaManager(pool.runner)
        self._provisioned: dict[str, SchemaPart] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def open(
        self, tenant: str, parts: Sequence[SchemaPart] | None = None
    ) -> GraphHandle:
        """Return the handle, provisioning the graph if this schema is new to it."""
        handle = GraphHandle(tenant, graph_name(tenant))
        wanted = merge_parts(
            [KERNEL, *(ALL if parts is None else parts)],
            {part.name: part for part in ALL},
        )
        if self._provisioned.get(handle.graph) == wanted:
            return handle
        lock = self._locks.setdefault(handle.graph, asyncio.Lock())
        async with lock:
            if self._provisioned.get(handle.graph) != wanted:
                await self.schema.provision(handle.graph, wanted)
                self._provisioned[handle.graph] = wanted
        return handle

    def schema_of(self, handle: GraphHandle) -> SchemaPart:
        """Return the schema the handle was opened with."""
        return self._provisioned.get(handle.graph) or merge_parts([KERNEL, *ALL])

    async def reprovision(self, handle: GraphHandle) -> None:
        """Create any missing index or constraint of the handle's schema."""
        await self.schema.provision(handle.graph, self.schema_of(handle))

    async def erase(self, tenant: str) -> None:
        """Delete the tenant's graph."""
        name = graph_name(tenant)
        await self._pool.delete_graph(name)
        self._provisioned.pop(name, None)
