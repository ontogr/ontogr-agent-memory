"""Tenant naming, provisioning and erasure in the shared database."""

import asyncio
from collections.abc import Sequence

from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import tenant_id
from ontogr_mem.graphstore.indices import IndexSpec, SchemaPart, merge_parts, plan
from ontogr_mem.graphstore.neo4j.runner import Neo4jRunner
from ontogr_mem.graphstore.neo4j.schema import SchemaManager
from ontogr_mem.graphstore.reserved import TENANT
from ontogr_mem.graphstore.schemas import ALL, KERNEL
from ontogr_mem.graphstore.types import GraphHandle


ERASE_BATCH = 10_000
"""Nodes removed per transaction when a tenant is erased."""


def graph_name(tenant: str) -> str:
    """Return the name that identifies `tenant`'s data."""
    return f"m_{tenant_id(tenant)}"


class TenantGraphs:
    """Opens tenants against the shared database schema."""

    def __init__(self, runner: Neo4jRunner) -> None:
        """Bind to a runner."""
        self._runner = runner
        self.schema = SchemaManager(runner)
        self._schemas: dict[str, SchemaPart] = {}
        self._provisioned: set[tuple[IndexSpec, ...]] = set()
        self._handle_locks: dict[str, asyncio.Lock] = {}
        self._schema_locks: dict[tuple[IndexSpec, ...], asyncio.Lock] = {}

    async def open(
        self, tenant: str, parts: Sequence[SchemaPart] | None = None
    ) -> GraphHandle:
        """Return the handle, ensuring its shared database schema is present."""
        handle = GraphHandle(tenant, graph_name(tenant))
        wanted = merge_parts(
            [KERNEL, *(ALL if parts is None else parts)],
            {part.name: part for part in ALL},
        )
        shape = plan(wanted)
        handle_lock = self._handle_locks.setdefault(handle.graph, asyncio.Lock())
        async with handle_lock:
            schema_lock = self._schema_locks.setdefault(shape, asyncio.Lock())
            async with schema_lock:
                if shape in self._provisioned:
                    report = await self.schema.verify(wanted)
                    if not report.ok:
                        await self.schema.provision(wanted)
                else:
                    await self.schema.provision(wanted)
                self._provisioned.add(shape)
                self._schemas[handle.graph] = wanted
        return handle

    def schema_of(self, handle: GraphHandle) -> SchemaPart:
        """Return the schema the handle was opened with."""
        return self._schemas.get(handle.graph) or merge_parts([KERNEL, *ALL])

    async def reprovision(self, handle: GraphHandle) -> None:
        """Create any missing index or constraint of the handle's schema."""
        wanted = self.schema_of(handle)
        await self.schema.provision(wanted)
        self._provisioned.add(plan(wanted))

    async def erase(self, tenant: str) -> None:
        """Delete every node and edge of the tenant, in batches."""
        name = graph_name(tenant)
        spec = QuerySpec(
            f"MATCH (n {{{TENANT}: $tenant}}) WITH n LIMIT {ERASE_BATCH} "
            "DETACH DELETE n RETURN count(*) AS removed",
            idempotent=True,
            tag="erase_tenant",
        )
        while (await self._runner.run(spec, tenant)).scalar("removed", 0):
            pass
        self._schemas.pop(name, None)
