"""A live graph backend that tests open tenants on and erase afterwards."""

from dataclasses import dataclass, field
from typing import Literal

from ontogr_mem.graphstore import GraphHandle
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.schemas import (
    CONSOLIDATION,
    EPISODES,
    KEYWORD,
    ONTOLOGY,
    RESOLUTION,
    TEMPORAL,
    VECTOR,
)
from ontogr_mem.graphstore.tenant_stacks import graph_stores
from ontogr_mem.observability import Tracer
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder
from tests.support.embedder import MockEmbedder
from tests.support.tenants import new_tenant


ALL_SCHEMAS = [
    EPISODES,
    ONTOLOGY,
    RESOLUTION,
    TEMPORAL,
    CONSOLIDATION,
    KEYWORD,
    VECTOR,
]


@dataclass
class GraphBackend:
    """One live graph store and the tenants opened on it during a test.

    Attributes:
        store: The FalkorDB or Neo4j store.
        name: The backend name, as the service settings name it.
        tenants: Tenant ids to erase when the test ends.
    """

    store: GraphStore
    name: Literal["falkordb", "neo4j"]
    tenants: list[str] = field(default_factory=list)

    @property
    def tenant_keyed(self) -> bool:
        """Whether the store keys nodes by tenant, as Neo4j does."""
        return self.name == "neo4j"

    async def open_tenant(self) -> GraphHandle:
        """Open a fresh tenant with every schema and track it for erasure."""
        handle = await self.store.open(new_tenant(), ALL_SCHEMAS)
        self.tenants.append(handle.tenant)
        return handle

    async def open_stores(self, embedder: Embedder | None = None) -> TenantStores:
        """Build every port for a fresh tenant on this backend."""
        return await self.stores_for(new_tenant(), embedder)

    async def stores_for(
        self,
        tenant: str,
        embedder: Embedder | None = None,
        *,
        tracer: Tracer | None = None,
    ) -> TenantStores:
        """Build every port for `tenant` on this backend, erased by `erase`."""
        if tenant not in self.tenants:
            self.tenants.append(tenant)
        return await graph_stores(
            self.store,
            tenant,
            embedder or MockEmbedder(),
            tenant_keyed=self.tenant_keyed,
            tracer=tracer,
        )

    async def erase(self) -> None:
        """Erase every tenant opened through this backend."""
        for tenant in self.tenants:
            await self.store.erase_tenant(tenant)
        self.tenants.clear()
