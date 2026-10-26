"""The resolver store contract on FalkorDB."""

import pytest

from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION
from tests.contracts.resolver_store import ResolverStoreContract
from tests.support.resolver import graph_resolver_harness
from tests.support.tenants import new_tenant


class TestFalkorDBResolverStore(ResolverStoreContract):
    """`GraphResolverStore` satisfies the contract on FalkorDB."""

    def setup_method(self) -> None:
        """Keep the harness of each store the contract asks for."""
        self.harnesses = {}

    @pytest.fixture
    async def make_harness(self, falkordb_store):
        """Return a factory for stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await falkordb_store.open(new_tenant(), [EPISODES, RESOLUTION])
            tenants.append(handle.tenant)
            return graph_resolver_harness(
                falkordb_store.runner(handle), tenant_keyed=False
            )

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)

    async def _store(self, make_harness):
        harness = await make_harness()
        self.harnesses[id(harness.store)] = harness
        return harness.store

    @pytest.fixture
    async def store(self, make_harness):
        """Return a store bound to a fresh tenant."""
        return await self._store(make_harness)

    @pytest.fixture
    async def other_store(self, make_harness):
        """Return a store bound to a different fresh tenant."""
        return await self._store(make_harness)

    async def seed_fact_edge(self, store, entity_id, *, live):
        """Write one live or withdrawn fact edge for the entity."""
        await self.harnesses[id(store)].seed_fact_edge(store, entity_id, live=live)

    async def fact_edge_count(self, store, entity_id, *, live):
        """Count the live or withdrawn fact edges of the entity."""
        return await self.harnesses[id(store)].fact_edge_count(
            store, entity_id, live=live
        )
