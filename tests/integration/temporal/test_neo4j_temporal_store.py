"""The temporal store contract on Neo4j."""

import pytest

from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION, TEMPORAL
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from tests.contracts.temporal_store import TemporalStoreContract
from tests.support.temporal import TemporalHarness, graph_temporal_harness
from tests.support.tenants import new_tenant


class TestNeo4jTemporalStore(TemporalStoreContract):
    """`GraphTemporalStore` satisfies the contract on Neo4j."""

    @pytest.fixture
    async def make_harness(self, neo4j_store: Neo4jGraphStore):
        """Return a factory of harnesses on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> TemporalHarness:
            handle = await neo4j_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(neo4j_store, handle),
                neo4j_store,
                handle,
                tenant_keyed=True,
            )

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)
