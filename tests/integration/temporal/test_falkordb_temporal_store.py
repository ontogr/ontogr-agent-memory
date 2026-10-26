"""The temporal store contract on FalkorDB."""

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION, TEMPORAL
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from tests.contracts.temporal_store import TemporalStoreContract
from tests.support.temporal import TemporalHarness, graph_temporal_harness
from tests.support.tenants import new_tenant


class TestFalkorDBTemporalStore(TemporalStoreContract):
    """`GraphTemporalStore` satisfies the contract on FalkorDB."""

    @pytest.fixture
    async def make_harness(self, falkordb_store: FalkorDBGraphStore):
        """Return a factory of harnesses on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> TemporalHarness:
            handle = await falkordb_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(falkordb_store, handle),
                falkordb_store,
                handle,
                tenant_keyed=False,
            )

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)
