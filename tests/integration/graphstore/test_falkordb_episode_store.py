"""The episode store contract on FalkorDB."""

import pytest

from ontogr_mem.graphstore.schemas import EPISODES
from tests.contracts.episode_store import EpisodeStoreContract
from tests.support.episodes import graph_harness
from tests.support.tenants import new_tenant


class TestFalkorDBEpisodeStore(EpisodeStoreContract):
    """GraphEpisodeStore satisfies the episode store contract on FalkorDB."""

    @pytest.fixture
    async def make_harness(self, falkordb_store):
        """Return a factory for stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await falkordb_store.open(new_tenant(), [EPISODES])
            tenants.append(handle.tenant)
            return graph_harness(falkordb_store.runner(handle), tenant_keyed=False)

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)
