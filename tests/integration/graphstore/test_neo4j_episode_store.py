"""The episode store contract on Neo4j."""

import pytest

from ontogr_mem.graphstore.schemas import EPISODES
from tests.contracts.episode_store import EpisodeStoreContract
from tests.support.episodes import graph_harness
from tests.support.tenants import new_tenant


class TestNeo4jEpisodeStore(EpisodeStoreContract):
    """GraphEpisodeStore satisfies the episode store contract on Neo4j."""

    @pytest.fixture
    async def make_harness(self, neo4j_store):
        """Return a factory for stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await neo4j_store.open(new_tenant(), [EPISODES])
            tenants.append(handle.tenant)
            return graph_harness(neo4j_store.runner(handle), tenant_keyed=True)

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)
