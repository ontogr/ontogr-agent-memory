"""The episode store contract on the in-memory twin."""

import pytest

from ontogr_mem.graphstore.memory import MemoryEpisodeStore
from tests.contracts.episode_store import EpisodeStoreContract
from tests.support.episodes import Harness, memory_node_ids


class TestMemoryEpisodeStore(EpisodeStoreContract):
    """MemoryEpisodeStore satisfies the episode store contract."""

    @pytest.fixture
    def make_harness(self):
        """Return a factory for twins that stand in for fresh tenants."""

        async def make() -> Harness:
            twin = MemoryEpisodeStore()

            async def set_temporal_state(episode_id, status, committed_hash):
                twin.set_temporal_state(episode_id, status, committed_hash)

            async def node_ids(label):
                return memory_node_ids(twin, label)

            return Harness(twin, set_temporal_state, node_ids)

        return make
