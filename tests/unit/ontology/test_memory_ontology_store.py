"""The ontology store contract on the in-memory twin."""

import pytest

from ontogr_mem.graphstore.memory import MemoryEpisodeStore, MemoryOntologyStore
from tests.contracts.ontology_store import OntologyStoreContract
from tests.support.ontology import OntologyHarness, ingest_episode


class TestMemoryOntologyStore(OntologyStoreContract):
    """MemoryOntologyStore satisfies the ontology store contract."""

    @pytest.fixture
    def make_harness(self):
        """Return a factory for twins that stand in for fresh tenants."""

        async def make() -> OntologyHarness:
            episodes = MemoryEpisodeStore()

            async def add_episode(episode_id: str) -> None:
                await ingest_episode(episodes, episode_id)

            return OntologyHarness(
                MemoryOntologyStore(episodes.ingested_ids), add_episode
            )

        return make
