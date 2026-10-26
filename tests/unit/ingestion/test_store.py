"""The episode store port as the reader of the previous episodes."""

from ontogr_mem.data_models import PreviousEpisodes
from ontogr_mem.graphstore.memory import MemoryEpisodeStore
from ontogr_mem.ingestion import EpisodeStore
from tests.support.episodes import make_episode


def as_previous(store: EpisodeStore) -> PreviousEpisodes:
    """Return the store as the reader of the previous episodes.

    The episode store is the port that reads the previous episodes, so a store
    typed as the port has to satisfy the reader protocol.
    """
    return store


class TestEpisodeStore:
    """The episode store port."""

    async def test_store_serves_the_previous_episodes(self):
        """The store that holds the episodes also serves the earlier ones."""
        store = as_previous(MemoryEpisodeStore())
        first = make_episode("e1", minutes=1)
        second = make_episode("e2", minutes=2)
        await store.write_episodes([first, second])
        for episode in (first, second):
            await store.record_content_hash(episode.id, "h")

        previous = await store.before(second, 10)

        assert [episode.id for episode in previous] == ["e1"]
