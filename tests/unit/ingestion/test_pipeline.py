"""Pipeline behavior on the in-memory episode store."""

import pytest

from ontogr_mem.data_models import Role
from ontogr_mem.graphstore.memory import MemoryEpisodeStore
from ontogr_mem.ingestion import (
    ChatMessage,
    DraftValidationError,
    IngestionPipeline,
    UnknownKindError,
    default_registry,
)
from tests.support.clock import FixedClock
from tests.support.episodes import T0, memory_node_ids


def message(message_id: str, text: str = "hello") -> ChatMessage:
    """A chat message at `T0`."""
    return ChatMessage(message_id, "s1", "u1", Role.USER, text, T0)


class FailingHashStore(MemoryEpisodeStore):
    """A twin whose first hash recording fails."""

    def __init__(self) -> None:
        """Create the twin armed to fail once."""
        super().__init__()
        self.fail = True

    async def record_content_hash(self, episode_id: str, content_hash: str) -> None:
        """Fail the first call, then record normally."""
        if self.fail:
            self.fail = False
            msg = "connection lost"
            raise ConnectionError(msg)
        await super().record_content_hash(episode_id, content_hash)


class TestIngestionPipeline:
    """Behavior of IngestionPipeline."""

    @pytest.fixture
    def store(self):
        """An empty episode store."""
        return MemoryEpisodeStore()

    @pytest.fixture
    def handed_off(self):
        """Episodes received by the hand-off."""
        return []

    @pytest.fixture
    def pipeline(self, store, handed_off):
        """A pipeline over the twin with a fixed clock."""

        async def handoff(episodes):
            handed_off.append([e.id for e in episodes])

        return IngestionPipeline(store, default_registry(), FixedClock(), handoff)

    async def test_stores_episodes_with_hash_and_hands_them_off(
        self, pipeline, store, handed_off
    ):
        """Every new episode is stored with its hash and handed off once."""
        await pipeline.ingest("message", [message("m1"), message("m2", "bye")])

        assert len(handed_off) == 1
        assert len(handed_off[0]) == 2
        for episode_id in handed_off[0]:
            stored = await store.get_episode(episode_id)
            assert stored.content_hash is not None
            assert stored.ingested_at == FixedClock().now()

    async def test_second_run_writes_and_hands_off_nothing(self, pipeline, handed_off):
        """Delivering the same input again is a no-op."""
        await pipeline.ingest("message", [message("m1")])

        await pipeline.ingest("message", [message("m1")])

        assert len(handed_off) == 1

    async def test_failed_handoff_is_retried_with_the_same_episode_id(self, store):
        """A failed hand-off retries, then successful duplicates are a no-op."""
        handed_off = []

        async def fail_once(episodes):
            handed_off.append([episode.id for episode in episodes])
            if len(handed_off) == 1:
                msg = "handoff failed"
                raise ConnectionError(msg)

        pipeline = IngestionPipeline(store, default_registry(), FixedClock(), fail_once)

        with pytest.raises(ConnectionError, match="handoff failed"):
            await pipeline.ingest("message", [message("m1")])

        await pipeline.ingest("message", [message("m1")])
        await pipeline.ingest("message", [message("m1")])

        assert len(handed_off) == 2
        assert handed_off[0] == handed_off[1]
        assert len(memory_node_ids(store, "Episode")) == 1

    async def test_duplicates_inside_one_batch_are_stored_once(
        self, pipeline, handed_off
    ):
        """Identical items in a batch give one episode."""
        await pipeline.ingest("message", [message("m1"), message("m1")])

        assert len(handed_off[0]) == 1

    async def test_invalid_item_writes_nothing(self, pipeline, store, handed_off):
        """One bad draft rejects the whole batch."""
        with pytest.raises(DraftValidationError):
            await pipeline.ingest("message", [message("m1"), message("m2", "  ")])

        assert memory_node_ids(store, "Episode") == set()
        assert handed_off == []

    async def test_unknown_kind_raises(self, pipeline):
        """No connector for the kind means an error."""
        with pytest.raises(UnknownKindError):
            await pipeline.ingest("nope", [])

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_batch_size_below_one_is_refused(self, batch_size):
        """A batch size that would write nothing is refused at construction."""
        with pytest.raises(ValueError, match="batch_size"):
            IngestionPipeline(
                MemoryEpisodeStore(),
                default_registry(),
                FixedClock(),
                batch_size=batch_size,
            )

    async def test_batches_are_written_in_chunks(self, store):
        """More items than batch_size are all stored."""
        pipeline = IngestionPipeline(
            store, default_registry(), FixedClock(), batch_size=2
        )

        await pipeline.ingest("message", [message(f"m{i}") for i in range(5)])

        assert len(memory_node_ids(store, "Episode")) == 5

    async def test_failure_before_the_hash_is_recorded_is_repaired_by_a_rerun(
        self, handed_off
    ):
        """A crash leaves no hash; the next run writes again and sets it."""
        store = FailingHashStore()
        pipeline = IngestionPipeline(store, default_registry(), FixedClock())

        with pytest.raises(ConnectionError):
            await pipeline.ingest("message", [message("m1")])
        (episode_id,) = memory_node_ids(store, "Episode")
        assert await store.get_content_hash(episode_id) is None

        await pipeline.ingest("message", [message("m1")])

        assert await store.get_content_hash(episode_id) is not None
