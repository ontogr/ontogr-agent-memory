"""Tests for the hand-off that writes a vector for every episode."""

from ontogr_mem.vectorstore import VectorKind, VectorStore, VectorWriter
from ontogr_mem.vectorstore.episode_vectors import write_episode_vectors
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.embedder import MockEmbedder, RecordingEmbedder, gateway
from tests.support.episodes import make_episode
from tests.support.vectors import A


async def ready_store(embedder: MockEmbedder) -> VectorStore:
    """Return a store that is ready to write."""
    store = VectorStore(InMemoryBackend(), gateway(embedder))
    await store.ensure_ready()
    return store


class TestEpisodeVectors:
    """Behavior of the hand-off that writes episode vectors."""

    async def test_writes_one_vector_per_episode(self) -> None:
        """Each episode of the hand-off gets a vector."""
        embedder = RecordingEmbedder(vectors={"Ana uses Postgres": A})
        store = await ready_store(embedder)
        writer = VectorWriter(store)
        episodes = [
            make_episode("ep1", content="Ana uses Postgres"),
            make_episode("ep2", content="Ana works for Acme"),
        ]

        await write_episode_vectors(writer, episodes)

        assert await store.exists(VectorKind.EPISODE, ["ep1", "ep2"]) == {"ep1", "ep2"}
        assert "Ana uses Postgres" in embedder.embedded

    async def test_writes_the_canonical_json_of_a_structured_episode(self) -> None:
        """A structured episode is written as the JSON the graph stores."""
        embedder = RecordingEmbedder(vectors={'{"a":"x","b":2}': A})
        store = await ready_store(embedder)
        writer = VectorWriter(store)
        episode = make_episode("ep1", content={"b": 2, "a": "x"})

        await write_episode_vectors(writer, [episode])

        hits = await store.search(VectorKind.EPISODE, A, k=10)

        assert [hit.id for hit in hits] == ["ep1"]
        assert '{"a":"x","b":2}' in embedder.embedded
