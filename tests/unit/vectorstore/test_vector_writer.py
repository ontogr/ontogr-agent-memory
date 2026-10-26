"""Tests for the writer that keeps the items it could not write."""

from collections.abc import Sequence

import pytest

from ontogr_mem.data_models import FactStatus
from ontogr_mem.vectorstore import (
    EdgeExtras,
    EmbedItem,
    FailReason,
    Validity,
    VectorKind,
    VectorStore,
    VectorWriter,
)
from ontogr_mem.vectorstore.errors import QueryTimeoutError
from ontogr_mem.vectorstore.memory import InMemoryBackend
from ontogr_mem.vectorstore.models import BackendWrite, Row
from tests.support.embedder import MockEmbedder, RecordingEmbedder, gateway


def edge(fact_id: str, text: str = "Ana uses Postgres") -> EmbedItem:
    """Build an edge item for the writer tests."""
    return EmbedItem(
        fact_id,
        text,
        EdgeExtras("s1", "o1", Validity(FactStatus.ACTIVE, 0, 0)),
    )


class FailingBackend(InMemoryBackend):
    """An in-memory backend whose writes fail with a scripted error.

    Args:
        error: The error that every write raises.
    """

    def __init__(self, error: Exception) -> None:
        """Create the backend."""
        super().__init__()
        self.error = error

    async def write(self, kind: VectorKind, rows: Sequence[Row]) -> BackendWrite:
        """Raise the scripted error instead of storing the rows."""
        raise self.error


async def ready_store(
    embedder: MockEmbedder, backend: InMemoryBackend | None = None
) -> VectorStore:
    """Return a store that is ready to write."""
    store = VectorStore(backend or InMemoryBackend(), gateway(embedder))
    await store.ensure_ready()
    return store


class TestVectorWriter:
    """Behavior of the writer that keeps the items that failed."""

    async def test_keeps_the_item_when_the_embedder_fails(self) -> None:
        """A failed item stays in `failed` with the reason the store gave."""
        embedder = RecordingEmbedder()
        store = await ready_store(embedder)
        writer = VectorWriter(store)
        embedder.failing = True

        await writer.write(VectorKind.EDGE, [edge("f1")])

        assert [failure.item.id for failure in writer.failed] == ["f1"]
        assert writer.failed[0].reason is FailReason.EMBED_FAILED
        assert await store.exists(VectorKind.EDGE, ["f1"]) == set()

    async def test_retry_writes_the_kept_item_once_the_embedder_recovers(self) -> None:
        """The kept item is written when the embedder works again."""
        embedder = RecordingEmbedder()
        store = await ready_store(embedder)
        writer = VectorWriter(store)
        embedder.failing = True
        await writer.write(VectorKind.EDGE, [edge("f1")])
        embedder.failing = False

        still_failing = await writer.retry()

        assert still_failing == 0
        assert writer.failed == ()
        assert await store.exists(VectorKind.EDGE, ["f1"]) == {"f1"}

    async def test_keeps_a_failure_of_an_earlier_write(self) -> None:
        """A later write does not drop the item that still needs writing."""
        embedder = RecordingEmbedder()
        store = await ready_store(embedder)
        writer = VectorWriter(store)
        embedder.failing = True
        await writer.write(VectorKind.EDGE, [edge("f1")])
        embedder.failing = False

        await writer.write(VectorKind.EDGE, [edge("f2")])

        assert [failure.item.id for failure in writer.failed] == ["f1"]
        assert await store.exists(VectorKind.EDGE, ["f1", "f2"]) == {"f2"}

    async def test_retry_writes_nothing_when_no_item_was_kept(self) -> None:
        """A writer with no failed item has nothing to retry."""
        embedder = RecordingEmbedder()
        store = await ready_store(embedder)
        writer = VectorWriter(store)

        assert await writer.retry() == 0
        assert writer.failed == ()

    async def test_keeps_every_item_when_the_store_raises_a_typed_error(self) -> None:
        """A store that cannot serve the batch keeps all of it."""
        store = await ready_store(
            RecordingEmbedder(),
            backend=FailingBackend(QueryTimeoutError("write timed out")),
        )
        writer = VectorWriter(store)

        await writer.write(VectorKind.EDGE, [edge("f1"), edge("f2")])

        assert [failure.item.id for failure in writer.failed] == ["f1", "f2"]
        assert {failure.reason for failure in writer.failed} == {FailReason.STORE_ERROR}
        assert "write timed out" in writer.failed[0].detail

    async def test_lets_an_untyped_error_from_the_store_propagate(self) -> None:
        """An error the store did not type belongs to the caller."""
        store = await ready_store(
            RecordingEmbedder(),
            backend=FailingBackend(ValueError("the row is not valid")),
        )
        writer = VectorWriter(store)

        with pytest.raises(ValueError):
            await writer.write(VectorKind.EDGE, [edge("f1")])

        assert writer.failed == ()

    async def test_does_not_embed_an_unchanged_text_again(self) -> None:
        """The store embeds a text once, however often it is written."""
        embedder = RecordingEmbedder()
        store = await ready_store(embedder)
        writer = VectorWriter(store)
        item = edge("f1", "Ana uses Postgres")

        await writer.write(VectorKind.EDGE, [item])
        await writer.write(VectorKind.EDGE, [item])

        assert embedder.embedded.count("Ana uses Postgres") == 1
