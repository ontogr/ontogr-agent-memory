"""Write vectors, keep the items that failed, and write them again."""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.vectorstore.errors import VectorStoreError
from ontogr_mem.vectorstore.models import EmbedItem, FailReason, VectorKind
from ontogr_mem.vectorstore.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class FailedWrite:
    """One item that was not written, with the reason.

    Attributes:
        kind: The kind of vector the item was written as.
        item: The item that was not written.
        reason: Why the item was not written.
        detail: What the store said about the failure.
    """

    kind: VectorKind
    item: EmbedItem
    reason: FailReason
    detail: str = ""


class VectorWriter:
    """Writes vectors to a store and keeps the items that failed.

    The store reports a failure for each item it could not write, and raises
    a typed error when it cannot serve the batch at all. Both are kept, so
    `retry` can write the items again. Any other error propagates, because it
    points at the call rather than at the store.

    Args:
        store: The store that receives the vectors.
    """

    def __init__(self, store: VectorStore) -> None:
        """Bind the store that receives the vectors."""
        self._store = store
        self._failed: tuple[FailedWrite, ...] = ()

    @property
    def failed(self) -> tuple[FailedWrite, ...]:
        """The items that still need writing."""
        return self._failed

    async def write(self, kind: VectorKind, items: Sequence[EmbedItem]) -> None:
        """Write the items and keep the ones that failed.

        Only the items of this call change. An item that failed again is kept
        with its new reason, and an item the store accepted is dropped. The
        items of another kind are left alone.

        Args:
            kind: The kind of vector to write.
            items: The items to write.
        """
        failures = await self._attempt(kind, items)
        attempted = {(kind, item.id) for item in items}
        kept = [
            failure
            for failure in self._failed
            if (failure.kind, failure.item.id) not in attempted
        ]
        self._failed = tuple(kept) + failures

    async def retry(self) -> int:
        """Write the kept items again, grouped by kind.

        Returns:
            The number of items that still fail.
        """
        if not self._failed:
            return 0
        by_kind: dict[VectorKind, list[EmbedItem]] = {}
        for failure in self._failed:
            by_kind.setdefault(failure.kind, []).append(failure.item)
        for kind, items in by_kind.items():
            await self.write(kind, items)
        return len(self._failed)

    async def _attempt(
        self, kind: VectorKind, items: Sequence[EmbedItem]
    ) -> tuple[FailedWrite, ...]:
        """Write the items and return the ones that were not written."""
        try:
            result = await self._store.upsert(kind, items)
        except VectorStoreError as exc:
            return tuple(
                FailedWrite(kind, item, FailReason.STORE_ERROR, str(exc))
                for item in items
            )
        by_id = {item.id: item for item in items}
        return tuple(
            FailedWrite(kind, by_id[failed.id], failed.reason, failed.detail)
            for failed in result.failed
            if failed.id in by_id
        )
