"""Tests for chunked writes with an item-by-item fallback."""

from ontogr_mem.graphstore.batching import (
    BatchOutcome,
    BatchReply,
    to_write_result,
    write_batches,
)
from ontogr_mem.graphstore.errors import StoreQueryError


def rows(n):
    """Build n batch rows with indexes 0..n-1."""
    return [{"i": i, "id": f"n{i}"} for i in range(n)]


class TestWriteBatches:
    """Behavior of write_batches."""

    async def test_splits_into_chunks(self):
        """Items are sent in chunks of the batch size."""
        sizes = []

        async def send(chunk):
            sizes.append(len(chunk))
            return BatchReply([item["i"] for item in chunk], len(chunk))

        outcome = BatchOutcome()
        await write_batches(outcome, rows(5), 2, send)

        assert sizes == [2, 2, 1]
        assert to_write_result(5, outcome).complete
        assert outcome.created == 5

    async def test_bad_item_is_reported_by_index_and_the_rest_is_written(self):
        """A rejected chunk falls back to single items."""

        async def send(chunk):
            if any(item["i"] == 3 for item in chunk):
                msg = "bad item"
                raise StoreQueryError(msg)
            return BatchReply([item["i"] for item in chunk], len(chunk))

        outcome = BatchOutcome()
        await write_batches(outcome, rows(6), 10, send)
        result = to_write_result(6, outcome)

        assert result.fell_back_to_single
        assert [e.index for e in result.errors] == [3]
        assert result.matched == 5
        assert not result.complete
        assert result.missing == ()

    async def test_items_the_server_skipped_are_reported_as_missing(self):
        """An item absent from the reply is missing, not failed."""

        async def send(chunk):
            return BatchReply([0], 1)

        outcome = BatchOutcome()
        await write_batches(outcome, rows(2), 10, send)

        assert to_write_result(2, outcome).missing == (1,)
