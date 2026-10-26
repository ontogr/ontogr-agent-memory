"""Contract for outbox queues, checked on the in-memory and Postgres outboxes."""

import pytest

from ontogr_mem.vectorstore import EmbedItem, Validity, VectorKind
from ontogr_mem.vectorstore.pgvector.outbox import (
    OutboxOp,
    delete_entry,
    upsert_entry,
    validity_entry,
)
from tests.contracts.vector_store import edge_item


NODE, EDGE = VectorKind.NODE, VectorKind.EDGE
CLOSED = Validity("closed", 100, 100, valid_to_us=200)


class OutboxContract:
    """Combining rules, leases and gauges every outbox must share.

    Subclasses provide `make_outbox`, a factory that returns a fresh outbox on a
    fresh tenant.
    """

    @pytest.fixture
    def make_outbox(self):
        """Return a factory for fresh outboxes."""
        raise NotImplementedError

    async def test_reenqueue_overwrites_the_text(self, make_outbox):
        """The newest text for an object is the one the drain sees."""
        outbox = await make_outbox()
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "old"))])
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "new"))])

        claimed = await outbox.claim(10)

        assert [(entry.object_id, entry.text) for entry in claimed] == [("n1", "new")]
        assert await outbox.depth() == 1

    async def test_delete_cancels_pending_upsert_and_validity(self, make_outbox):
        """A delete replaces both a pending upsert and a pending validity change."""
        outbox = await make_outbox()
        await outbox.enqueue(
            [upsert_entry(EDGE, edge_item("f1", "a")), validity_entry("f2", CLOSED)]
        )
        await outbox.enqueue([delete_entry(EDGE, "f1"), delete_entry(EDGE, "f2")])

        claimed = await outbox.claim(10)

        assert sorted((entry.object_id, entry.op) for entry in claimed) == [
            ("f1", OutboxOp.DELETE),
            ("f2", OutboxOp.DELETE),
        ]

    async def test_upsert_cancels_pending_delete(self, make_outbox):
        """A later upsert resurrects the object."""
        outbox = await make_outbox()
        await outbox.enqueue([delete_entry(NODE, "n1")])
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "back"))])

        claimed = await outbox.claim(10)

        assert [(entry.op, entry.text) for entry in claimed] == [
            (OutboxOp.UPSERT, "back")
        ]

    async def test_upsert_cancels_pending_validity_change(self, make_outbox):
        """A new full row supersedes an older validity-only update."""
        outbox = await make_outbox()
        await outbox.enqueue([validity_entry("f1", CLOSED)])
        await outbox.enqueue([upsert_entry(EDGE, edge_item("f1", "new"))])

        claimed = await outbox.claim(10)

        assert [(entry.op, entry.text) for entry in claimed] == [
            (OutboxOp.UPSERT, "new")
        ]

    async def test_validity_is_merged_into_a_pending_upsert(self, make_outbox):
        """A validity change for a pending upsert updates that entry's mirror."""
        outbox = await make_outbox()
        await outbox.enqueue([upsert_entry(EDGE, edge_item("f1", "a"))])
        await outbox.enqueue([validity_entry("f1", CLOSED)])

        claimed = await outbox.claim(10)

        assert len(claimed) == 1
        assert claimed[0].op is OutboxOp.UPSERT
        assert claimed[0].extras is not None
        assert claimed[0].extras.validity == CLOSED

    async def test_validity_alone_is_its_own_row_and_the_latest_wins(self, make_outbox):
        """Without a pending upsert, the newest validity change is stored."""
        outbox = await make_outbox()
        newer = Validity("withdrawn", 100, 300)
        await outbox.enqueue([validity_entry("f1", CLOSED)])
        await outbox.enqueue([validity_entry("f1", newer)])

        claimed = await outbox.claim(10)

        assert [(entry.op, entry.validity) for entry in claimed] == [
            (OutboxOp.VALIDITY, newer)
        ]

    async def test_validity_after_pending_delete_is_ignored(self, make_outbox):
        """A delete wins over a validity change for the same object."""
        outbox = await make_outbox()
        await outbox.enqueue([delete_entry(EDGE, "f1")])
        await outbox.enqueue([validity_entry("f1", CLOSED)])

        assert [entry.op for entry in await outbox.claim(10)] == [OutboxOp.DELETE]

    async def test_ack_of_a_stale_claim_keeps_the_reenqueued_entry(self, make_outbox):
        """A drain that finishes an old version must not delete the new one."""
        outbox = await make_outbox()
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "v1"))])
        claimed = await outbox.claim(10)
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "v2"))])

        await outbox.ack(claimed)

        assert await outbox.depth() == 1
        assert [entry.text for entry in await outbox.claim(10)] == ["v2"]

    async def test_claimed_entries_are_leased(self, make_outbox):
        """A claim hides the entry until the lease expires."""
        outbox = await make_outbox()
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "x"))])

        assert len(await outbox.claim(10)) == 1
        assert await outbox.claim(10) == []

    async def test_release_does_not_count_the_attempt(self, make_outbox):
        """A release gives the entry back without using a try."""
        outbox = await make_outbox()
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "x"))])
        first = await outbox.claim(10)
        assert first[0].attempts == 1

        await outbox.release(first)
        again = await outbox.claim(10)

        assert again[0].attempts == 1

    async def test_entries_become_dead_letters_after_max_attempts(self, make_outbox):
        """A failing entry stops being claimable but stays countable."""
        outbox = await make_outbox()
        await outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "x"))])

        for _ in range(outbox.max_attempts):
            (entry,) = await outbox.claim(10)
            await outbox.fail([entry], "boom")

        assert await outbox.claim(10) == []
        assert await outbox.dead_count() == 1
        assert await outbox.depth() == 1

    async def test_claim_order_and_gauges(self, make_outbox):
        """Upserts drain first, then validity, then deletes."""
        outbox = await make_outbox()
        await outbox.enqueue(
            [
                delete_entry(NODE, "d"),
                validity_entry("v", CLOSED),
                upsert_entry(NODE, EmbedItem("u", "x"), episode_id="ep1"),
            ]
        )

        claimed = await outbox.claim(10)

        assert [entry.op for entry in claimed] == [
            OutboxOp.UPSERT,
            OutboxOp.VALIDITY,
            OutboxOp.DELETE,
        ]
        assert await outbox.pending_for_episode("ep1") == 1
        assert await outbox.pending_for_episode("other") == 0
        assert (await outbox.oldest_age_s() or 0.0) >= 0.0
