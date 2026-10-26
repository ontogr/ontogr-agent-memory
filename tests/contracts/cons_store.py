"""Contract every consolidation store backend must satisfy.

A backend test module subclasses `ConsStoreContract` and provides a `make_store`
fixture that returns an async factory. Each call returns a store on a fresh tenant.
The class has no `Test` prefix, so pytest collects only its subclasses.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.consolidation.errors import StoreError
from ontogr_mem.consolidation.models import (
    FeedbackEvent,
    Proposal,
    ProposalStatus,
    Watermark,
)
from ontogr_mem.consolidation.store import ConsStore


NOW = datetime(2030, 1, 1, tzinfo=UTC)
CONCURRENT = 20
TICKS = 50

MakeStore = Callable[[], Awaitable[ConsStore]]


def event(event_id: str, *, target: str = "x", applied: bool = False) -> FeedbackEvent:
    """Build one feedback event."""
    return FeedbackEvent(
        event_id=event_id,
        target_id=target,
        rating=1.0,
        created_at=NOW,
        applied=applied,
    )


def proposal(proposal_id: str, status: ProposalStatus) -> Proposal:
    """Build one reflection proposal in ``status``."""
    return Proposal(
        proposal_id=proposal_id,
        session_id="s1",
        text="lesson",
        source_entry_ids=["se-1"],
        status=status,
        created_at=NOW,
    )


class ConsStoreContract:
    """Behavior of a consolidation store, shared by every backend."""

    async def test_round_trips_a_document(self, make_store: MakeStore) -> None:
        """A document written with put is read back unchanged."""
        store = await make_store()
        await store.put(Watermark(key="ingest", value="7", updated_at=NOW))

        got = await store.get(Watermark, "ingest")

        assert got == Watermark(key="ingest", value="7", updated_at=NOW)

    async def test_a_returned_document_is_a_copy(self, make_store: MakeStore) -> None:
        """Changing a returned document does not change the stored one."""
        store = await make_store()
        await store.put(event("fb-1"))
        got = await store.get(FeedbackEvent, "fb-1")
        assert got is not None

        got.target_id = "changed"

        again = await store.get(FeedbackEvent, "fb-1")
        assert again is not None and again.target_id == "x"

    async def test_put_replaces_and_delete_removes(self, make_store: MakeStore) -> None:
        """A second put replaces the document and delete removes it."""
        store = await make_store()
        await store.put(event("fb-1", applied=False))
        await store.put(event("fb-1", applied=True))
        replaced = await store.get(FeedbackEvent, "fb-1")
        assert replaced is not None and replaced.applied is True

        await store.delete(FeedbackEvent, "fb-1")

        assert await store.get(FeedbackEvent, "fb-1") is None

    async def test_deleting_a_missing_document_is_not_an_error(
        self, make_store: MakeStore
    ) -> None:
        """Deleting an id that was never written does nothing."""
        store = await make_store()
        await store.delete(FeedbackEvent, "never-written")

    async def test_find_matches_a_declared_field_with_bools_as_text(
        self, make_store: MakeStore
    ) -> None:
        """Find on a declared bool field compares the value as text."""
        store = await make_store()
        await store.put(event("fb-1", applied=True))
        await store.put(event("fb-2", applied=False))
        await store.put(event("fb-3", applied=False))

        pending = await store.find(FeedbackEvent, "applied", False)

        assert [e.event_id for e in pending] == ["fb-2", "fb-3"]

    async def test_find_matches_an_enum_field_by_its_value(
        self, make_store: MakeStore
    ) -> None:
        """Find on a declared enum field compares the enum value."""
        store = await make_store()
        await store.put(proposal("p-1", ProposalStatus.PENDING))
        await store.put(proposal("p-2", ProposalStatus.ACCEPTED))

        found = await store.find(Proposal, "status", ProposalStatus.PENDING)

        assert [p.proposal_id for p in found] == ["p-1"]

    async def test_find_refuses_a_field_the_model_did_not_declare(
        self, make_store: MakeStore
    ) -> None:
        """Find on an undeclared field raises StoreError."""
        store = await make_store()
        with pytest.raises(StoreError):
            await store.find(Watermark, "value", "7")

    async def test_list_is_ordered_by_id(self, make_store: MakeStore) -> None:
        """List returns the documents ordered by id."""
        store = await make_store()
        for event_id in ("fb-2", "fb-3", "fb-1"):
            await store.put(event(event_id))

        listed = await store.list(FeedbackEvent)

        assert [e.event_id for e in listed] == ["fb-1", "fb-2", "fb-3"]

    async def test_tick_is_strictly_increasing(self, make_store: MakeStore) -> None:
        """Tick returns 1, 2, 3 in order."""
        store = await make_store()

        values = [await store.tick() for _ in range(3)]

        assert values == [1, 2, 3]

    async def test_concurrent_ticks_are_distinct_and_consecutive(
        self, make_store: MakeStore
    ) -> None:
        """Concurrent ticks return each value exactly once."""
        store = await make_store()

        values = await asyncio.gather(*(store.tick() for _ in range(TICKS)))

        assert sorted(values) == list(range(1, TICKS + 1))

    async def test_a_live_lease_refuses_another_holder(
        self, make_store: MakeStore
    ) -> None:
        """A live lease refuses another holder."""
        store = await make_store()
        assert await store.acquire("a", NOW, 60)

        assert not await store.acquire("b", NOW + timedelta(seconds=30), 60)

    async def test_the_holder_renews_its_lease(self, make_store: MakeStore) -> None:
        """The holder can renew its lease before it expires."""
        store = await make_store()
        assert await store.acquire("a", NOW, 60)

        assert await store.acquire("a", NOW + timedelta(seconds=50), 60)
        assert not await store.acquire("b", NOW + timedelta(seconds=100), 60)

    async def test_an_expired_lease_is_taken_over(self, make_store: MakeStore) -> None:
        """A lease past its expiry can be taken by another holder."""
        store = await make_store()
        assert await store.acquire("dead", NOW, 60)

        assert await store.acquire("b", NOW + timedelta(seconds=61), 60)

    async def test_a_non_holder_cannot_release(self, make_store: MakeStore) -> None:
        """Only the holder can release the lease."""
        store = await make_store()
        assert await store.acquire("a", NOW, 60)

        assert await store.release("b") is False
        assert not await store.acquire("b", NOW, 60)

    async def test_a_rerun_request_is_claimed_once_by_release(
        self, make_store: MakeStore
    ) -> None:
        """A rerun request is reported once, by the release that follows it."""
        store = await make_store()
        assert await store.acquire("a", NOW, 60)
        await store.request_rerun()

        assert await store.release("a") is True
        assert await store.acquire("a", NOW, 60)
        assert await store.release("a") is False

    async def test_a_rerun_request_without_a_holder_waits_for_the_next_one(
        self, make_store: MakeStore
    ) -> None:
        """A rerun request with no holder is reported to the next holder."""
        store = await make_store()
        await store.request_rerun()

        assert await store.acquire("next", NOW, 60)
        assert await store.release("next") is True

    async def test_concurrent_acquires_yield_exactly_one_holder(
        self, make_store: MakeStore
    ) -> None:
        """Concurrent acquires leave exactly one holder."""
        store = await make_store()

        results = await asyncio.gather(
            *(store.acquire(f"w{i}", NOW, 600) for i in range(CONCURRENT))
        )

        assert results.count(True) == 1

    async def test_tenants_do_not_share_documents_or_leases(
        self, make_store: MakeStore
    ) -> None:
        """Two tenants do not share documents, the lease or the counter."""
        mine = await make_store()
        theirs = await make_store()
        await mine.put(event("fb-1", target="mine"))
        assert await mine.acquire("a", NOW, 600)

        assert await theirs.get(FeedbackEvent, "fb-1") is None
        assert await theirs.list(FeedbackEvent) == []
        assert await theirs.acquire("b", NOW, 600)
        assert await theirs.tick() == 1
