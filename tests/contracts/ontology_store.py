"""Contract for `OntologyStore` implementations."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.errors import (
    MissingVersionError,
    ProposalDecidedError,
    VersionConflict,
)
from ontogr_mem.ontology.models import (
    AddEntity,
    Example,
    Proposal,
    ProposalStatus,
    Unmatched,
    UnmatchedKind,
)
from tests.support.ontology import OntologyHarness, commit_entity, cs


def unmatched_row(label: str, episode_id: str) -> Unmatched:
    """Build an unmatched entity-type row."""
    return Unmatched(
        kind=UnmatchedKind.ENTITY_TYPE,
        raw_label=label,
        normalized_label=label.lower(),
        episode_id=episode_id,
        quote=f"a {label}",
        catalog_version=0,
    )


def proposal(proposal_id: str, status: ProposalStatus) -> Proposal:
    """Build a proposal that adds one entity type."""
    return Proposal(
        proposal_id=proposal_id,
        ops=[AddEntity(label="Gadget")],
        labels=["gadget"],
        support=3,
        examples=[Example(episode_id="e1", quote="a gadget")],
        status=status,
        decided_at=(
            None
            if status is ProposalStatus.PENDING
            else datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
        ),
    )


class OntologyStoreContract:
    """Checks every `OntologyStore` must pass.

    Subclasses provide the `make_harness` fixture, which returns a factory that
    creates a store for a fresh tenant on each call.
    """

    @pytest.fixture
    def make_harness(self) -> Callable[[], Awaitable[OntologyHarness]]:
        """Return a factory for stores on fresh tenants."""
        raise NotImplementedError

    @pytest.fixture
    async def harness(self, make_harness) -> OntologyHarness:
        """A store on a fresh tenant."""
        return await make_harness()

    async def test_first_access_returns_the_genesis_ontology(self, harness):
        """A new tenant starts at version 0 with only the Unknown entity."""
        head = await harness.store.get()

        assert head.version == 0
        assert [e.label for e in head.entities] == ["Unknown"]
        assert await harness.store.history() == []

    async def test_concurrent_first_access_creates_genesis_once(self, harness):
        """Many first readers see the same genesis and write it once."""
        heads = await asyncio.gather(*(harness.store.get() for _ in range(20)))

        assert {h.content_hash for h in heads} == {heads[0].content_hash}
        assert await harness.store.history() == []
        assert (await harness.store.get(0)) == heads[0]

    async def test_commit_stores_the_version_and_moves_the_head(self, harness):
        """A commit on the head adds a snapshot, a change set and a new head."""
        new = await commit_entity(harness.store, "Person")

        assert (await harness.store.get()) == new
        assert (await harness.store.get(1)) == new
        assert (await harness.store.get(0)).version == 0
        history = await harness.store.history()
        assert [c.base_version for c in history] == [0]

    async def test_commit_on_a_stale_base_raises_and_writes_nothing(self, harness):
        """A second commit from the old base loses and leaves the head alone."""
        stale = await harness.store.get()
        await commit_entity(harness.store, "Person")
        change_set = cs(stale, AddEntity(label="Other"))
        losing = apply_changeset(stale, change_set)

        with pytest.raises(VersionConflict):
            await harness.store.commit(losing, change_set)

        assert (await harness.store.get()).version == 1
        assert len(await harness.store.history()) == 1
        with pytest.raises(MissingVersionError):
            await harness.store.get(2)

    async def test_thirty_concurrent_commits_have_one_winner(self, harness):
        """Commits from the same base: one wins, the rest conflict."""
        base = await harness.store.get()

        async def attempt(i: int) -> bool:
            change_set = cs(base, AddEntity(label=f"Type{i}"))
            try:
                await harness.store.commit(
                    apply_changeset(base, change_set), change_set
                )
            except VersionConflict:
                return False
            return True

        results = await asyncio.gather(*(attempt(i) for i in range(30)))

        assert results.count(True) == 1
        assert (await harness.store.get()).version == 1
        assert len(await harness.store.history()) == 1

    async def test_versions_are_consecutive_across_commits(self, harness):
        """Sequential commits number versions 1, 2, 3 and history is ordered."""
        for label in ("A", "B", "C"):
            await commit_entity(harness.store, label)

        history = await harness.store.history()

        assert [c.base_version for c in history] == [0, 1, 2]
        assert (await harness.store.get()).version == 3

    async def test_replace_unmatched_touches_only_that_episode(self, harness):
        """Replacing one episode's rows keeps the rows of the others."""
        store = harness.store
        await store.replace_unmatched("e1", [unmatched_row("Gadget", "e1")])
        await store.replace_unmatched("e2", [unmatched_row("Widget", "e2")])

        await store.replace_unmatched("e1", [unmatched_row("Gizmo", "e1")])
        rows = await store.unmatched()

        assert sorted((r.episode_id, r.raw_label) for r in rows) == [
            ("e1", "Gizmo"),
            ("e2", "Widget"),
        ]

    async def test_replace_unmatched_with_no_rows_clears_the_episode(self, harness):
        """An empty replacement removes the episode's rows."""
        await harness.store.replace_unmatched("e1", [unmatched_row("Gadget", "e1")])

        await harness.store.replace_unmatched("e1", [])

        assert await harness.store.unmatched() == []

    async def test_proposals_save_get_and_list_by_status(self, harness):
        """A saved proposal reads back and lists under its status."""
        store = harness.store
        pending = proposal("p1", ProposalStatus.PENDING)
        rejected = proposal("p2", ProposalStatus.REJECTED)
        await store.save_proposal(pending)
        await store.save_proposal(rejected)

        assert await store.get_proposal("p1") == pending
        assert await store.get_proposal("nope") is None
        assert await store.proposals(ProposalStatus.PENDING) == [pending]
        assert await store.proposals() == [pending, rejected]

    async def test_decided_proposal_is_not_overwritten_by_a_pending_save(self, harness):
        """A pending save over an accepted proposal is refused."""
        store = harness.store
        await store.save_proposal(proposal("p1", ProposalStatus.PENDING))
        await store.save_proposal(proposal("p1", ProposalStatus.ACCEPTED))

        with pytest.raises(ProposalDecidedError):
            await store.save_proposal(proposal("p1", ProposalStatus.PENDING))

        stored = await store.get_proposal("p1")
        assert stored.status is ProposalStatus.ACCEPTED

    async def test_known_episode_ids_lists_the_ingested_episodes(self, harness):
        """Only fully ingested episodes of the tenant are known."""
        await harness.add_episode("e1")
        await harness.add_episode("e2")

        assert await harness.store.known_episode_ids() == {"e1", "e2"}

    async def test_tenants_do_not_see_each_other(self, make_harness):
        """The same ids in two tenants never cross."""
        first, second = await make_harness(), await make_harness()
        await commit_entity(first.store, "Person")
        await first.store.replace_unmatched("e1", [unmatched_row("Gadget", "e1")])
        await first.store.save_proposal(proposal("p1", ProposalStatus.PENDING))
        await first.add_episode("e1")

        assert (await second.store.get()).version == 0
        assert await second.store.history() == []
        assert await second.store.unmatched() == []
        assert await second.store.get_proposal("p1") is None
        assert await second.store.known_episode_ids() == set()
        assert (await first.store.get()).version == 1
