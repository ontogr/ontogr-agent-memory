"""Contract for `OntologyStore` implementations."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.errors import MissingVersionError, VersionConflict
from ontogr_mem.ontology.models import AddEntity
from tests.support.ontology import OntologyHarness, commit_entity, cs


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

    async def test_tenants_do_not_see_each_other(self, make_harness):
        """The same ids in two tenants never cross."""
        first, second = await make_harness(), await make_harness()
        await commit_entity(first.store, "Person")

        assert (await second.store.get()).version == 0
        assert await second.store.history() == []
        assert (await first.store.get()).version == 1
        assert len(await first.store.history()) == 1
