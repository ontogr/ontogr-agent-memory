"""Tests for planning one slot commit against the stored versions."""

from datetime import UTC, datetime

from ontogr_mem.temporal.cardinality import Cardinality
from ontogr_mem.temporal.commit import plan_commit
from ontogr_mem.temporal.reconcile import Conflict, ConflictKind, derive
from ontogr_mem.temporal.temporal_store import SlotState
from tests.unit.temporal.helpers import at, fact_of, obs


RECORDED = datetime(2026, 10, 22, tzinfo=UTC)


def pair_conflict(conflict_id: str) -> Conflict:
    """A pair conflict over one slot."""
    return Conflict(
        conflict_id=conflict_id,
        slot_key="slot",
        kind=ConflictKind.PAIR,
        obs_ids=("o1",),
        episode_ids=(),
        reason="concurrent statements",
    )


def state_with(*conflicts: Conflict) -> SlotState:
    """The slot as a writer read it: one live fact and the given conflicts."""
    observations = [obs("o1", "acme", at(2010))]
    return SlotState(
        slot_key="slot",
        revision=1,
        facts=(fact_of("f1", observations),),
        conflicts=conflicts,
    )


def test_a_commit_carries_the_full_conflict_set_and_says_when_it_changed():
    """The commit holds every conflict of the slot and flags the change."""
    observations = [obs("o1", "acme", at(2010))]
    derived = derive("slot", observations, Cardinality.MANY)

    commit = plan_commit(
        state_with(), observations, derived, RECORDED, [pair_conflict("c1")]
    )

    assert [c.conflict_id for c in commit.conflicts] == ["c1"]
    assert commit.conflicts_changed is True
    assert commit.empty is False


def test_a_commit_that_changes_nothing_carries_the_stored_set_and_is_empty():
    """A commit that leaves the conflicts alone leaves the stored set alone."""
    observations = [obs("o1", "acme", at(2010))]
    derived = derive("slot", observations, Cardinality.MANY)

    commit = plan_commit(
        state_with(pair_conflict("c1")),
        observations,
        derived,
        RECORDED,
        [pair_conflict("c1")],
    )

    assert [c.conflict_id for c in commit.conflicts] == ["c1"]
    assert commit.conflicts_changed is False
    assert commit.empty is True


def test_a_commit_with_no_conflicts_carries_an_empty_set():
    """The conflict set is always a set, never a mode."""
    observations = [obs("o1", "acme", at(2010))]
    derived = derive("slot", observations, Cardinality.MANY)

    commit = plan_commit(state_with(), observations, derived, RECORDED, [])

    assert commit.conflicts == ()
    assert commit.conflicts_changed is False
