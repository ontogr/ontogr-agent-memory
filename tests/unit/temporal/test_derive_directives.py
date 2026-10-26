"""Tests for ``derive`` with the handler's directives: exclusions, no-cut, closes."""

from ontogr_mem.data_models.fact_versions import FactStatus, SupersessionKind
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.temporal.cardinality import Cardinality
from ontogr_mem.temporal.reconcile import Close, SlotDirectives, derive
from tests.unit.temporal.helpers import at, obs


A10 = obs("a10", "A", at(2010))
B15 = obs("b15", "B", at(2015))


def intervals(derived) -> list[tuple[str, object, object]]:
    """Each run as ``(value, valid_from, valid_to)``, in opening order."""
    return [(r.value_id, r.valid_from, r.valid_to) for r in derived.runs]


def test_an_excluded_observation_takes_no_part_in_the_walk():
    """An overridden assertion does not cut the value that stays."""
    derived = derive(
        "slot",
        [A10, B15],
        Cardinality.ONE,
        SlotDirectives(excluded={"b15": FactStatus.OVERRIDDEN}),
    )

    assert intervals(derived) == [("A", at(2010), None)]
    [(shadow_run, status)] = derived.shadow
    assert (shadow_run.value_id, status) == ("B", FactStatus.OVERRIDDEN)


def test_a_no_cut_group_keeps_both_values_open():
    """Two values in an open dispute do not cut each other."""
    derived = derive(
        "slot",
        [A10, B15],
        Cardinality.ONE,
        SlotDirectives(no_cut=(frozenset({"a10", "b15"}),)),
    )

    assert intervals(derived) == [("A", at(2010), None), ("B", at(2015), None)]


def test_a_judged_close_ends_the_run_that_holds_its_anchor():
    """A close from another slot ends the local run at its instant."""
    close = Close(
        anchor_obs_id="a10", at=at(2013), at_precision=Precision.YEAR, cause_id="v1"
    )

    derived = derive("slot", [A10], Cardinality.ONE, SlotDirectives(closes=(close,)))

    assert intervals(derived) == [("A", at(2010), at(2013))]
    assert derived.runs[0].close_kind is SupersessionKind.VERDICT
    assert derived.runs[0].closed_by == "v1"
    assert derived.unmatched_closes == ()


def test_a_judged_close_with_no_open_run_is_reported_as_unmatched():
    """A close whose run is not open is kept for review, not applied."""
    close = Close(
        anchor_obs_id="missing", at=at(2013), at_precision=None, cause_id="v9"
    )

    derived = derive("slot", [A10], Cardinality.ONE, SlotDirectives(closes=(close,)))

    assert intervals(derived) == [("A", at(2010), None)]
    assert derived.unmatched_closes == ("v9",)
