"""Tests for the classification table and person decisions on one pair."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.temporal.cardinality import ConflictPolicy
from ontogr_mem.temporal.classify import (
    ConflictClass,
    Disposition,
    RelationToTime,
    Side,
    apply_decision,
    classify,
)
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    VerdictSource,
)


def side(
    fact_id: str,
    start: int | None,
    *,
    rank: int = 0,
    corrects: bool = False,
    end: int | None = None,
) -> Side:
    """A side that starts on 1 January of ``start`` and ends at ``end``, if given."""
    valid_from = None if start is None else datetime(start, 1, 1, tzinfo=UTC)
    valid_to = None if end is None else datetime(end, 1, 1, tzinfo=UTC)
    return Side(
        fact_id=fact_id,
        slot_key="slot",
        first_obs_id=fact_id,
        obs_ids=(fact_id,),
        valid_from=valid_from,
        valid_from_prec=Precision.YEAR if start else None,
        valid_to=valid_to,
        rank=rank,
        corrects=corrects,
    )


@pytest.mark.parametrize(
    ("left", "right", "policy", "cls", "disposition"),
    [
        pytest.param(
            side("a", 2010, end=2015),
            side("b", 2015, rank=0),
            ConflictPolicy.REVIEW,
            ConflictClass.UPDATE,
            Disposition.SUPERSEDE,
            id="later_equal_trust_supersedes",
        ),
        pytest.param(
            side("a", 2010, rank=2, end=2015),
            side("b", 2015, rank=1),
            ConflictPolicy.REVIEW,
            ConflictClass.DISPUTE,
            Disposition.COEXIST,
            id="later_less_trusted_does_not_close",
        ),
        pytest.param(
            side("a", 2010),
            side("b", 2015, corrects=True),
            ConflictPolicy.REVIEW,
            ConflictClass.CORRECTION,
            Disposition.CORRECT,
            id="correction_wins_over_the_value_it_replaces",
        ),
        pytest.param(
            side("a", 2010, rank=0),
            side("b", 2010, rank=2),
            ConflictPolicy.AUTO,
            ConflictClass.DISPUTE,
            Disposition.PREFER,
            id="equal_start_prefers_by_gap_under_auto",
        ),
        pytest.param(
            side("a", 2010, rank=0),
            side("b", 2010, rank=2),
            ConflictPolicy.REVIEW,
            ConflictClass.DISPUTE,
            Disposition.COEXIST,
            id="equal_start_coexists_under_review",
        ),
        pytest.param(
            side("a", 2010, rank=2),
            side("b", 2010, rank=2),
            ConflictPolicy.AUTO,
            ConflictClass.DISPUTE,
            Disposition.COEXIST,
            id="auto_with_equal_trust_coexists",
        ),
        pytest.param(
            side("a", None),
            side("b", 2015),
            ConflictPolicy.REVIEW,
            ConflictClass.UNCERTAIN,
            Disposition.REVIEW,
            id="undated_fact_goes_to_review",
        ),
    ],
)
def test_the_table_gives_the_class_and_disposition(
    left, right, policy, cls, disposition
):
    """Each row of the table gives its class and disposition."""
    outcome = classify(left, right, policy=policy, margin=1)

    assert (outcome.cls, outcome.disposition) == (cls, disposition)


def test_overlapping_facts_do_not_supersede_each_other():
    """Facts whose valid periods overlap coexist rather than superseding."""
    outcome = classify(
        side("a", 2010, end=2020),
        side("b", 2015, end=2030),
        policy=ConflictPolicy.REVIEW,
        margin=1,
    )

    assert outcome.relation_to_time is RelationToTime.OVERLAPPING
    assert outcome.disposition is Disposition.COEXIST


def test_open_ended_later_fact_overlaps_an_earlier_fact():
    """A later fact without an end can still overlap an earlier fact."""
    outcome = classify(
        side("a", 2010, end=2020),
        side("b", 2015),
        policy=ConflictPolicy.REVIEW,
        margin=1,
    )

    assert outcome.relation_to_time is RelationToTime.OVERLAPPING
    assert outcome.disposition is Disposition.COEXIST


def test_confirmed_contradiction_supersedes_overlapping_earlier_fact():
    """A confirmed contradiction closes the earlier fact at the later start."""
    outcome = classify(
        side("a", 2010),
        side("b", 2015),
        policy=ConflictPolicy.REVIEW,
        margin=1,
        confirmed_contradiction=True,
    )

    assert outcome.relation_to_time is RelationToTime.OVERLAPPING
    assert outcome.disposition is Disposition.SUPERSEDE
    assert outcome.close_earlier


def test_touching_valid_periods_are_ordered():
    """A fact ending when the next one starts does not overlap it."""
    outcome = classify(
        side("a", 2010, end=2015),
        side("b", 2015, end=2020),
        policy=ConflictPolicy.REVIEW,
        margin=1,
    )

    assert outcome.relation_to_time is RelationToTime.ORDERED
    assert outcome.disposition is Disposition.SUPERSEDE


def test_a_judged_uncertain_pair_is_sent_to_review_even_when_dated():
    """A judge that cannot decide sends a dated pair to a person."""
    outcome = classify(
        side("a", 2010, end=2015),
        side("b", 2015),
        policy=ConflictPolicy.REVIEW,
        margin=1,
        judged_uncertain=True,
    )

    assert outcome.cls is ConflictClass.UNCERTAIN
    assert outcome.relation_to_time is RelationToTime.ORDERED


def test_an_ignored_relation_produces_no_classification():
    """A relation that opts out of conflicts is never classified."""
    assert (
        classify(
            side("a", 2010),
            side("b", 2015),
            policy=ConflictPolicy.IGNORE,
            margin=1,
        )
        is None
    )


def test_a_prefer_names_the_loser_as_overridden():
    """The side that loses a preference is kept with status OVERRIDDEN."""
    outcome = classify(
        side("a", 2010, rank=0),
        side("b", 2010, rank=2),
        policy=ConflictPolicy.AUTO,
        margin=1,
    )

    assert outcome.loser == "a"
    assert outcome.loser_status is FactStatus.OVERRIDDEN


@pytest.mark.parametrize(
    ("action", "chosen", "wrong", "expected"),
    [
        pytest.param(
            DecisionAction.CHOOSE,
            "b",
            "",
            (ConflictClass.DISPUTE, "a", FactStatus.OVERRIDDEN),
            id="choose_overrides_the_other_side",
        ),
        pytest.param(
            DecisionAction.MARK_WRONG,
            "",
            "a",
            (ConflictClass.CORRECTION, "a", FactStatus.CORRECTED),
            id="mark_wrong_corrects_the_named_fact",
        ),
    ],
)
def test_a_person_decision_overrides_the_rules(action, chosen, wrong, expected):
    """A decision names the loser the way the person meant it."""
    decision = ConflictDecision(
        decision_id="d",
        conflict_id="c",
        fact_ids=("a", "b"),
        action=action,
        decided_by=VerdictSource.USER,
        at=datetime(2026, 1, 1, tzinfo=UTC),
        chosen_fact_id=chosen,
        wrong_fact_id=wrong,
    )

    outcome = apply_decision(side("a", 2010), side("b", 2015), decision)

    cls, loser, loser_status = expected
    assert outcome.cls is cls
    assert (outcome.loser, outcome.loser_status) == (loser, loser_status)
    assert outcome.decided_by == "user"


def test_marking_both_valid_keeps_both_sides():
    """Both-valid dismisses the conflict and excludes nothing."""
    decision = ConflictDecision(
        decision_id="d",
        conflict_id="c",
        fact_ids=("a", "b"),
        action=DecisionAction.MARK_BOTH_VALID,
        decided_by=VerdictSource.USER,
        at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    outcome = apply_decision(side("a", 2010), side("b", 2015), decision)

    assert (outcome.disposition, outcome.loser) == (Disposition.COEXIST, "")
