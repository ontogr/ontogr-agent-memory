"""Turn two conflicting facts into a class and a disposition.

The table, read top to bottom, where the first matching row decides::

    judge uncertain, or a fact has no dates    uncertain   review
    a correction, trust >= the other side      correction  correct
    ordered, later trust >= earlier            update      supersede
    ordered, later trust <  earlier            dispute     coexist
    overlap or equal start, gap >= margin,
        policy auto                            dispute     prefer (higher trust)
    overlap or equal start, otherwise          dispute     coexist
    policy ignore                              (nothing)
    a person's decision                        as decided; authoritative

Trust never orders facts in time. "Trust" is the integer rank of a fact's
sources (see ``trust.py``), and "margin" is how much higher one rank must be
before the system prefers it automatically. "Ordered" means the two facts do
not overlap in valid time; "overlap or equal start" means they claim the same
period, where dates cannot say which is right.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.temporal.cardinality import ConflictPolicy
from ontogr_mem.temporal.timeutil import same_span
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    FactFingerprint,
)


class ConflictClass(StrEnum):
    """What the two statements are."""

    UPDATE = "update"
    CORRECTION = "correction"
    DISPUTE = "dispute"
    UNCERTAIN = "uncertain"


class RelationToTime(StrEnum):
    """How the two statements sit on the valid-time axis."""

    ORDERED = "ordered"
    OVERLAPPING = "overlapping"
    EQUAL_START = "equal_start"
    UNKNOWN = "unknown"


class Disposition(StrEnum):
    """What is done about a conflict."""

    SUPERSEDE = "supersede"
    CORRECT = "correct"
    PREFER = "prefer"
    COEXIST = "coexist"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class Side:
    """One fact as the classifier sees it.

    Attributes:
        fact_id: Stable id of the fact.
        slot_key: Slot the fact belongs to.
        first_obs_id: Observation that anchors the fact.
        obs_ids: Every observation that supports the fact.
        valid_from: Start of the fact, or ``None`` when undated.
        valid_from_prec: Precision of the start.
        valid_to: End of the fact, or ``None`` while open.
        rank: Best trust rank among the supporting observations.
        corrects: True when one supporting observation is a correction.
        fp: What a verdict depends on: the identity of the fact, its slot and
            the first statement. The end is left out, because it is the effect
            of a verdict.
    """

    fact_id: str
    slot_key: str
    first_obs_id: str
    obs_ids: tuple[str, ...]
    valid_from: datetime | None
    valid_from_prec: Precision | None
    valid_to: datetime | None
    rank: int
    corrects: bool = False
    fp: FactFingerprint = FactFingerprint(slot_key="")


@dataclass(frozen=True, slots=True)
class Classification:
    """What to do about one conflicting pair.

    Attributes:
        cls: What kind of disagreement the pair is.
        disposition: What is done about it.
        relation_to_time: How the two sit on the valid-time axis.
        reason: Plain explanation for a reader.
        loser: Fact excluded from the current view, or empty.
        loser_status: Status the excluded fact takes.
        close_earlier: True when the earlier fact ends where the later starts.
        trust_inputs: Trust ranks used, for later explanation.
        decided_by: ``rule`` or the name of the person decision that applied.
    """

    cls: ConflictClass
    disposition: Disposition
    relation_to_time: RelationToTime
    reason: str
    loser: str = ""
    loser_status: FactStatus | None = None
    close_earlier: bool = False
    trust_inputs: tuple[tuple[str, int], ...] = ()
    decided_by: str = "rule"


def order(x: Side, y: Side) -> tuple[Side, Side]:
    """Return the pair with the earlier-starting fact first, ties by fact id."""
    return (x, y) if _key(x) <= _key(y) else (y, x)


def _key(side: Side) -> tuple[int, float, str]:
    """Sort dated facts before undated ones, then by start, then by fact id."""
    return (
        1 if side.valid_from is None else 0,
        0.0 if side.valid_from is None else side.valid_from.timestamp(),
        side.fact_id,
    )


def time_relation(a: Side, b: Side) -> RelationToTime:
    """Return how the two sides sit on the valid-time axis, ``a`` first."""
    if a.valid_from is None or b.valid_from is None:
        return RelationToTime.UNKNOWN
    if same_span(a.valid_from, a.valid_from_prec, b.valid_from, b.valid_from_prec):
        return RelationToTime.EQUAL_START
    if a.valid_to is None or a.valid_to > b.valid_from:
        return RelationToTime.OVERLAPPING
    return RelationToTime.ORDERED


def classify(  # noqa: PLR0911
    x: Side,
    y: Side,
    *,
    policy: ConflictPolicy,
    margin: int,
    judged_uncertain: bool = False,
    confirmed_contradiction: bool = False,
) -> Classification | None:
    """Classify a pair that the rules or the judge say cannot both stand.

    Args:
        x: One fact in the pair.
        y: The other fact in the pair.
        policy: Rules for handling a contradiction.
        margin: Trust-rank gap needed for an automatic preference.
        judged_uncertain: Whether the judge could not decide the pair.
        confirmed_contradiction: Whether a judge confirmed the pair conflicts.

    Returns:
        The classification, or ``None`` when the policy ignores the relation.
    """
    if policy is ConflictPolicy.IGNORE:
        return None
    a, b = order(x, y)
    rtt = time_relation(a, b)
    inputs = ((a.fact_id, a.rank), (b.fact_id, b.rank))
    if judged_uncertain or rtt is RelationToTime.UNKNOWN:
        why = (
            "the judge could not decide" if judged_uncertain else "a fact has no dates"
        )
        return Classification(
            ConflictClass.UNCERTAIN,
            Disposition.REVIEW,
            rtt,
            why,
            trust_inputs=inputs,
        )
    corrector = b if b.corrects else (a if a.corrects else None)
    if corrector is not None:
        wrong = a if corrector is b else b
        if corrector.rank >= wrong.rank:
            return Classification(
                ConflictClass.CORRECTION,
                Disposition.CORRECT,
                rtt,
                "a correction says the other statement was wrong",
                loser=wrong.fact_id,
                loser_status=FactStatus.CORRECTED,
                trust_inputs=inputs,
            )
    if (
        confirmed_contradiction
        and rtt is RelationToTime.OVERLAPPING
        and b.rank >= a.rank
    ):
        return Classification(
            ConflictClass.UPDATE,
            Disposition.SUPERSEDE,
            rtt,
            "a later statement from an equal or more trusted source "
            "contradicts an earlier one",
            close_earlier=True,
            trust_inputs=inputs,
        )
    if rtt is RelationToTime.ORDERED:
        if b.rank >= a.rank:
            return Classification(
                ConflictClass.UPDATE,
                Disposition.SUPERSEDE,
                rtt,
                "a later statement from an equal or more trusted source",
                close_earlier=True,
                trust_inputs=inputs,
            )
        return Classification(
            ConflictClass.DISPUTE,
            Disposition.COEXIST,
            rtt,
            "a later statement from a less trusted source does not close a fact",
            trust_inputs=inputs,
        )
    gap = abs(a.rank - b.rank)
    if policy is ConflictPolicy.AUTO and gap >= margin:
        winner, loser = (a, b) if a.rank > b.rank else (b, a)
        return Classification(
            ConflictClass.DISPUTE,
            Disposition.PREFER,
            rtt,
            f"preferred {winner.fact_id} by trust gap {gap} >= margin {margin}",
            loser=loser.fact_id,
            loser_status=FactStatus.OVERRIDDEN,
            trust_inputs=inputs,
        )
    return Classification(
        ConflictClass.DISPUTE,
        Disposition.COEXIST,
        rtt,
        "concurrent statements and no trust gap that permits a preference",
        trust_inputs=inputs,
    )


def apply_decision(x: Side, y: Side, decision: ConflictDecision) -> Classification:
    """Apply a person's decision, which overrides every rule.

    ``CHOOSE`` keeps one fact and overrides the other, ``MARK_WRONG`` marks one
    fact corrected, and every other action keeps both sides.
    """
    a, b = order(x, y)
    rtt = time_relation(a, b)
    if decision.action is DecisionAction.CHOOSE:
        loser = y if decision.chosen_fact_id == x.fact_id else x
        return Classification(
            ConflictClass.DISPUTE,
            Disposition.PREFER,
            rtt,
            decision.reason or "chosen by a person",
            loser=loser.fact_id,
            loser_status=FactStatus.OVERRIDDEN,
            decided_by=decision.decided_by.value,
        )
    if decision.action is DecisionAction.MARK_WRONG:
        return Classification(
            ConflictClass.CORRECTION,
            Disposition.CORRECT,
            rtt,
            decision.reason or "marked wrong by a person",
            loser=decision.wrong_fact_id,
            loser_status=FactStatus.CORRECTED,
            decided_by=decision.decided_by.value,
        )
    return Classification(
        ConflictClass.DISPUTE,
        Disposition.COEXIST,
        rtt,
        decision.reason or "both statements are valid",
        decided_by=decision.decided_by.value,
    )
