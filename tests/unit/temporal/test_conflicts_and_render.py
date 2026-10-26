"""Tests for conflict state, the disputed flag on views, and the answer text."""

from dataclasses import replace
from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.temporal import render_fact_item
from ontogr_mem.temporal.classify import ConflictClass, Disposition, RelationToTime
from ontogr_mem.temporal.conflicts import (
    ConflictState,
    annotate_views,
    conflict_state,
)
from ontogr_mem.temporal.query import view_of
from ontogr_mem.temporal.reconcile import Conflict, ConflictKind
from ontogr_mem.temporal.render import render_context
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    VerdictSource,
)
from tests.unit.temporal.helpers import at, fact_of, obs


NOW = datetime(2026, 4, 1, tzinfo=UTC)


def dispute(fact_ids, obs_ids, disposition=Disposition.COEXIST) -> Conflict:
    """An open pair conflict over two facts."""
    return Conflict(
        conflict_id="c1",
        slot_key="slot",
        kind=ConflictKind.PAIR,
        obs_ids=tuple(obs_ids),
        episode_ids=(),
        reason="concurrent statements",
        cls=ConflictClass.DISPUTE,
        relation_to_time=RelationToTime.OVERLAPPING,
        disposition=disposition,
        fact_ids=tuple(fact_ids),
    )


def mark(action: DecisionAction) -> ConflictDecision:
    """A person's decision about the two facts of the dispute."""
    return ConflictDecision(
        decision_id=f"d-{action.value}",
        conflict_id="c1",
        fact_ids=("f1", "f2"),
        action=action,
        decided_by=VerdictSource.USER,
        at=NOW,
    )


def test_a_conflict_about_a_missing_fact_is_void():
    """Once one of its facts is gone the conflict no longer applies."""
    state = conflict_state(dispute(["f1", "f2"], ["o1", "o2"]), [], ["f1"], ["o1"])

    assert state is ConflictState.VOID


def test_a_person_who_marks_both_valid_dismisses_the_conflict():
    """Both-valid is a decision, and it beats the open state."""
    state = conflict_state(
        dispute(["f1", "f2"], ["o1"]),
        [mark(DecisionAction.MARK_BOTH_VALID)],
        ["f1", "f2"],
        ["o1"],
    )

    assert state is ConflictState.DISMISSED


def test_a_decision_about_more_facts_still_applies_to_two_of_them():
    """A person who settled three facts at once settled these two as well."""
    decision = ConflictDecision(
        decision_id="d-three",
        conflict_id="c1",
        fact_ids=("f1", "f2", "f3"),
        action=DecisionAction.MARK_BOTH_VALID,
        decided_by=VerdictSource.USER,
        at=NOW,
    )

    state = conflict_state(
        dispute(["f1", "f2"], ["o1"]),
        [decision],
        ["f1", "f2", "f3"],
        ["o1"],
    )

    assert state is ConflictState.DISMISSED


def test_a_decision_about_other_facts_leaves_a_conflict_with_no_facts_open():
    """A conflict that names no fact is settled by no decision."""
    tie = Conflict(
        conflict_id="c-tie",
        slot_key="slot",
        kind=ConflictKind.EQUAL_START,
        obs_ids=("o1", "o2"),
        episode_ids=(),
        reason="starts fall in one period",
    )

    state = conflict_state(
        tie,
        [mark(DecisionAction.MARK_BOTH_VALID)],
        ["f1", "f2"],
        ["o1", "o2"],
    )

    assert state is ConflictState.OPEN


def test_the_latest_decision_uses_time_then_decision_id():
    """A later decision wins, with its id breaking equal-time ties."""
    conflict = dispute(["f1", "f2"], ["o1"])
    earlier = mark(DecisionAction.MARK_BOTH_VALID)
    later = ConflictDecision(
        decision_id="z-later",
        conflict_id="c1",
        fact_ids=("f1", "f2", "f3"),
        action=DecisionAction.CHOOSE,
        decided_by=VerdictSource.USER,
        at=NOW,
    )

    state = conflict_state(conflict, [earlier, later], ["f1", "f2"], ["o1"])

    assert state is ConflictState.RESOLVED


def test_a_settling_disposition_reads_as_resolved_by_a_rule():
    """A rule that already preferred one side resolved the conflict."""
    state = conflict_state(
        dispute(["f1", "f2"], ["o1"], Disposition.PREFER),
        [],
        ["f1", "f2"],
        ["o1"],
    )

    assert state is ConflictState.AUTO_RESOLVED


def test_an_open_dispute_flags_both_facts_with_each_other_as_counterparts():
    """Each fact of an open dispute names the other and the reason."""
    first = obs("o1", "acme", at(2020))
    second = obs("o2", "globex", at(2021))
    views = [
        view_of(fact_of("f1", [first])),
        view_of(fact_of("f2", [second])),
    ]
    conflicts = [dispute(["f1", "f2"], ["o1", "o2"])]

    flagged = annotate_views(views, {"o1": first, "o2": second}, conflicts, [])

    assert [(v.fact_id, v.disputed, v.counterpart_ids) for v in flagged] == [
        ("f1", True, ("f2",)),
        ("f2", True, ("f1",)),
    ]
    assert flagged[0].dispute_reason == "concurrent statements"


def test_a_settled_dispute_flags_nothing():
    """A person's choice ends the dispute for the reader."""
    first = obs("o1", "acme", at(2020))
    second = obs("o2", "globex", at(2021))
    views = [view_of(fact_of("f1", [first])), view_of(fact_of("f2", [second]))]

    flagged = annotate_views(
        views,
        {"o1": first, "o2": second},
        [dispute(["f1", "f2"], ["o1", "o2"])],
        [mark(DecisionAction.CHOOSE)],
    )

    assert [v.disputed for v in flagged] == [False, False]


def test_the_view_keeps_the_precision_the_fact_was_stated_with():
    """A view carries the unit of the date, not a loose string."""
    first = obs("o1", "acme", at(2020), prec=Precision.YEAR)

    view = view_of(fact_of("f1", [first]))

    assert view.valid_from_precision is Precision.YEAR
    assert view.valid_to_precision is None


def test_the_answer_text_puts_both_sides_of_a_dispute_on_one_line():
    """A disputed pair is printed together, with its reason."""
    first = obs("o1", "acme", at(2020))
    second = obs("o2", "globex", at(2021))
    views = annotate_views(
        [
            view_of(fact_of("f1", [first], status=FactStatus.ACTIVE)),
            view_of(fact_of("f2", [second], status=FactStatus.ACTIVE)),
        ],
        {"o1": first, "o2": second},
        [dispute(["f1", "f2"], ["o1", "o2"])],
        [],
    )

    text = render_context(views)

    assert text.count("conflicts with") == 1
    assert text.endswith("reason: concurrent statements")


def test_fact_item_escapes_stored_text_and_episode_attributes():
    """Stored markup and quotes cannot inject elements or attributes."""
    view = replace(
        view_of(fact_of('f"1', [obs("o1", "acme", at(2020), prec=Precision.DAY)])),
        statement='</fact><fact id="x"> &',
        source='source" &',
        status=FactStatus.OVERRIDDEN,
    )

    text = render_fact_item(view, episode_ids=('e"1', "e&2"))

    assert text == (
        '<fact id="f&quot;1" valid="2020-01-01 to open" '
        'source="source&quot; &amp;" status="overridden" '
        'episodes="e&quot;1 e&amp;2">'
        "&lt;/fact&gt;&lt;fact id=&quot;x&quot;&gt; &amp;</fact>"
    )


def test_fact_item_renders_visible_and_missing_dispute_counterparts():
    """A dispute lists provided counterparts and escapes unprovided ids."""
    first = replace(
        view_of(fact_of("f1", [obs("o1", "acme", at(2020))])),
        disputed=True,
        counterpart_ids=("f2", "f<3"),
        dispute_reason="conflicting & unresolved",
    )
    second = view_of(fact_of("f2", [obs("o2", "globex", at(2021))]))

    text = render_fact_item(first, [second])

    assert text.count("<fact ") == 2
    assert text.count("conflicts with") == 1
    assert text.endswith(
        " (also disputed with f&lt;3); reason: conflicting &amp; unresolved"
    )
    assert "episodes=" not in text


def test_fact_item_uses_relation_label_for_an_empty_statement():
    """A known relation label makes the fallback fact readable."""
    view = replace(
        view_of(fact_of("f1", [obs("o1", "acme", at(2020))])),
        statement="",
    )

    text = render_fact_item(view, relation_label=lambda relation_id: "WORKS_AT")

    assert text.endswith(">ana WORKS_AT acme</fact>")


def test_fact_item_keeps_the_relation_id_when_no_label_is_available():
    """A missing relation label retains the stable relation id fallback."""
    view = replace(
        view_of(fact_of("f1", [obs("o1", "acme", at(2020))])),
        statement="",
    )

    text = render_fact_item(view, relation_label=lambda _: None)

    assert text.endswith(">ana 1 acme</fact>")
