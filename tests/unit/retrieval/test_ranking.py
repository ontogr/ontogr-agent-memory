"""Tests for the status penalties applied to the fused facts."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models import FactStatus
from ontogr_mem.retrieval import FactResult, RetrievalSettings
from ontogr_mem.retrieval.ranking import apply_penalties
from ontogr_mem.temporal.query import FactView


SETTINGS = RetrievalSettings()


def make_view(
    fact_id: str,
    *,
    disputed: bool = False,
    status: FactStatus = FactStatus.ACTIVE,
) -> FactView:
    """Build a fact view with the given status and dispute flag."""
    return FactView(
        fact_id=fact_id,
        version=1,
        slot_key="slot-1",
        subject_id="entity-1",
        object_id="entity-2",
        relation_type_id=7,
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=None,
        valid_from_precision=None,
        valid_to_precision=None,
        recorded_at=datetime(2020, 1, 1, tzinfo=UTC),
        expired_at=None,
        status=status,
        evidence=("observation-1",),
        disputed=disputed,
    )


class TestApplyPenalties:
    """The status penalties of the fused facts."""

    def test_puts_the_disputed_fact_above_the_overridden_one(self) -> None:
        """Two facts with equal scores are ordered by the penalty they earn."""
        facts = [
            FactResult(
                view=make_view("f2", status=FactStatus.OVERRIDDEN),
                score=1.0,
            ),
            FactResult(view=make_view("f1", disputed=True), score=1.0),
        ]

        result = apply_penalties(facts, limit=10, settings=SETTINGS)

        assert [fact.view.fact_id for fact in result] == ["f1", "f2"]
        assert result[0].score == 1.0
        assert result[1].score == pytest.approx(0.875)

    def test_multiplies_a_fact_that_is_disputed_and_overridden(self) -> None:
        """A fact with both statuses gets the product of the two penalties."""
        facts = [
            FactResult(view=make_view("f1"), score=1.0),
            FactResult(
                view=make_view("f2", disputed=True, status=FactStatus.OVERRIDDEN),
                score=1.0,
            ),
        ]

        result = apply_penalties(facts, limit=10, settings=SETTINGS)

        assert result[1].score == pytest.approx(0.56)

    def test_leaves_a_fact_without_a_status_alone(self) -> None:
        """An active fact that is not disputed keeps its score."""
        facts = [FactResult(view=make_view("f1"), score=0.4)]

        result = apply_penalties(facts, limit=10, settings=SETTINGS)

        assert result[0].score == pytest.approx(1.0)

    def test_breaks_a_tie_with_the_fact_key(self) -> None:
        """Two facts with the same score keep the order of their keys."""
        facts = [
            FactResult(view=make_view("f2"), score=1.0),
            FactResult(view=make_view("f1"), score=1.0),
        ]

        result = apply_penalties(facts, limit=10, settings=SETTINGS)

        assert [fact.view.fact_id for fact in result] == ["f1", "f2"]

    def test_cuts_the_list_at_the_limit(self) -> None:
        """Only the first facts of the ranked list survive the cut."""
        facts = [
            FactResult(view=make_view("f1"), score=0.9),
            FactResult(view=make_view("f2"), score=0.8),
            FactResult(view=make_view("f3"), score=0.7),
        ]

        result = apply_penalties(facts, limit=2, settings=SETTINGS)

        assert [fact.view.fact_id for fact in result] == ["f1", "f2"]

    def test_leaves_the_given_facts_unchanged(self) -> None:
        """The penalty works on copies, so the fused list stays as it was."""
        fact = FactResult(view=make_view("f1", disputed=True), score=1.0)

        apply_penalties([fact], limit=10, settings=SETTINGS)

        assert fact.score == 1.0
