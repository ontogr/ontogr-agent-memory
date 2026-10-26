"""Time rules for fact extraction."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.facts import FactKind, Modality, Precision, TimeBasis
from ontogr_mem.fact_extraction import FactDropReason
from ontogr_mem.fact_extraction.timeparse import parse_instant, resolve_time
from tests.support.facts import T0, TEXT


def resolve(**kw: object):
    """Resolve a time with the test defaults."""
    args: dict[str, object] = {
        "valid_from": None,
        "valid_to": None,
        "declared": None,
        "expression": None,
        "kind": FactKind.STATE,
        "modality": Modality.ACTUAL,
        "episode_text": TEXT,
        "reference": T0,
    }
    return resolve_time(**(args | kw))  # type: ignore[arg-type]


class TestTime:
    """Parsing and resolving times."""

    @pytest.mark.parametrize(
        ("raw", "when", "precision"),
        [
            ("2024", datetime(2024, 1, 1, tzinfo=UTC), Precision.YEAR),
            ("2024-03", datetime(2024, 3, 1, tzinfo=UTC), Precision.MONTH),
            ("2024-03-05", datetime(2024, 3, 5, tzinfo=UTC), Precision.DAY),
            (
                "2024-03-05T10:30:00Z",
                datetime(2024, 3, 5, 10, 30, tzinfo=UTC),
                Precision.TIME,
            ),
            (
                "2024-03-05T10:30:00",
                datetime(2024, 3, 5, 10, 30, tzinfo=UTC),
                Precision.TIME,
            ),
        ],
    )
    def test_parse_instant_reports_the_precision_it_was_written_in(
        self, raw: str, when: datetime, precision: Precision
    ) -> None:
        """A partial date keeps the unit the text stated."""
        assert parse_instant(raw) == (when, precision)

    def test_parse_instant_rejects_non_dates(self) -> None:
        """A phrase that is not a date is an error."""
        with pytest.raises(ValueError):
            parse_instant("last spring")

    def test_stated_month_keeps_month_precision(self) -> None:
        """A stated month is cut to the first day and keeps month precision."""
        out = resolve(
            valid_from="2024-03",
            expression="March 2024",
            kind=FactKind.EVENT,
        )
        assert (out.valid_from, out.precision, out.basis) == (
            datetime(2024, 3, 1, tzinfo=UTC),
            Precision.MONTH,
            TimeBasis.STATED,
        )
        assert out.problems == []

    def test_state_without_start_defaults_to_reference_time(self) -> None:
        """An actual state with no start begins at the episode time."""
        out = resolve()
        assert (out.valid_from, out.precision, out.basis) == (
            T0,
            Precision.TIME,
            TimeBasis.REFERENCE,
        )

    @pytest.mark.parametrize(
        ("kind", "modality"),
        [
            (FactKind.EVENT, Modality.ACTUAL),
            (FactKind.STATE, Modality.PLANNED),
            (FactKind.STATE, Modality.CONDITIONAL),
            (FactKind.STATE, Modality.DESIRED),
        ],
    )
    def test_no_default_for_events_and_non_actual_modalities(
        self, kind: FactKind, modality: Modality
    ) -> None:
        """Only an actual state gets the reference time."""
        out = resolve(kind=kind, modality=modality)
        assert (out.valid_from, out.basis) == (None, TimeBasis.NONE)

    def test_unparsed_time_is_cleared_with_reason_and_gets_no_default(self) -> None:
        """A bad time is cleared and the fact keeps no time."""
        out = resolve(valid_from="spring", expression="March 2024")
        assert out.valid_from is None
        assert out.basis is TimeBasis.NONE
        assert [r for r, _ in out.problems] == [FactDropReason.TIME_UNPARSED]

    def test_expression_missing_from_episode_clears_times(self) -> None:
        """A time phrase not in the episode clears the resolved times."""
        out = resolve(valid_from="2024-03", expression="in April 2031")
        assert out.valid_from is None
        assert out.time_expression is None
        assert out.basis is TimeBasis.NONE
        assert FactDropReason.TIME_EXPRESSION_NOT_FOUND in [r for r, _ in out.problems]

    def test_times_without_an_expression_are_cleared(self) -> None:
        """A time with no phrase cannot be grounded."""
        out = resolve(valid_from="2024-03")
        assert out.valid_from is None
        assert [r for r, _ in out.problems] == [
            FactDropReason.TIME_EXPRESSION_NOT_FOUND
        ]

    def test_expression_without_resolved_times_is_a_problem(self) -> None:
        """A phrase with no time leaves the fact undated."""
        out = resolve(expression="March 2024")
        assert out.valid_from is None
        assert [r for r, _ in out.problems] == [FactDropReason.TIME_UNPARSED]

    def test_end_before_start_clears_both(self) -> None:
        """An end before its start is not a window."""
        out = resolve(valid_from="2024-05", valid_to="2024-03", expression="March 2024")
        assert (out.valid_from, out.valid_to, out.basis) == (
            None,
            None,
            TimeBasis.NONE,
        )
        assert [r for r, _ in out.problems] == [FactDropReason.TIME_ORDER]

    def test_order_is_compared_at_the_coarser_precision(self) -> None:
        """A year and a month in the same year do not conflict."""
        out = resolve(valid_from="2024", valid_to="2024-03", expression="March 2024")
        assert out.problems == []
        assert out.precision is Precision.YEAR
        assert out.valid_to == datetime(2024, 1, 1, tzinfo=UTC)

    def test_event_interval_drops_the_end_only(self) -> None:
        """An event is a point, so its end is cleared."""
        out = resolve(
            kind=FactKind.EVENT,
            valid_from="2024-03",
            valid_to="2024-04",
            expression="March 2024",
        )
        assert out.valid_to is None
        assert out.valid_from == datetime(2024, 3, 1, tzinfo=UTC)
        assert [r for r, _ in out.problems] == [FactDropReason.EVENT_INTERVAL]

    def test_state_with_only_an_end_does_not_invent_a_start(self) -> None:
        """An end without a start stays open at the start."""
        out = resolve(valid_to="2024-03", expression="March 2024")
        assert out.valid_from is None
        assert out.valid_to == datetime(2024, 3, 1, tzinfo=UTC)
        assert out.basis is TimeBasis.STATED
