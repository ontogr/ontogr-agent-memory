"""Table tests for precision spans, UTC rules and the microsecond form."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.timestamps import OPEN_MICROS, from_micros, to_micros
from ontogr_mem.temporal.timeutil import (
    coarser,
    require_utc,
    same_span,
    span_end,
    truncate,
)


MOMENT = datetime(2023, 7, 14, 13, 45, 30, 123456, tzinfo=UTC)


class TestTimeUtil:
    """Behavior of the time helpers used by reconciliation."""

    @pytest.mark.parametrize(
        ("precision", "expected"),
        [
            (Precision.YEAR, datetime(2023, 1, 1, tzinfo=UTC)),
            (Precision.MONTH, datetime(2023, 7, 1, tzinfo=UTC)),
            (Precision.DAY, datetime(2023, 7, 14, tzinfo=UTC)),
            (Precision.TIME, MOMENT),
        ],
    )
    def test_truncate_cuts_to_the_start_of_the_period(self, precision, expected):
        """Each precision keeps only the units it states."""
        assert truncate(MOMENT, precision) == expected

    @pytest.mark.parametrize(
        ("start", "precision", "expected"),
        [
            (
                datetime(2023, 7, 14, tzinfo=UTC),
                Precision.YEAR,
                datetime(2024, 1, 1, tzinfo=UTC),
            ),
            (
                datetime(2023, 12, 14, tzinfo=UTC),
                Precision.MONTH,
                datetime(2024, 1, 1, tzinfo=UTC),
            ),
            (
                datetime(2023, 7, 14, tzinfo=UTC),
                Precision.DAY,
                datetime(2023, 7, 15, tzinfo=UTC),
            ),
            (MOMENT, Precision.TIME, MOMENT),
        ],
    )
    def test_span_end_is_the_first_instant_after_the_period(
        self, start, precision, expected
    ):
        """A stated end covers its whole period, so it ends at the next one."""
        assert span_end(start, precision) == expected

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (Precision.YEAR, Precision.MONTH, Precision.YEAR),
            (Precision.DAY, Precision.TIME, Precision.DAY),
            (None, Precision.MONTH, Precision.MONTH),
            (None, None, Precision.TIME),
        ],
    )
    def test_coarser_returns_the_less_exact_unit(self, a, b, expected):
        """An unknown precision counts as exact."""
        assert coarser(a, b) is expected

    @pytest.mark.parametrize(
        ("a", "a_precision", "b", "b_precision", "expected"),
        [
            (
                datetime(2023, 1, 1, tzinfo=UTC),
                Precision.YEAR,
                datetime(2023, 11, 5, tzinfo=UTC),
                Precision.MONTH,
                True,
            ),
            (
                datetime(2023, 1, 1, tzinfo=UTC),
                Precision.YEAR,
                datetime(2024, 1, 1, tzinfo=UTC),
                Precision.YEAR,
                False,
            ),
            (
                datetime(2023, 4, 1, tzinfo=UTC),
                Precision.DAY,
                datetime(2023, 4, 1, 9, 30, tzinfo=UTC),
                Precision.TIME,
                True,
            ),
        ],
    )
    def test_same_span_compares_at_the_coarser_precision(
        self, a, a_precision, b, b_precision, expected
    ):
        """Two starts in one period cannot be ordered against each other."""
        assert same_span(a, a_precision, b, b_precision) is expected

    def test_micros_round_trip_and_open_sentinel(self):
        """The stored form is exact and None is the open sentinel."""
        assert to_micros(datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)) == 1_000_000
        assert to_micros(None) == OPEN_MICROS == 2**62
        assert from_micros(to_micros(MOMENT)) == MOMENT
        assert from_micros(OPEN_MICROS) is None

    def test_naive_datetime_is_rejected(self):
        """A datetime without a timezone would put a fact on the wrong day."""
        with pytest.raises(ValueError, match="timezone"):
            require_utc(datetime(2023, 7, 14, 13, 45))

    def test_a_period_past_year_9999_is_rejected(self):
        """A year-level end at the last year has no representable next instant."""
        with pytest.raises(ValueError):
            span_end(datetime(9999, 12, 31, tzinfo=UTC), Precision.YEAR)
