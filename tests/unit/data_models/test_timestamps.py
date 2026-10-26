"""Tests for stored-time conversion."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from ontogr_mem.data_models import OPEN_MICROS, from_micros, to_micros


class TestTimestamps:
    """Behavior of the microsecond time helpers."""

    def test_epoch_plus_one_second(self):
        """One second after the epoch is 1,000,000 microseconds."""
        assert to_micros(datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)) == 1_000_000

    def test_round_trips_microseconds(self):
        """A datetime with microseconds survives a round trip exactly."""
        moment = datetime(2026, 10, 21, 12, 30, 45, 123457, tzinfo=UTC)

        assert from_micros(to_micros(moment)) == moment

    def test_open_end_is_two_to_the_sixty_second(self):
        """None stands for an open end on both sides of the conversion."""
        assert OPEN_MICROS == 2**62
        assert to_micros(None) == OPEN_MICROS
        assert from_micros(OPEN_MICROS) is None

    def test_rejects_naive_datetime(self):
        """A datetime without a timezone is refused."""
        with pytest.raises(ValueError, match="timezone"):
            to_micros(datetime(2026, 1, 1))

    def test_converts_other_zones_to_the_same_instant(self):
        """A non-UTC aware datetime maps to the same stored value as its UTC form."""
        plus_two = timezone(timedelta(hours=2))

        assert to_micros(datetime(2026, 1, 1, 2, tzinfo=plus_two)) == to_micros(
            datetime(2026, 1, 1, tzinfo=UTC)
        )

    def test_rejects_values_beyond_year_9999(self):
        """A stored value past the datetime range raises instead of wrapping."""
        with pytest.raises(ValueError, match="representable"):
            from_micros(OPEN_MICROS - 1)
