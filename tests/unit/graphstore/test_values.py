"""Tests for property cleaning."""

from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from ontogr_mem.graphstore.errors import ValidationError
from ontogr_mem.graphstore.values import clean_properties, micros


class TestMicros:
    """Behavior of micros."""

    def test_converts_to_utc_epoch_microseconds(self):
        """A datetime becomes an exact integer."""
        moment = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC)

        assert micros(moment) == 1767323045678901

    def test_normalises_offsets_to_utc(self):
        """The same instant gives the same value in any zone."""
        minus_five = timezone(timedelta(hours=-5))

        assert micros(datetime(2026, 1, 1, tzinfo=minus_five)) == micros(
            datetime(2026, 1, 1, 5, tzinfo=UTC)
        )

    def test_rejects_naive_datetime(self):
        """A datetime without a timezone is refused."""
        with pytest.raises(ValidationError):
            micros(datetime(2026, 1, 1))


class TestCleanProperties:
    """Behavior of clean_properties."""

    def test_converts_and_omits_values(self):
        """Each supported type is converted as documented."""
        cleaned = clean_properties(
            {
                "omitted": None,
                "text": "x\x00y\x07z\tw\n",
                "mapping": {"k": [1, 2]},
                "numbers": [1, 2, 3],
                "uuid": UUID("12345678-1234-5678-1234-567812345678"),
                "time": datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
                "flag": True,
            }
        )

        assert cleaned == {
            "text": "xyz\tw\n",
            "mapping_json": '{"k": [1, 2]}',
            "numbers": [1, 2, 3],
            "uuid": "12345678-1234-5678-1234-567812345678",
            "time": 1_000_000,
            "flag": True,
        }

    @pytest.mark.parametrize(
        "props",
        [
            {"items": [1, {"a": 1}]},
            {"a b": 1},
            {"x": float("nan")},
            {"x": object()},
            {"when": datetime(2026, 1, 1)},
        ],
    )
    def test_rejects_unsupported_input(self, props):
        """Unsafe keys and unsupported values raise instead of being dropped."""
        with pytest.raises(ValidationError):
            clean_properties(props)

    @pytest.mark.parametrize(
        "props",
        [
            {"nested": {"x": float("nan")}},
            {"nested": {"x": float("inf")}},
            {"nested": {"x": {"y": float("nan")}}},
        ],
    )
    def test_rejects_non_finite_float_inside_mapping(self, props):
        """A NaN or infinity inside a mapping raises and names the property key."""
        with pytest.raises(ValidationError, match="'nested'"):
            clean_properties(props)
