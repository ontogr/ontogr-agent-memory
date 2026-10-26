"""Tests for vector helpers."""

import math

import pytest

from ontogr_mem.vectorstore import FailReason, InvalidVectorError
from ontogr_mem.vectorstore.vecmath import (
    classify_vector,
    mean_unit,
    to_score,
    to_unit,
)


class TestClassifyVector:
    """Behavior of `classify_vector`."""

    def test_scales_to_unit_length(self) -> None:
        """A valid vector comes back with norm one and no reason."""
        unit, reason, detail = classify_vector([3.0, 4.0], 2)

        assert unit == pytest.approx((0.6, 0.8))
        assert reason is None
        assert detail == ""

    @pytest.mark.parametrize(
        ("raw", "dim", "reason"),
        [
            ([], 2, FailReason.EMPTY_VECTOR),
            ([1.0], 2, FailReason.BAD_DIMENSION),
            ([1.0, 2.0, 3.0], 2, FailReason.BAD_DIMENSION),
            ([math.nan, 1.0], 2, FailReason.NON_FINITE),
            ([math.inf, 1.0], 2, FailReason.NON_FINITE),
            ([1.0, -math.inf], 2, FailReason.NON_FINITE),
            ([0.0, 0.0], 2, FailReason.ZERO_NORM),
        ],
    )
    def test_reports_unusable_vectors(
        self, raw: list[float], dim: int, reason: FailReason
    ) -> None:
        """Each unusable vector gets its own reason and no vector."""
        unit, got, detail = classify_vector(raw, dim)

        assert unit is None
        assert got is reason
        assert detail


class TestToUnit:
    """Behavior of `to_unit`."""

    def test_returns_a_unit_vector(self) -> None:
        """A valid query vector is scaled."""
        assert to_unit([0.0, 5.0], 2) == pytest.approx((0.0, 1.0))

    @pytest.mark.parametrize(
        "raw",
        [[], [1.0], [math.nan, 1.0], [0.0, 0.0]],
        ids=["empty", "wrong-dimension", "nan", "zero-norm"],
    )
    def test_raises_for_an_unusable_vector(self, raw: list[float]) -> None:
        """A query the index cannot compare is refused."""
        with pytest.raises(InvalidVectorError):
            to_unit(raw, 2)


class TestToScore:
    """Behavior of `to_score`."""

    @pytest.mark.parametrize(
        ("cosine", "score"),
        [
            (-1.0, 0.0),
            (-0.5, 0.25),
            (0.0, 0.5),
            (0.5, 0.75),
            (1.0, 1.0),
        ],
    )
    def test_maps_cosine_to_zero_one(self, cosine: float, score: float) -> None:
        """Cosine -1..1 maps to 0..1, higher meaning closer."""
        assert to_score(cosine) == score

    @pytest.mark.parametrize(
        ("cosine", "score"),
        [(1.0000000000000002, 1.0), (-1.0000000000000002, 0.0)],
    )
    def test_clamps_rounding_noise(self, cosine: float, score: float) -> None:
        """A cosine a hair past the range stays in the range."""
        assert to_score(cosine) == score


class TestMeanUnit:
    """Behavior of `mean_unit`."""

    def test_returns_a_single_vector_unchanged(self) -> None:
        """A mean of one unit vector is that vector."""
        assert mean_unit([(0.6, 0.8)]) == pytest.approx((0.6, 0.8))

    def test_rescales_the_average_to_unit_length(self) -> None:
        """The mean of two unit vectors is normalised again."""
        got = mean_unit([(1.0, 0.0), (0.0, 1.0)])

        assert got is not None
        assert got == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
        assert math.hypot(*got) == pytest.approx(1.0)

    def test_weights_the_pieces_by_count(self) -> None:
        """Repeated directions pull the mean toward them."""
        got = mean_unit([(1.0, 0.0), (0.0, 1.0), (1.0, 0.0)])

        assert got == pytest.approx((2 / math.sqrt(5), 1 / math.sqrt(5)))

    def test_returns_none_when_the_vectors_cancel(self) -> None:
        """Opposite unit vectors have no direction."""
        assert mean_unit([(1.0, 0.0), (-1.0, 0.0)]) is None
