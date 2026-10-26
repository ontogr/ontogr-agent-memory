"""Tests for the reciprocal rank fusion of the retrieval lanes."""

import pytest

from ontogr_mem.retrieval import RetrievalSettings
from ontogr_mem.retrieval.fusion import fuse


SETTINGS = RetrievalSettings()


class TestFuse:
    """Fusion of the ranked lists of the lanes."""

    def test_scores_one_list_by_its_rank(self) -> None:
        """One lane keeps its order, and the best key scores 1.0."""
        result = fuse([["a", "b", "c"]], limit=10, settings=SETTINGS)

        assert [key for key, _ in result] == ["a", "b", "c"]
        assert result[0][1] == 1.0
        assert result[1][1] == pytest.approx(0.9761904761904762)
        assert result[2][1] == pytest.approx(0.9534883720930233)

    def test_raises_a_key_two_lanes_hold_above_a_key_one_lane_holds(self) -> None:
        """A key two lanes hold scores higher than a key only one lane holds."""
        result = fuse([["a", "b"], ["b", "c"]], limit=10, settings=SETTINGS)

        assert [key for key, _ in result] == ["b", "a", "c"]
        assert result[0][1] == 1.0
        assert result[1][1] == pytest.approx(0.5060240963855421)
        assert result[2][1] == pytest.approx(0.4939759036144578)

    def test_counts_a_key_in_three_lanes_once_per_lane(self) -> None:
        """A key every lane holds wins the fusion."""
        result = fuse([["a"], ["a"], ["a"]], limit=10, settings=SETTINGS)

        assert result == [("a", 1.0)]

    def test_uses_the_floor_of_the_fusion_constant(self) -> None:
        """A small limit clamps the fusion constant to its floor."""
        result = fuse([["a", "b"]], limit=1, settings=SETTINGS)

        assert result[0][1] == 1.0
        assert result[1][1] == pytest.approx(0.96875)

    def test_uses_the_ceiling_of_the_fusion_constant(self) -> None:
        """A large limit clamps the fusion constant to its ceiling."""
        result = fuse([["a", "b"]], limit=50, settings=SETTINGS)

        assert result[0][1] == 1.0
        assert result[1][1] == pytest.approx(0.9838709677419355)

    def test_uses_a_fusion_constant_between_the_two_clamps(self) -> None:
        """A middle limit leaves the fusion constant between the clamps."""
        result = fuse([["a", "b"]], limit=15, settings=SETTINGS)

        assert result[0][1] == 1.0
        assert result[1][1] == pytest.approx(0.9807692307692307)

    def test_counts_a_duplicate_inside_one_list_once(self) -> None:
        """A repeated key takes one position in its lane, not two."""
        result = fuse([["a", "a", "b"]], limit=10, settings=SETTINGS)

        assert [key for key, _ in result] == ["a", "b"]
        assert result[1][1] == pytest.approx(0.9761904761904762)

    def test_breaks_a_tie_with_the_smaller_key(self) -> None:
        """Two keys with the same score keep the order of their keys."""
        result = fuse([["b"], ["a"]], limit=10, settings=SETTINGS)

        assert [key for key, _ in result] == ["a", "b"]

    def test_gives_the_best_result_exactly_one(self) -> None:
        """The top score is 1.0 and not a rounded value."""
        result = fuse([["a", "b"], ["b"]], limit=10, settings=SETTINGS)

        assert result[0][1] == 1.0

    def test_returns_nothing_for_an_empty_input(self) -> None:
        """No lane list gives no result."""
        assert fuse([], limit=10, settings=SETTINGS) == []
        assert fuse([[], []], limit=10, settings=SETTINGS) == []
