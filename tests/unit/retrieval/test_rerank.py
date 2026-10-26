"""Literal reranker blends protect shortlist and missing-score behavior."""

import pytest

from ontogr_mem.retrieval.rerank import blend


class TestBlend:
    """Blend normalized reranker scores with fused ranks."""

    @pytest.mark.parametrize(
        ("scores", "weight", "expected"),
        [
            ({"a": 0.0, "b": 10.0}, 0.5, [("b", 0.75), ("a", 0.5)]),
            ({"a": 0.0, "b": 10.0}, 0.0, [("a", 1.0), ("b", 0.5)]),
            ({"a": 0.0, "b": 10.0}, 1.0, [("b", 1.0), ("a", 0.0)]),
            ({"a": 7.0, "b": 7.0}, 0.5, [("a", 0.5), ("b", 0.25)]),
            ({"a": 7.0}, 0.5, [("a", 0.5), ("b", 0.25)]),
            ({}, 0.5, [("a", 0.5), ("b", 0.25)]),
        ],
    )
    def test_blends_missing_equal_and_opposing_scores(self, scores, weight, expected):
        """Missing and equal scores contribute zero after normalization."""
        assert blend([("a", 1.0), ("b", 0.5)], scores, weight=weight) == expected

    def test_tail_uses_zero_reranker_score(self):
        """The unscored tail is blended too, rather than keeping its full score."""
        assert blend(
            [("a", 1.0), ("b", 0.9), ("c", 0.8)],
            {"a": 0.0, "b": 10.0},
            weight=0.5,
        ) == [("b", 0.95), ("a", 0.5), ("c", 0.4)]

    def test_returns_empty_without_candidates(self):
        """An empty candidate set produces no ranking."""
        assert blend([], {}, weight=0.5) == []
