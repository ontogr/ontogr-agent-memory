"""Literal ranking examples for the retrieval ablation report."""

import pytest

from tests.eval.retrieval_runner import list_metrics, mean_metrics


class TestRetrievalMetrics:
    """Measure first hits, top-five coverage, and omitted empty targets."""

    @pytest.mark.parametrize(
        ("ranked", "expected", "metrics"),
        [
            (["a", "b"], ["a"], (1.0, 1.0, 1.0)),
            (["x", "b", "a"], ["a"], (0.0, 1.0, 1 / 3)),
            (["x", "y"], ["a"], (0.0, 0.0, 0.0)),
            (["x", "a", "b", "c", "d", "e"], ["a", "e"], (0.0, 0.5, 0.5)),
            (["a", "a", "x"], ["a", "b"], (1.0, 0.5, 1.0)),
            ([], ["a"], (0.0, 0.0, 0.0)),
            (["a"], [], None),
        ],
    )
    def test_scores_literal_rankings(self, ranked, expected, metrics):
        """Rank and top-five boundaries have independent literal expectations."""
        assert list_metrics(ranked, expected) == metrics

    def test_averages_exclude_lists_without_expected_ids(self):
        """An unlabelled list does not increase the metric denominator."""
        samples = [
            list_metrics(["x", "a"], ["a", "b"]),
            list_metrics(["b"], ["b"]),
            list_metrics(["x"], []),
        ]
        assert mean_metrics(samples) == {
            "hit_at_1": 0.5,
            "recall_at_5": 0.75,
            "mrr": 0.75,
            "queries": 2,
        }
        assert mean_metrics([None]) == {
            "hit_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "queries": 0,
        }
