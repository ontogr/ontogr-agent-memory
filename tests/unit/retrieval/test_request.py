"""Tests for the retrieval request and the temporal query it maps to."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.retrieval import RetrievalError, RetrievalRequest
from ontogr_mem.temporal.query import TemporalQuery


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
"""The present time every request in this module uses."""


def make_request(**overrides: object) -> RetrievalRequest:
    """Build a valid request and apply the given field overrides."""
    fields: dict[str, object] = {"query": "where does Ana work", "now": NOW}
    fields.update(overrides)
    return RetrievalRequest(**fields)


class TestRetrievalRequest:
    """Validation of the fields of a retrieval request."""

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"query": ""}, id="blank-query"),
            pytest.param({"query": "   "}, id="whitespace-query"),
            pytest.param({"now": datetime(2026, 10, 23, 12, 0)}, id="naive-now"),
            pytest.param({"valid_at": datetime(2020, 1, 1)}, id="naive-valid-at"),
            pytest.param({"known_at": datetime(2020, 1, 1)}, id="naive-known-at"),
            pytest.param(
                {
                    "overlaps": (
                        datetime(2020, 1, 1, tzinfo=UTC),
                        datetime(2023, 1, 1),
                    )
                },
                id="naive-overlaps-end",
            ),
            pytest.param(
                {
                    "overlaps": (
                        datetime(2020, 1, 1),
                        datetime(2023, 1, 1, tzinfo=UTC),
                    )
                },
                id="naive-overlaps-start",
            ),
            pytest.param(
                {
                    "overlaps": (
                        datetime(2023, 1, 1, tzinfo=UTC),
                        datetime(2023, 1, 1, tzinfo=UTC),
                    )
                },
                id="empty-overlaps",
            ),
            pytest.param(
                {
                    "overlaps": (
                        datetime(2025, 1, 1, tzinfo=UTC),
                        datetime(2023, 1, 1, tzinfo=UTC),
                    )
                },
                id="reversed-overlaps",
            ),
            pytest.param(
                {
                    "valid_at": datetime(2020, 1, 1, tzinfo=UTC),
                    "overlaps": (
                        datetime(2020, 1, 1, tzinfo=UTC),
                        datetime(2023, 1, 1, tzinfo=UTC),
                    ),
                },
                id="valid-at-with-overlaps",
            ),
            pytest.param({"limit": 0}, id="limit-below-one"),
            pytest.param({"limit": 101}, id="limit-above-cap"),
            pytest.param({"budget_tokens": 0}, id="budget-below-one"),
            pytest.param({"graph_depth": 0}, id="depth-below-one"),
            pytest.param({"graph_depth": 4}, id="depth-above-cap"),
            pytest.param({"disputed": "ignore"}, id="unknown-disputed-mode"),
            pytest.param(
                {"mentions": ("a", "b", "c", "d", "e", "f", "g", "h", "i")},
                id="too-many-mentions",
            ),
            pytest.param({"mentions": ("Ana", "  ")}, id="blank-mention"),
        ],
    )
    def test_rejects_an_invalid_request(self, overrides: dict[str, object]) -> None:
        """Every invalid field raises the retrieval error."""
        with pytest.raises(RetrievalError):
            make_request(**overrides)

    def test_accepts_a_request_with_the_default_fields(self) -> None:
        """A request with only a query and a present time is valid."""
        request = make_request()

        assert request.query == "where does Ana work"
        assert request.now == NOW
        assert request.limit == 10
        assert request.budget_tokens == 2000
        assert request.include_history is False
        assert request.disputed == "flag"
        assert request.mentions == ()
        assert request.graph_depth == 1

    def test_maps_the_time_filters_to_a_temporal_query(self) -> None:
        """The request keeps its valid time, system time and history flag."""
        request = make_request(
            valid_at=datetime(2020, 1, 1, tzinfo=UTC),
            known_at=datetime(2021, 1, 1, tzinfo=UTC),
            include_history=True,
        )

        assert request.to_temporal_query() == TemporalQuery(
            now=NOW,
            valid_at=datetime(2020, 1, 1, tzinfo=UTC),
            known_at=datetime(2021, 1, 1, tzinfo=UTC),
            include_history=True,
        )

    def test_maps_the_overlap_window_to_a_temporal_query(self) -> None:
        """An overlap window reaches the temporal query unchanged."""
        window = (datetime(2020, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC))
        request = make_request(overlaps=window)

        assert request.to_temporal_query() == TemporalQuery(now=NOW, overlaps=window)

    def test_maps_a_request_without_time_filters_to_the_present(self) -> None:
        """A request with no time filter asks for what is true now."""
        request = make_request()

        assert request.to_temporal_query() == TemporalQuery(now=NOW)
