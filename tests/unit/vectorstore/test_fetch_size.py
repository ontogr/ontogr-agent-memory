"""Tests for the fetch sizer."""

import pytest

from ontogr_mem.vectorstore import SearchFilters, TemporalFilter
from ontogr_mem.vectorstore.fetch_size import fetch_size


class TestFetchSize:
    """Behavior of `fetch_size`."""

    @pytest.mark.parametrize(
        ("k", "filters", "min_score", "expected"),
        [
            (10, SearchFilters(), None, 10),
            (10, SearchFilters(entity_types=frozenset({1})), None, 20),
            (10, SearchFilters(statuses=frozenset({"active"})), None, 20),
            (10, SearchFilters(relation_type_ids=frozenset({7})), None, 20),
            (10, SearchFilters(subject_ids=frozenset({"e1"})), None, 20),
            (10, SearchFilters(object_ids=frozenset({"e2"})), None, 20),
            (10, SearchFilters(temporal=TemporalFilter()), None, 20),
            (10, SearchFilters(), 0.5, 20),
        ],
        ids=[
            "unfiltered",
            "entity-types",
            "statuses",
            "relation-types",
            "subject-ids",
            "object-ids",
            "temporal",
            "min-score",
        ],
    )
    def test_doubles_the_fetch_for_a_narrowed_or_cut_search(
        self,
        k: int,
        filters: SearchFilters,
        min_score: float | None,
        expected: int,
    ) -> None:
        """Only searches a filter or cutoff can shrink fetch extra neighbours."""
        assert fetch_size(k, filters, min_score, 2.0) == expected

    @pytest.mark.parametrize(
        ("over_fetch", "k", "expected"),
        [(3.0, 4, 12), (2.5, 3, 8), (1.0, 7, 7)],
    )
    def test_uses_the_configured_over_fetch_factor_and_rounds_up(
        self, over_fetch: float, k: int, expected: int
    ) -> None:
        """The factor comes from the config and the result is rounded up."""
        filters = SearchFilters(statuses=frozenset({"active"}))

        assert fetch_size(k, filters, None, over_fetch) == expected

    def test_unfiltered_search_fetches_exactly_k_whatever_the_factor(self) -> None:
        """A search nothing can shrink asks the index for `k` neighbours."""
        assert fetch_size(4, SearchFilters(), None, 3.0) == 4

    def test_a_backend_that_filters_rows_fetches_extra_neighbours(self) -> None:
        """A backend that drops rows after the index needs a wider fetch."""
        size = fetch_size(4, SearchFilters(), None, 2.0, backend_filters_rows=True)

        assert size == 8
