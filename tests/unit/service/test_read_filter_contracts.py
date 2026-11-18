"""Validate bounded type filters and paged read request contracts."""

import pytest
from pydantic import ValidationError

from ontogr_mem.service.contracts import ListEntities, SearchMemory
from ontogr_mem.service.limits import MAX_FILTER_LABELS


class TestReadFilterContracts:
    """Validate read request defaults and bounds."""

    def test_search_defaults_to_actual_and_empty_type_filters(self) -> None:
        """Default search to actual facts with no type filters."""
        request = SearchMemory(query="query")

        assert request.modalities == ["actual"]
        assert request.entity_types == []
        assert request.relation_types == []

    def test_search_rejects_too_many_filter_labels(self) -> None:
        """Reject search requests above the filter label limit."""
        with pytest.raises(ValidationError):
            SearchMemory(
                query="query", entity_types=["Person"] * (MAX_FILTER_LABELS + 1)
            )

    def test_list_entities_enforces_page_bounds(self) -> None:
        """Reject entity list pages above the configured bound."""
        with pytest.raises(ValidationError):
            ListEntities(limit=MAX_FILTER_LABELS + 100)
