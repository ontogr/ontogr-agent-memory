"""Fetch sizing: how many neighbours a search asks the index for."""

import math

from ontogr_mem.vectorstore.models import SearchFilters


def fetch_size(
    k: int,
    filters: SearchFilters,
    min_score: float | None,
    over_fetch: float,
    *,
    backend_filters_rows: bool = False,
) -> int:
    """Choose the fetch size for a search of `k` hits.

    Args:
        k: The number of hits the caller asked for.
        filters: The filters of the search.
        min_score: The lowest score the caller will keep.
        over_fetch: Factor for extra neighbors when filtering can remove hits.
        backend_filters_rows: True when the backend drops rows after the
            index returned them, so the fetch must be wider than `k`.

    Returns:
        The number of neighbors to request from the index.
    """
    shrinks = (
        filters.narrows
        or filters.temporal is not None
        or min_score is not None
        or backend_filters_rows
    )
    return math.ceil(k * over_fetch) if shrinks else k
