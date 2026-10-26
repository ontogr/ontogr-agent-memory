"""Shared typed-error handling for retrieval lane searches."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ontogr_mem.errors import StoreError
from ontogr_mem.resolution.errors import ResolutionError
from ontogr_mem.retrieval.errors import KeywordSearchError
from ontogr_mem.retrieval.models import LaneRecord, LaneStatus
from ontogr_mem.temporal.errors import TemporalError
from ontogr_mem.vectorstore.errors import VectorStoreError


T = TypeVar("T")
_LANE_ERRORS = (
    VectorStoreError,
    KeywordSearchError,
    TemporalError,
    ResolutionError,
    StoreError,
)


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[T]):
    """The value of one search, or its typed failure."""

    value: T | None = None
    error: Exception | None = None


async def run_search(operation: Callable[[], Awaitable[T]]) -> SearchResult[T]:
    """Return a typed port failure while letting caller errors propagate."""
    try:
        return SearchResult(value=await operation())
    except _LANE_ERRORS as exc:
        if isinstance(exc, ValueError):
            raise
        return SearchResult(error=exc)


def lane_record(
    name: str,
    found: bool,
    failures: list[tuple[str, Exception]],
    *,
    search_count: int,
) -> LaneRecord:
    """Build a lane status and concise diagnostics from its failed searches."""
    failed = tuple(search for search, _ in failures)
    if failures and len(failures) == search_count:
        status = LaneStatus.FAILED
    elif not found:
        status = LaneStatus.EMPTY
    else:
        status = LaneStatus.OK
    details = [
        f"{search}: {type(error).__name__}: {error}" for search, error in failures
    ]
    detail = "; ".join(details)
    return LaneRecord(name, status, detail, failed)
