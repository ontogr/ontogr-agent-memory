"""Shared row-level search filter semantics."""

from collections.abc import Mapping

from ontogr_mem.data_models import OPEN_MICROS, FactStatus
from ontogr_mem.vectorstore.errors import CapabilityError
from ontogr_mem.vectorstore.models import SearchFilters, TemporalFilter, VectorKind


def reject_past_belief(filters: SearchFilters) -> None:
    """Refuse filters that ask what was believed at an instant in the past.

    A stored validity mirror holds the current version of a fact, so it cannot
    say what was believed then. Only the graph can answer that. A backend returns
    rows and never the belief behind them.

    Args:
        filters: The filters of the search.

    Raises:
        CapabilityError: If the filters ask for `known_at`.
    """
    temporal = filters.temporal
    if temporal is not None and temporal.known_at_us is not None:
        msg = "known_at needs the graph re-check; a backend cannot answer it"
        raise CapabilityError(msg)


def matches(
    kind: VectorKind, properties: Mapping[str, object], filters: SearchFilters
) -> bool:
    """Return whether a row satisfies every filter that applies to its kind.

    Args:
        kind: The kind of vector the row holds.
        properties: The properties of the row the filters read.
        filters: The filters to apply.

    Returns:
        True when the row passes every filter.

    Raises:
        CapabilityError: If the filters ask for past belief.
    """
    reject_past_belief(filters)
    filters.for_kind(kind)
    if kind is VectorKind.NODE:
        return (
            filters.entity_types is None
            or properties.get("entity_type") in filters.entity_types
        ) and (filters.statuses is None or properties.get("status") in filters.statuses)
    if kind is not VectorKind.EDGE:
        return True

    if any(
        selected is not None and properties.get(name) not in selected
        for name, selected in (
            ("relation_type_id", filters.relation_type_ids),
            ("subject_id", filters.subject_ids),
            ("object_id", filters.object_ids),
        )
    ):
        return False

    temporal = filters.temporal or TemporalFilter()
    if temporal.history:
        return True
    current = (
        properties.get("status") != FactStatus.ACTIVE.value
        or properties.get("expired_at") != OPEN_MICROS
    )
    start = properties.get("valid_from")
    end = properties.get("valid_to")
    return not current and (
        temporal.valid_at_us is None
        or (
            isinstance(start, int)
            and isinstance(end, int)
            and start <= temporal.valid_at_us < end
        )
    )
