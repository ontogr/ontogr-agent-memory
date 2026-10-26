"""Precision spans and the stored microsecond form.

Domain code holds timezone-aware UTC datetimes and uses ``None`` for an open
end. Integer microseconds and the ``OPEN_MICROS`` sentinel exist only at the
store boundary, so range predicates there are plain numeric comparisons.
"""

from datetime import datetime, timedelta

from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.timestamps import require_utc


def truncate(value: datetime, precision: Precision) -> datetime:
    """Cut a datetime down to the start of its precision period.

    Args:
        value: A timezone-aware datetime.
        precision: Unit to keep.

    Returns:
        The start of the period that contains ``value``, in UTC.

    Raises:
        ValueError: If ``value`` has no timezone.
    """
    utc = require_utc(value)
    if precision is Precision.YEAR:
        return utc.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if precision is Precision.MONTH:
        return utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if precision is Precision.DAY:
        return utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return utc


def span_end(start: datetime, precision: Precision) -> datetime:
    """Return the exclusive end of the period that begins at ``start``.

    "2020" at year precision ends at 2021-01-01. An exact instant (``TIME``) is
    its own end.

    Args:
        start: First instant of the period.
        precision: Unit of the period.

    Returns:
        The first instant after the period, in UTC.

    Raises:
        ValueError: If the result is outside the datetime range.
        OverflowError: If the result is outside the datetime range.
    """
    begin = truncate(start, precision)
    if precision is Precision.YEAR:
        return begin.replace(year=begin.year + 1)
    if precision is Precision.MONTH:
        if begin.month == 12:
            return begin.replace(year=begin.year + 1, month=1)
        return begin.replace(month=begin.month + 1)
    if precision is Precision.DAY:
        return begin + timedelta(days=1)
    return begin


def coarser(a: Precision | None, b: Precision | None) -> Precision:
    """Return the coarser of two precisions; an unknown one counts as exact."""
    left = a or Precision.TIME
    right = b or Precision.TIME
    return left if left.rank <= right.rank else right


def same_span(
    a: datetime,
    a_precision: Precision | None,
    b: datetime,
    b_precision: Precision | None,
) -> bool:
    """Return whether two instants fall in one period at the coarser precision.

    A year-level statement cannot be ordered against a day in the same year, so
    the two tie. Reconciliation reports such starts as a conflict instead of
    guessing an order.

    Args:
        a: First instant.
        a_precision: Precision stated for ``a``.
        b: Second instant.
        b_precision: Precision stated for ``b``.

    Returns:
        True when both instants truncate to the same period.
    """
    unit = coarser(a_precision, b_precision)
    return truncate(a, unit) == truncate(b, unit)
