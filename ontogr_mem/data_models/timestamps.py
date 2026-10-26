"""Conversion between datetimes and stored integer microseconds."""

from datetime import UTC, datetime, timedelta
from typing import Final


EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND: Final = timedelta(microseconds=1)

OPEN_MICROS: Final = 2**62
"""Stored value for an open end of a time range."""


def require_utc(value: datetime) -> datetime:
    """Return the datetime in UTC.

    Args:
        value: A timezone-aware datetime.

    Returns:
        The same instant with the UTC timezone.

    Raises:
        ValueError: If the datetime has no timezone.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        msg = "datetime must have a timezone"
        raise ValueError(msg)
    return value.astimezone(UTC)


def to_micros(value: datetime | None) -> int:
    """Convert a datetime to integer UTC microseconds since the epoch.

    Args:
        value: A timezone-aware datetime, or None for an open end.

    Returns:
        Microseconds since the epoch, or `OPEN_MICROS` for None.

    Raises:
        ValueError: If the datetime has no timezone.
    """
    if value is None:
        return OPEN_MICROS
    return (require_utc(value) - EPOCH) // _MICROSECOND


def from_micros(value: int) -> datetime | None:
    """Convert stored microseconds to a UTC datetime.

    Args:
        value: Microseconds since the epoch, or `OPEN_MICROS`.

    Returns:
        A UTC datetime, or None for `OPEN_MICROS`.

    Raises:
        ValueError: If the value is not representable as a datetime.
    """
    if value == OPEN_MICROS:
        return None
    try:
        return EPOCH + timedelta(microseconds=value)
    except OverflowError as exc:
        msg = f"{value} microseconds is not representable as a datetime"
        raise ValueError(msg) from exc
