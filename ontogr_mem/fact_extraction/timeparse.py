"""Time handling: partial dates with explicit precision, and the time rules."""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ontogr_mem.data_models.facts import (
    FactKind,
    Modality,
    Precision,
    TimeBasis,
)
from ontogr_mem.fact_extraction.models import FactDropReason
from ontogr_mem.fact_extraction.text import locate


# YYYY, YYYY-MM or YYYY-MM-DD; the groups tell us how much was written.
_PARTIAL = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
# A full timestamp must carry a time part after "T" or a space.
_HAS_TIME = re.compile(r"[T ]\d")


def parse_instant(value: str) -> tuple[datetime, Precision]:
    """Parse a partial date or a full ISO timestamp.

    A partial date maps to the start of its period and reports the precision it
    was written in, so "2024" is never confused with January 1st 00:00. Naive
    timestamps are taken as UTC.

    Args:
        value: The text to parse.

    Returns:
        The instant and the precision the text was written in.

    Raises:
        ValueError: If the text is not a date or timestamp.
    """
    text = value.strip()
    match = _PARTIAL.match(text)
    if match:
        year, month, day = match.groups()
        when = datetime(int(year), int(month or 1), int(day or 1), tzinfo=UTC)
        return (
            when,
            Precision.DAY if day else Precision.MONTH if month else Precision.YEAR,
        )
    if not _HAS_TIME.search(text):
        msg = f"not a date or timestamp: {value!r}"
        raise ValueError(msg)
    when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (when if when.tzinfo else when.replace(tzinfo=UTC)), Precision.TIME


def truncate(when: datetime, precision: Precision) -> datetime:
    """Cut `when` down to the start of its `precision` period."""
    utc = when.astimezone(UTC)
    if precision is Precision.YEAR:
        return utc.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if precision is Precision.MONTH:
        return utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if precision is Precision.DAY:
        return utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return utc


def coarsest(*items: Precision | None) -> Precision | None:
    """Return the coarsest of the given precisions, ignoring None."""
    found = [p for p in items if p is not None]
    return min(found, key=lambda p: p.rank) if found else None


@dataclass(slots=True)
class ResolvedTime:
    """The time fields of one fact plus every problem found on the way."""

    valid_from: datetime | None = None
    valid_to: datetime | None = None
    precision: Precision | None = None
    time_expression: str | None = None
    basis: TimeBasis = TimeBasis.NONE
    problems: list[tuple[FactDropReason, str]] = field(default_factory=list)


def resolve_time(
    *,
    valid_from: str | None,
    valid_to: str | None,
    declared: Precision | None,
    expression: str | None,
    kind: FactKind,
    modality: Modality,
    episode_text: str,
    reference: datetime,
) -> ResolvedTime:
    """Apply the time rules to what the model proposed.

    A bad time never loses the fact: its time fields are cleared and every
    reason is returned in `problems`. A cleared fact gets no reference-time
    default. A state with only an end keeps no start.

    Args:
        valid_from: Proposed start, as the model wrote it.
        valid_to: Proposed end, as the model wrote it.
        declared: Precision the model declared.
        expression: Literal time phrase from the episode.
        kind: State or event.
        modality: Actual, planned, conditional or desired.
        episode_text: The episode text in NFC.
        reference: Episode time used for a present-tense default.

    Returns:
        The resolved time fields and every problem found.
    """
    out = ResolvedTime()
    expr = (expression or "").strip() or None
    # `cleared` is set when a proposed time was unusable; a cleared fact does not
    # get the reference-time default at the end.
    cleared = False

    # A time needs evidence in the episode text.
    if expr is not None and locate(episode_text, expr) is None:
        out.problems.append((FactDropReason.TIME_EXPRESSION_NOT_FOUND, expr))
        expr = None
        cleared = True
    out.time_expression = expr

    # Each bound retains its stated precision for interval comparisons.
    parsed: dict[str, tuple[datetime, Precision]] = {}
    for name, raw in (("valid_from", valid_from), ("valid_to", valid_to)):
        if raw is None or not raw.strip():
            continue
        try:
            parsed[name] = parse_instant(raw)
        except ValueError:
            out.problems.append((FactDropReason.TIME_UNPARSED, f"{name}={raw!r}"))
            cleared = True

    # Times and their evidence phrase must both be usable.
    if parsed and expr is None and not cleared:
        out.problems.append(
            (FactDropReason.TIME_EXPRESSION_NOT_FOUND, "times without time_expression")
        )
        cleared = True
    if not cleared and expr is not None and not parsed:
        out.problems.append((FactDropReason.TIME_UNPARSED, "expression not resolved"))
        cleared = True
    if cleared:
        parsed = {}

    # An event has a point time, so only its end is removed.
    if kind is FactKind.EVENT and "valid_to" in parsed:
        out.problems.append((FactDropReason.EVENT_INTERVAL, "event has a point time"))
        del parsed["valid_to"]

    # Interval comparisons must not imply finer precision than the stated times.
    precision = coarsest(declared, *(p for _, p in parsed.values()))
    if "valid_from" in parsed and "valid_to" in parsed and precision is not None:
        start = truncate(parsed["valid_from"][0], precision)
        end = truncate(parsed["valid_to"][0], precision)
        if end < start:
            out.problems.append((FactDropReason.TIME_ORDER, "valid_to < valid_from"))
            parsed = {}
            cleared = True

    # Only an actual state without unusable times can default to the episode time.
    if parsed and precision is not None:
        out.valid_from = _cut(parsed.get("valid_from"), precision)
        out.valid_to = _cut(parsed.get("valid_to"), precision)
        out.precision = precision
        out.basis = TimeBasis.STATED
    elif not cleared and kind is FactKind.STATE and modality is Modality.ACTUAL:
        out.valid_from = reference
        out.precision = Precision.TIME
        out.basis = TimeBasis.REFERENCE
    return out


def _cut(
    item: tuple[datetime, Precision] | None, precision: Precision
) -> datetime | None:
    """Cut a parsed bound to the fact's precision."""
    return truncate(item[0], precision) if item else None
