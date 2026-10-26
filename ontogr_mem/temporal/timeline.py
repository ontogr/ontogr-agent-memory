"""Types for one slot's timeline: its inputs and the results it returns.

A slot's timeline is derived from its observations alone, so it never depends
on the order the observations arrived in. This module holds the types that
describe it: the directives the contradiction handler supplies, the ordered
events the walk reads, and the runs, conflicts and leftovers the walk returns.

"Value" below means the side of the fact that varies inside its slot: the
object, or the subject of an inverse slot.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ontogr_mem.data_models.fact_versions import FactStatus, SupersessionKind
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.temporal.classify import ConflictClass, Disposition, RelationToTime


class ConflictKind(StrEnum):
    """Inconsistencies reported instead of being guessed away."""

    EQUAL_START = "equal_start"
    NO_DATES = "no_dates"
    PAIR = "pair"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class Conflict:
    """An inconsistency that reconciliation reports but never resolves.

    Ties and undated statements are found by the walk. Pair conflicts come from
    the contradiction handler, and carry the classification fields.

    Attributes:
        conflict_id: Deterministic id of the finding.
        slot_key: Slot the finding belongs to.
        kind: What kind of inconsistency was found.
        obs_ids: Observations that take part in it, sorted.
        episode_ids: Episodes those observations came from, sorted.
        reason: Plain description for a reader.
        cls: Kind of disagreement, for pair conflicts.
        relation_to_time: Layout of the two statements, for pair conflicts.
        disposition: What is done about it, for pair conflicts.
        fact_ids: The two facts in dispute, for pair conflicts.
        pair_key: Order-independent key of the two facts, for pair conflicts.
    """

    conflict_id: str
    slot_key: str
    kind: ConflictKind
    obs_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    reason: str
    cls: ConflictClass | None = None
    relation_to_time: RelationToTime | None = None
    disposition: Disposition | None = None
    fact_ids: tuple[str, ...] = ()
    pair_key: str | None = None


@dataclass(frozen=True, slots=True)
class Run:
    """One derived interval of one value, with the observations behind it.

    Attributes:
        value_id: The side of the fact that varies inside the slot.
        first_obs_id: Observation that opened the run.
        obs_ids: Every supporting observation id, sorted.
        valid_from: Start of the interval.
        valid_from_precision: Coarsest unit stated for the start.
        valid_to: End of the interval, or ``None`` while open.
        valid_to_precision: Coarsest unit stated for the end.
        close_kind: Why the run was closed, or ``None``.
        closed_by: Observation id that closed the run, or ``None``.
        last_confirmed_at: Newest observation time that restated the run.
    """

    value_id: str
    first_obs_id: str
    obs_ids: tuple[str, ...]
    valid_from: datetime
    valid_from_precision: Precision | None
    valid_to: datetime | None
    valid_to_precision: Precision | None
    close_kind: SupersessionKind | None
    closed_by: str | None
    last_confirmed_at: datetime


@dataclass(frozen=True, slots=True)
class Derived:
    """What ``derive`` returns for one slot.

    Attributes:
        runs: Derived intervals, in the order they were opened.
        conflicts: Ties and undated statements that cannot be ordered.
        unmatched_ends: Stated ends that closed nothing, sorted.
        orphans: Undated assertions that attached to nothing, sorted.
        unmatched_closes: Judged closes whose run was not open, sorted.
        shadow: Runs of excluded observations, with the status each takes.
    """

    runs: tuple[Run, ...]
    conflicts: tuple[Conflict, ...]
    unmatched_ends: tuple[str, ...]
    orphans: tuple[str, ...]
    unmatched_closes: tuple[str, ...] = ()
    shadow: tuple[tuple[Run, FactStatus], ...] = ()


@dataclass(frozen=True, slots=True)
class Close:
    """Close the run that holds ``anchor_obs_id`` at ``at``.

    The close comes from a verdict about a fact in another slot. It is computed
    on each reconcile and never stored as an observation.

    Attributes:
        anchor_obs_id: Observation that identifies the run to close.
        at: Instant the run ends.
        at_precision: Precision of ``at``.
        cause_id: Id of the verdict that caused the close.
    """

    anchor_obs_id: str
    at: datetime
    at_precision: Precision | None
    cause_id: str


@dataclass(frozen=True, slots=True)
class SlotDirectives:
    """What the contradiction handler tells ``derive`` about one slot.

    Attributes:
        excluded: Observation ids dropped from the walk, with the status their
            fact takes.
        no_cut: Groups of observation ids that do not cut each other.
        closes: Judged closes that come from another slot.
    """

    excluded: Mapping[str, FactStatus] = field(default_factory=dict)
    no_cut: tuple[frozenset[str], ...] = ()
    closes: tuple[Close, ...] = ()


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """One event of the slot's ordered timeline.

    Attributes:
        at: Instant the event takes place.
        rank: Tie-break rank for closes, assertions, and stated ends.
        key: Tie-break key: the observation id, or the close's cause and anchor.
        event: What happens: an observation or a judged close.
    """

    at: datetime
    rank: int
    key: str
    event: Observation | Close


def build_timeline(
    observations: Sequence[Observation], closes: Sequence[Close]
) -> list[TimelineEvent]:
    """Return the slot's events in the order the walk reads them.

    Events are ordered by time. At the same instant, judged closes run first,
    assertions run next, and stated ends run last. This lets a stated end close
    an assertion that starts at the same instant.

    Args:
        observations: The slot's dated observations.
        closes: Judged closes that come from another slot.

    Returns:
        The events in order. An observation without a start is left out.
    """
    events: list[TimelineEvent] = []
    for observation in observations:
        start = observation.valid_from
        if start is None:
            continue
        rank = 2 if observation.kind is ObservationKind.STATED_END else 1
        events.append(TimelineEvent(start, rank, observation.obs_id, observation))
    for close in closes:
        events.append(
            TimelineEvent(close.at, 0, f"{close.cause_id}:{close.anchor_obs_id}", close)
        )
    events.sort(key=lambda item: (item.at, item.rank, item.key))
    return events
