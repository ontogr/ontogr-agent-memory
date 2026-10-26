"""The request and result types of the retrieval component.

A request says what the caller asks for. A result says what the retriever
found: the ranked facts, the ranked episodes, the entity cards, one record per
lane and the notes the retriever made on the way.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from ontogr_mem.data_models.timestamps import require_utc
from ontogr_mem.retrieval.errors import RetrievalError
from ontogr_mem.temporal.query import FactView, TemporalQuery


GRAPH_DEPTH_CAP: Final = 3
"""The deepest graph walk a request may ask for."""

LIMIT_CAP: Final = 100
"""The most results a request may ask for."""

MENTION_CAP: Final = 8
"""The most entity mentions a request may carry."""

DISPUTED_MODES: Final = ("flag", "exclude")
"""The ways a request treats a fact in an open dispute."""


class LaneStatus(StrEnum):
    """Outcome of one retrieval lane.

    ``OK`` means the lane ran and found something, ``EMPTY`` that it ran and
    found nothing, ``FAILED`` that every search of the lane failed, and
    ``SKIPPED`` that the lane had no work to do, such as a graph walk without
    seeds.
    """

    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class LaneRecord:
    """What one retrieval lane did.

    Attributes:
        name: Name of the lane.
        status: Outcome of the lane.
        detail: What the lane found or why it degraded.
        failed: Names of the searches of the lane that failed.
    """

    name: str
    status: LaneStatus
    detail: str = ""
    failed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FactResult:
    """A fact version in the result.

    Attributes:
        view: The fact version the reader returned.
        score: Fused score, relative to the best fact of the same result.
        lanes: Names of the lanes that found the fact.
        episode_ids: Episodes whose observations support the fact.
        quote: First evidence quote of the fact.
    """

    view: FactView
    score: float
    lanes: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()
    quote: str = ""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """An episode in the result.

    Attributes:
        episode_id: Id of the episode.
        text: Text of the episode.
        occurred_at: When the episode happened.
        source_id: Source the episode came from.
        score: Fused score, relative to the best episode of the same result.
        lanes: Names of the lanes that found the episode.
    """

    episode_id: str
    text: str
    occurred_at: datetime
    source_id: str
    score: float
    lanes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityCard:
    """An entity in the result.

    Attributes:
        entity_id: Id of the entity.
        name: Canonical name of the entity.
        type_label: Type of the entity, such as ``Person``.
        aliases: Alternate names of the entity.
        summary: Short description of the entity.
        lanes: Names of the lanes that found the entity.
    """

    entity_id: str
    name: str
    type_label: str
    aliases: tuple[str, ...] = ()
    summary: str = ""
    lanes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """A rendered retrieval context and the items it contains.

    Attributes:
        text: Escaped context text, or empty when the frame cannot fit.
        items: Placed item identities in rendered section order.
        dropped: Unplaced items; a dispute group counts as one item.
        tokens_used: Count of text using the supplied token counter.
        budget_tokens: Maximum allowed token count.
        degraded: Failed lane or search names shown in the frame.
    """

    text: str
    items: tuple[tuple[str, str], ...]
    dropped: int
    tokens_used: int
    budget_tokens: int
    degraded: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What the retriever found.

    Attributes:
        facts: The ranked facts.
        episodes: The ranked episodes.
        entities: The entity cards.
        lanes: One record per lane.
        notes: What the retriever reported on the way.
        context: Budgeted text for an answering model, independent of list limits.
    """

    facts: tuple[FactResult, ...] = ()
    episodes: tuple[EpisodeResult, ...] = ()
    entities: tuple[EntityCard, ...] = ()
    lanes: tuple[LaneRecord, ...] = ()
    notes: tuple[str, ...] = ()
    context: ContextBlock = ContextBlock("", (), 0, 0, 0)


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """What the caller asks the retriever for.

    Attributes:
        query: The text of the question.
        now: Present time, used by the default valid-time and system-time
            filters.
        limit: Most results the answer may hold.
        budget_tokens: Token budget the context block may use.
        valid_at: Instant of the world the question is about.
        known_at: Instant of the system belief the question is about.
        overlaps: Window the valid interval of a fact must meet.
        include_history: Return closed and past versions as well.
        disputed: How to treat a fact in an open dispute, either ``flag`` or
            ``exclude``.
        mentions: Entity names the question names.
        graph_depth: Hops the graph lane walks from its seeds.
    """

    query: str
    now: datetime
    limit: int = 10
    budget_tokens: int = 2000
    valid_at: datetime | None = None
    known_at: datetime | None = None
    overlaps: tuple[datetime, datetime] | None = None
    include_history: bool = False
    disputed: str = "flag"
    mentions: tuple[str, ...] = ()
    graph_depth: int = 1

    def __post_init__(self) -> None:
        """Reject a request the retriever cannot serve.

        Raises:
            RetrievalError: If the query is blank, a timestamp has no timezone,
                the time filters conflict, or a bound is out of range.
        """
        if not self.query.strip():
            msg = "query must not be blank"
            raise RetrievalError(msg)
        for name, value in (
            ("now", self.now),
            ("valid_at", self.valid_at),
            ("known_at", self.known_at),
        ):
            if value is not None:
                _require_aware(name, value)
        if self.overlaps is not None:
            _require_aware("overlaps", self.overlaps[0])
            _require_aware("overlaps", self.overlaps[1])
            if self.overlaps[0] >= self.overlaps[1]:
                msg = "overlaps must run from an earlier instant to a later one"
                raise RetrievalError(msg)
        if self.valid_at is not None and self.overlaps is not None:
            msg = "use valid_at or overlaps, not both"
            raise RetrievalError(msg)
        if not 1 <= self.limit <= LIMIT_CAP:
            msg = f"limit must be between 1 and {LIMIT_CAP}"
            raise RetrievalError(msg)
        if self.budget_tokens < 1:
            msg = "budget_tokens must be at least 1"
            raise RetrievalError(msg)
        if not 1 <= self.graph_depth <= GRAPH_DEPTH_CAP:
            msg = f"graph_depth must be between 1 and {GRAPH_DEPTH_CAP}"
            raise RetrievalError(msg)
        if self.disputed not in DISPUTED_MODES:
            msg = f"disputed must be one of {DISPUTED_MODES}"
            raise RetrievalError(msg)
        if len(self.mentions) > MENTION_CAP:
            msg = f"mentions must hold at most {MENTION_CAP} names"
            raise RetrievalError(msg)
        if any(not mention.strip() for mention in self.mentions):
            msg = "mentions must not hold a blank name"
            raise RetrievalError(msg)

    def to_temporal_query(self) -> TemporalQuery:
        """Build the temporal query that matches the time filters of the request.

        Returns:
            A query with the same present time, valid time, system time,
            overlap window and history flag.
        """
        return TemporalQuery(
            now=self.now,
            valid_at=self.valid_at,
            known_at=self.known_at,
            overlaps=self.overlaps,
            include_history=self.include_history,
        )


def _require_aware(name: str, value: datetime) -> None:
    """Reject a datetime that carries no timezone.

    Args:
        name: Name of the field, used in the message.
        value: The datetime to check.

    Raises:
        RetrievalError: If the datetime has no timezone.
    """
    try:
        require_utc(value)
    except ValueError as exc:
        msg = f"{name} must have a timezone"
        raise RetrievalError(msg) from exc
