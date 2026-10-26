"""The extractor interface and the shared span-based base for local models."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from ontogr_mem.data_models import (
    UNKNOWN_TYPE_ID,
    Episode,
    ExtractedEntity,
    Method,
    Span,
    TypeCatalog,
    Usage,
)
from ontogr_mem.data_models.text import normalize_name
from ontogr_mem.entity_extraction.text import text_of


log = logging.getLogger(__name__)


class Trigger(StrEnum):
    """A sign that a cheap extraction pass handled an episode badly."""

    TOO_FEW_ENTITIES = "too_few_entities"
    HIGH_UNKNOWN_SHARE = "high_unknown_share"
    LOW_MEAN_CONFIDENCE = "low_mean_confidence"
    DOMAIN_TERM_UNCOVERED = "domain_term_uncovered"


@dataclass(slots=True)
class Extracted:
    """What an extractor found in one episode.

    Attributes:
        entities: The entities.
        usage: Language model tokens spent. Empty for local models.
        triggers: Escalation triggers that fired. Only composite and fallback
            extractors set them.
    """

    entities: list[ExtractedEntity]
    usage: Usage = field(default_factory=Usage)
    triggers: tuple[Trigger, ...] = ()


class EntityExtractor(ABC):
    """Base class for entity extractors.

    Attributes:
        languages: Languages the extractor handles, or None for any. Episodes in
            other languages are not routed to it.
    """

    languages: frozenset[str] | None = None

    @abstractmethod
    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Find the entities in the episode text.

        Args:
            episode: The episode to extract from.
            types: The entity types to label with.
            previous: Earlier episodes, oldest first, for disambiguation. Only
                model extractors read them.
        """


@dataclass(frozen=True, slots=True)
class RawSpan:
    """One model hit. `label` is a catalog label when the backend can map it."""

    start: int
    end: int
    text: str
    label: str
    score: float


class SpanEntityExtractor(EntityExtractor):
    """Base for local extractors that return scored character spans.

    Spans scoring at least `threshold` get their catalog type. Spans between
    `floor` and `threshold` are typed Unknown. Spans that do not match the episode
    text are discarded.
    """

    method: ClassVar[Method]

    def __init__(
        self,
        *,
        threshold: float,
        floor: float,
        languages: frozenset[str] | None,
    ) -> None:
        """Check that `0 <= floor <= threshold <= 1`."""
        if not 0.0 <= floor <= threshold <= 1.0:
            msg = "need 0 <= floor <= threshold <= 1"
            raise ValueError(msg)
        self._threshold = threshold
        self._floor = floor
        self.languages = languages

    @abstractmethod
    async def _predict(self, text: str, types: TypeCatalog) -> list[RawSpan]:
        """Run the model off the event loop and return hits at or above the floor."""

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Predict, check the spans against the text and group repeat mentions."""
        text = text_of(episode)
        if not text.strip():
            return Extracted([])
        groups: dict[tuple[str, int], tuple[list[Span], float, str]] = {}
        for hit in await self._predict(text, types):
            if hit.score < self._floor:
                continue
            if not (0 <= hit.start < hit.end <= len(text)) or (
                text[hit.start : hit.end] != hit.text
            ):
                log.debug("discarding span not matching episode: %r", hit)
                continue
            known = types.entity_by_label(hit.label)
            typed = known is not None and hit.score >= self._threshold
            type_id = known.id if known and typed else UNKNOWN_TYPE_ID
            key = (normalize_name(hit.text), type_id)
            spans, best, name = groups.get(key, ([], 0.0, hit.text))
            spans.append((hit.start, hit.end))
            groups[key] = (spans, max(best, hit.score), name)

        return Extracted(
            [
                ExtractedEntity(
                    local_id=f"e{n}",
                    name=name,
                    type_id=type_id,
                    confidence=min(1.0, score),
                    method=self.method,
                    episode_id=episode.id,
                    spans=tuple(sorted(spans)),
                )
                for n, ((_, type_id), (spans, score, name)) in enumerate(groups.items())
            ]
        )
