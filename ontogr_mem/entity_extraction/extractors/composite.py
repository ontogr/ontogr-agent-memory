"""Composite: run several extractors over an episode and merge their entities."""

import asyncio
import logging
from collections.abc import Sequence

from ontogr_mem.data_models import (
    Episode,
    ExtractedEntity,
    TypeCatalog,
    Usage,
)
from ontogr_mem.entity_extraction.errors import ExtractionError
from ontogr_mem.entity_extraction.extractors.base import (
    EntityExtractor,
    Extracted,
)
from ontogr_mem.entity_extraction.extractors.overlaps import resolve_overlaps
from ontogr_mem.entity_extraction.extractors.triggers import (
    EscalationThresholds,
    evaluate_triggers,
)
from ontogr_mem.entity_extraction.text import language_of, text_of
from ontogr_mem.llm import LLMError


log = logging.getLogger(__name__)


class CompositeExtractor(EntityExtractor):
    """Run several extractors over the same episode and merge their entities.

    No single extractor wins everywhere: GLiNER is stronger on multi-word domain
    terms and spaCy on plain proper nouns. Members run concurrently. Duplicates
    merge by normalized name. The earliest member in the list wins the type, and a
    later one only fills an Unknown type or a missing description, so put the most
    trusted member first. Spans are joined. Members that do not support the
    episode language are skipped.

    With `suppress_overlaps`, entities whose spans collide across members are
    resolved by the shared rule. A longer span that adds a content word replaces
    the fragment it contains (`Paris Observatory` over `Paris`). An article-only
    extension or an identical span keeps the earlier one. A span that only
    overlaps another one is kept as well. Entities with no spans are always kept.

    Configuration faults (`ImportError`, `OSError`) and counted failures
    (`ExtractionError`, `LLMError`) propagate so the episode gets an error status.
    Any other member exception is logged and that member is skipped. If every
    member fails, the first error is raised.

    The result also lists the escalation triggers the merged entities hit.
    """

    def __init__(
        self,
        extractors: Sequence[EntityExtractor],
        *,
        suppress_overlaps: bool = True,
        thresholds: EscalationThresholds | None = None,
    ) -> None:
        """Create the extractor with members in priority order."""
        if not extractors:
            msg = "CompositeExtractor requires at least one extractor"
            raise ValueError(msg)
        self._extractors = list(extractors)
        self._suppress_overlaps = suppress_overlaps
        self.thresholds = thresholds or EscalationThresholds()
        languages = [e.languages for e in self._extractors]
        self.languages = (
            None
            if any(lang is None for lang in languages)
            else frozenset().union(*(lang for lang in languages if lang is not None))
        )

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Run the supporting members concurrently and merge their entities."""
        members = [
            e
            for e in self._extractors
            if e.languages is None or language_of(episode) in e.languages
        ]
        results = await asyncio.gather(
            *(e.extract(episode, types, previous) for e in members),
            return_exceptions=True,
        )
        usage = Usage()
        groups: list[list[ExtractedEntity]] = []
        failures: list[Exception] = []
        for member, result in zip(members, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception) or isinstance(
                    result, ImportError | OSError | ExtractionError | LLMError
                ):
                    raise result
                failures.append(result)
                log.warning(
                    "%s failed, skipping its results: %s", type(member).__name__, result
                )
                continue
            usage.merge(result.usage)
            groups.append(result.entities)
        if failures and len(failures) == len(members):
            raise failures[0]
        entities = resolve_overlaps(
            groups,
            text=text_of(episode),
            suppress_overlaps=self._suppress_overlaps,
        )
        return Extracted(
            entities, usage, evaluate_triggers(episode, entities, self.thresholds)
        )
