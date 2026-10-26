"""Entity extraction for a batch of episodes."""

import asyncio
import logging
from collections import Counter
from collections.abc import Sequence

from ontogr_mem.data_models import (
    Episode,
    ExtractedEntity,
    ExtractionResult,
    Kind,
    Method,
    PreviousEpisodes,
    Status,
    TypeCatalog,
)
from ontogr_mem.data_models.text import same_name
from ontogr_mem.entity_extraction.errors import (
    ExtractionError,
    ParseFailedError,
    UnknownKindError,
)
from ontogr_mem.entity_extraction.extractors import (
    CompositeExtractor,
    EntityExtractor,
    RecordMapper,
)
from ontogr_mem.entity_extraction.report import ExtractionReport
from ontogr_mem.entity_extraction.settings import EntityExtractionSettings
from ontogr_mem.entity_extraction.text import language_of
from ontogr_mem.llm import InputTooLargeError, LLMError


log = logging.getLogger(__name__)

TEXT_KINDS = frozenset({Kind.MESSAGE, Kind.DOCUMENT, Kind.TOOL_OUTPUT})
STRUCTURED_KINDS = frozenset({Kind.RECORD, Kind.EVENT})


def _compose(extractors: Sequence[EntityExtractor]) -> EntityExtractor:
    """Combine the extractors into the one the pipeline calls.

    A single extractor is used as it is, so it keeps its own thresholds and
    escalation triggers. Several extractors become a `CompositeExtractor`,
    which runs the ones that support the episode's language and merges their
    entities.

    Args:
        extractors: The text extractors to combine.

    Returns:
        The extractor the pipeline calls for every text episode.

    Raises:
        ValueError: If no extractor is given.
    """
    if not extractors:
        msg = "EntityExtraction requires at least one extractor"
        raise ValueError(msg)
    if len(extractors) == 1:
        return extractors[0]
    return CompositeExtractor(extractors)


class EntityExtraction:
    """Extract entities from episodes, routing each by kind.

    Text episodes go to one extractor. With several extractors the pipeline
    builds a `CompositeExtractor` from them, so every extractor that supports
    the episode's language runs and their entities are merged. With one
    extractor that extractor is used as it is, which keeps its own thresholds
    and escalation triggers. With several extractors the triggers in the
    report are the composite's, computed over the merged entities. Records and
    events go to the record mapper. A failure in one episode never affects
    another. An episode too long for the model is skipped with the detail
    `oversize`, and the caller splits it. A message episode also gets an `actor`
    entity for its speaker.
    """

    def __init__(
        self,
        extractors: Sequence[EntityExtractor],
        record_mapper: RecordMapper,
        types: TypeCatalog,
        previous: PreviousEpisodes | None = None,
        settings: EntityExtractionSettings | None = None,
    ) -> None:
        """Create the pipeline.

        Args:
            extractors: Text extractors, composed into the one the pipeline
                calls. Several extractors run together and their entities are
                merged.
            record_mapper: Maps record and event episodes without a model.
            types: The entity types to label with.
            previous: Reads earlier episodes for context. None turns it off.
            settings: Concurrency and the number of earlier episodes.
        """
        self._extractor = _compose(extractors)
        self._records = record_mapper
        self._types = types
        self._previous = previous
        self._settings = settings or EntityExtractionSettings()
        self._semaphore = asyncio.Semaphore(self._settings.concurrency)

    async def extract(self, episodes: Sequence[Episode]) -> ExtractionReport:
        """Extract the episodes concurrently.

        Raises:
            UnknownKindError: If an episode has a kind with no extraction path.
        """
        outcomes = await asyncio.gather(*(self._guarded(e) for e in episodes))
        stats: Counter[str] = Counter()
        for _, triggers in outcomes:
            stats.update(triggers)
        return ExtractionReport([r for r, _ in outcomes], dict(stats))

    async def _guarded(self, episode: Episode) -> tuple[ExtractionResult, list[str]]:
        async with self._semaphore:
            try:
                return await self._extract_one(episode)
            except UnknownKindError:
                raise
            except InputTooLargeError:
                oversize = ExtractionResult(
                    episode.id, Status.SKIPPED, detail="oversize"
                )
                return oversize, []
            except (ExtractionError, LLMError, TimeoutError) as exc:
                log.warning("extraction failed for %s: %r", episode.id, exc)
                status = (
                    Status.PARSE_FAILED
                    if isinstance(exc, ParseFailedError)
                    else Status.ERROR
                )
                detail = f"{type(exc).__name__}: {exc}"
                return ExtractionResult(episode.id, status, detail=detail), []

    async def _extract_one(
        self, episode: Episode
    ) -> tuple[ExtractionResult, list[str]]:
        if episode.kind in STRUCTURED_KINDS:
            return self._records.map(episode, self._types), []
        if episode.kind not in TEXT_KINDS:
            msg = f"no extraction path for kind {episode.kind!r}"
            raise UnknownKindError(msg)
        language = language_of(episode)
        if (
            self._extractor.languages is not None
            and language not in self._extractor.languages
        ):
            detail = f"no extractor for language {language!r}"
            return ExtractionResult(episode.id, Status.SKIPPED, detail=detail), []
        previous: Sequence[Episode] = ()
        if self._previous is not None and self._settings.max_previous > 0:
            previous = await self._previous.before(episode, self._settings.max_previous)
        found = await self._extractor.extract(episode, self._types, previous)
        result = ExtractionResult(
            episode.id,
            Status.OK if found.entities else Status.EMPTY,
            self._with_actor(episode, found.entities),
            usage=found.usage,
        )
        return result, [t.value for t in found.triggers]

    def _with_actor(
        self, episode: Episode, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        if episode.kind is not Kind.MESSAGE:
            return entities
        actor = episode.actor
        name = actor.display_name or actor.id
        if any(same_name(e.name, name) for e in entities):
            return entities
        speaker = ExtractedEntity(
            local_id="actor",
            name=name,
            type_id=self._types.actor_type_id(),
            confidence=1.0,
            method=Method.RECORD,
            episode_id=episode.id,
            attributes={"actor_id": actor.id, "role": actor.role.value},
        )
        return [*entities, speaker]
