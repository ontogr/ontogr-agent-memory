"""Per-episode status, storage and the batch driver.

The batch driver:

* returns a stored final result (ok, empty) without calling the extractor;
* processes parse_failed, error and skipped episodes again;
* saves status, facts, drops and usage together, replacing everything stored for
  the episode.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

from ontogr_mem.data_models import (
    RETRYABLE,
    Episode,
    ExtractedEntity,
    PreviousEpisodes,
    Status,
    TypeView,
)
from ontogr_mem.fact_extraction.extractor import FactExtractor
from ontogr_mem.fact_extraction.models import FactReport, FactResult
from ontogr_mem.fact_extraction.settings import FactExtractionSettings
from ontogr_mem.llm import LLMError


log = logging.getLogger(__name__)


class FactStatusKey(NamedTuple):
    """Identity of one episode's fact-extraction status.

    A new extractor version or type catalog gives a new key.
    """

    episode_id: str
    extractor_version: str
    catalog_version: str


class FactStore(Protocol):
    """Persistence for fact results and statuses (owned by the job record)."""

    async def get(self, key: FactStatusKey) -> FactResult | None:
        """Return the stored result for `key`, if any."""
        ...

    async def save(self, key: FactStatusKey, result: FactResult) -> None:
        """Replace all of the episode's records and set its status, atomically."""
        ...

    async def remove_episode(self, episode_id: str) -> None:
        """Remove every record and status belonging to the episode."""
        ...


class InMemoryFactStore:
    """Reference implementation of `FactStore`.

    The store keeps the whole result, so a resumed episode returns its facts.
    Extracted facts are not stored beyond the process.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._results: dict[FactStatusKey, FactResult] = {}
        self._by_episode: defaultdict[str, set[FactStatusKey]] = defaultdict(set)

    async def get(self, key: FactStatusKey) -> FactResult | None:
        """Return the stored result for `key`."""
        return self._results.get(key)

    async def save(self, key: FactStatusKey, result: FactResult) -> None:
        """Drop the episode's older records (any version), then store this one."""
        for old in self._by_episode[key.episode_id]:
            self._results.pop(old, None)
        self._by_episode[key.episode_id] = {key}
        self._results[key] = result

    async def remove_episode(self, episode_id: str) -> None:
        """Remove the records and statuses of one episode."""
        for key in self._by_episode.pop(episode_id, set()):
            self._results.pop(key, None)


class FactExtraction:
    """Runs an extractor over episodes, resuming from stored final statuses."""

    def __init__(
        self,
        extractor: FactExtractor,
        store: FactStore,
        types: TypeView | None = None,
        previous: PreviousEpisodes | None = None,
        settings: FactExtractionSettings | None = None,
    ) -> None:
        """Wire the extractor, store, type view and context reader.

        Args:
            extractor: The fact extractor.
            store: Per-episode status and results.
            types: The relation types to offer, or None.
            previous: Reads earlier episodes for context. None turns it off.
            settings: Concurrency and the number of earlier episodes.
        """
        self._extractor = extractor
        self._store = store
        self._types = types
        self._previous = previous
        self._settings = settings or FactExtractionSettings()
        # Bounds how many episodes are in flight at once (the model gate has its
        # own per-size limits on top of this).
        self._semaphore = asyncio.Semaphore(self._settings.concurrency)

    def _key(self, episode: Episode) -> FactStatusKey:
        """Return the status key for an episode under the current settings."""
        catalog = self._types.version if self._types else "no-types"
        return FactStatusKey(episode.id, self._extractor.version, catalog)

    async def extract(
        self,
        episodes: Sequence[Episode],
        entities: Mapping[str, Sequence[ExtractedEntity]] | None = None,
    ) -> FactReport:
        """Extract the episodes concurrently.

        Results come back in the same order as `episodes`. A failure in one
        episode never affects another.

        Args:
            episodes: The episodes to extract.
            entities: Candidate entities per episode id. An episode with no entry
                gets no candidates.
        """
        by_episode = entities or {}
        results = await asyncio.gather(
            *(
                self._guarded(episode, by_episode.get(episode.id, ()))
                for episode in episodes
            )
        )
        return FactReport(list(results))

    async def _guarded(
        self, episode: Episode, entities: Sequence[ExtractedEntity]
    ) -> FactResult:
        """Extract one episode, resuming a final stored result when possible."""
        async with self._semaphore:
            key = self._key(episode)
            stored = await self._store.get(key)
            # Final statuses (ok, empty) are returned without a call; parse_failed,
            # error and skipped are processed again.
            if (
                stored is not None
                and stored.status not in RETRYABLE
                and stored.status is not Status.SKIPPED
            ):
                return stored
            previous: Sequence[Episode] = ()
            if self._previous is not None and self._settings.max_previous > 0:
                previous = await self._previous.before(
                    episode, self._settings.max_previous
                )
            try:
                result = await self._extractor.extract(
                    episode, list(entities), self._types, previous
                )
            except (LLMError, TimeoutError) as exc:
                log.warning("fact extraction failed for %s: %r", episode.id, exc)
                result = FactResult(
                    episode.id,
                    Status.ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            # Status, facts, drops and usage are saved together.
            await self._store.save(key, result)
            return result
