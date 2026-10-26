"""Ingestion pipeline: validate, id, write, hand off, record the hash last."""

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from ontogr_mem.data_models import Clock, Episode, EpisodeDraft, Kind
from ontogr_mem.ingestion.connectors import Connector, connector_for
from ontogr_mem.ingestion.ids import content_hash, seal
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.ingestion.validate import validate_batch


Handoff = Callable[[Sequence[Episode]], Awaitable[None]]
"""Callback that receives the newly written episodes, such as extraction."""


class IngestionPipeline:
    """Runs the ingestion steps for a batch of connector inputs.

    The whole batch is validated before anything is written. Episodes whose
    stored hash already matches are skipped. New episodes are written in
    batches, handed off, then recorded by hash. A failed hand-off propagates
    without recording the hashes, so the next run retries it. Delivery is
    at least once: an uncertain hand-off can be repeated on a later run.
    """

    def __init__(
        self,
        store: EpisodeStore,
        registry: Mapping[Kind, Connector[Any]],
        clock: Clock,
        handoff: Handoff | None = None,
        *,
        batch_size: int = 500,
    ) -> None:
        """Wire the store, connector registry, clock and optional hand-off.

        Raises:
            ValueError: If `batch_size` is below one. A batch size of zero or
                less would write nothing and report success.
        """
        if batch_size < 1:
            msg = "batch_size must be at least 1"
            raise ValueError(msg)
        self._store = store
        self._registry = registry
        self._clock = clock
        self._handoff = handoff
        self._batch_size = batch_size

    async def ingest(self, kind: Kind | str, items: Iterable[Any]) -> None:
        """Ingest `items` through the connector registered for `kind`.

        Raises:
            UnknownKindError: If no connector handles `kind`.
            DraftValidationError: If any draft is invalid. Nothing is written.
        """
        connector = connector_for(self._registry, kind)
        drafts: list[EpisodeDraft] = [
            draft for item in items for draft in connector.collect(item)
        ]
        validate_batch(drafts)

        ingested_at = self._clock.now()
        pending: dict[str, tuple[Episode, str]] = {}
        for draft in drafts:
            episode = seal(draft, ingested_at=ingested_at)
            digest = content_hash(draft.content)
            if episode.id in pending:
                continue
            if await self._store.get_content_hash(episode.id) == digest:
                continue
            pending[episode.id] = (episode, digest)

        new = [episode for episode, _ in pending.values()]
        for start in range(0, len(new), self._batch_size):
            batch = new[start : start + self._batch_size]
            await self._store.write_episodes(batch)

        if new and self._handoff is not None:
            await self._handoff(new)

        for episode in new:
            await self._store.record_content_hash(episode.id, pending[episode.id][1])
