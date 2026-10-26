"""In-memory twin of the episode store."""

import dataclasses
from collections.abc import Sequence
from datetime import datetime

from ontogr_mem.data_models import Episode, TemporalStatus
from ontogr_mem.data_models.episode import EpisodeMeta


class MemoryEpisodeStore:
    """`EpisodeStore` that keeps one tenant's episodes in memory."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._episodes: dict[str, Episode] = {}
        self._meta: dict[str, EpisodeMeta] = {}

    async def get_content_hash(self, episode_id: str) -> str | None:
        """Return the recorded content hash, or None if never fully ingested."""
        episode = self._episodes.get(episode_id)
        return episode.content_hash if episode else None

    async def write_episodes(self, episodes: Sequence[Episode]) -> None:
        """Write episodes, keeping the ingestion and temporal state of stored ones."""
        for episode in episodes:
            stored = self._episodes.get(episode.id)
            if stored is None:
                self._episodes[episode.id] = dataclasses.replace(
                    episode,
                    content_hash=None,
                    temporal_status=None,
                    temporal_committed_hash=None,
                )
                continue
            self._episodes[episode.id] = dataclasses.replace(
                episode,
                ingested_at=stored.ingested_at,
                content_hash=stored.content_hash,
                temporal_status=stored.temporal_status,
                temporal_committed_hash=stored.temporal_committed_hash,
            )

    async def record_content_hash(self, episode_id: str, content_hash: str) -> None:
        """Record the content hash."""
        episode = self._episodes.get(episode_id)
        if episode is not None:
            self._episodes[episode_id] = dataclasses.replace(
                episode, content_hash=content_hash
            )

    async def remove_episode(self, episode_id: str) -> None:
        """Delete an episode and its metadata; unused sources and actors follow."""
        self._meta.pop(episode_id, None)
        self._episodes.pop(episode_id, None)

    async def write_episode_meta(self, meta: EpisodeMeta) -> None:
        """Replace the session, thread, subject reference and fact ids.

        Args:
            meta: Metadata for an episode in this tenant.

        Raises:
            ValueError: The episode does not exist in this tenant.
        """
        if meta.episode_id not in self._episodes:
            msg = "unknown episode"
            raise ValueError(msg)
        self._meta[meta.episode_id] = meta

    async def get_episode_meta(self, episode_id: str) -> EpisodeMeta | None:
        """Return the episode metadata, or None if none is stored."""
        return self._meta.get(episode_id)

    async def episode_ids_by_subject(self, subject_ref: str) -> list[str]:
        """Return episode ids whose metadata has this subject reference."""
        return sorted(
            episode_id
            for episode_id, meta in self._meta.items()
            if meta.subject_ref == subject_ref
        )

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Return the episode, or None if it does not exist."""
        return self._episodes.get(episode_id)

    async def get_episodes(self, ids: Sequence[str]) -> list[Episode]:
        """Return the requested episodes, ordered by id.

        Ids that no episode carries are left out.
        """
        found = {
            episode_id: self._episodes[episode_id]
            for episode_id in dict.fromkeys(ids)
            if episode_id in self._episodes
        }
        return [found[key] for key in sorted(found)]

    async def list_episodes(
        self,
        *,
        limit: int,
        before: tuple[datetime, str] | None = None,
        session_id: str | None = None,
    ) -> list[Episode]:
        """Return newest episodes using metadata-backed session filtering."""
        candidates = [
            episode
            for episode in self._episodes.values()
            if session_id is None
            or (
                episode.id in self._meta
                and self._meta[episode.id].session_id == session_id
            )
        ]
        if before is not None:
            candidates = [
                episode
                for episode in candidates
                if (episode.ingested_at, episode.id) < before
            ]
        candidates.sort(
            key=lambda episode: (episode.ingested_at, episode.id), reverse=True
        )
        return candidates[:limit]

    async def latest_before(
        self, before: datetime, limit: int, *, exclude_id: str | None = None
    ) -> Sequence[Episode]:
        """Return up to `limit` fully ingested episodes at or before `before`.

        The result is ordered oldest first.
        """
        candidates = [
            e
            for e in self._episodes.values()
            if e.occurred_at <= before
            and e.id != exclude_id
            and e.content_hash is not None
        ]
        candidates.sort(key=lambda e: (e.occurred_at, e.ingested_at, e.id))
        return candidates[max(len(candidates) - limit, 0) :]

    async def before(self, episode: Episode, limit: int) -> Sequence[Episode]:
        """Return the episodes that came before `episode`, oldest first."""
        return await self.latest_before(
            episode.occurred_at, limit, exclude_id=episode.id
        )

    def ingested_ids(self) -> set[str]:
        """Return the ids of the episodes whose content hash is recorded."""
        return {e.id for e in self._episodes.values() if e.content_hash is not None}

    def all_episodes(self) -> list[Episode]:
        """Return every stored episode, sorted by id."""
        return [self._episodes[key] for key in sorted(self._episodes)]

    def set_temporal_state(
        self,
        episode_id: str,
        status: TemporalStatus | None,
        committed_hash: str | None,
    ) -> None:
        """Set the temporal fields, as the temporal store does."""
        episode = self._episodes[episode_id]
        self._episodes[episode_id] = dataclasses.replace(
            episode, temporal_status=status, temporal_committed_hash=committed_hash
        )
