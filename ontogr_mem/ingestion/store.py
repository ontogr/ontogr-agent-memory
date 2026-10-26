"""The episode store port."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from ontogr_mem.data_models import Episode, PreviousEpisodes
from ontogr_mem.data_models.episode import EpisodeMeta


class EpisodeStore(PreviousEpisodes, Protocol):
    """Stores episodes with their sources and actors for one tenant.

    The store also reads previous episodes. The extraction pipelines can use
    the same store as their context source.
    """

    async def get_content_hash(self, episode_id: str) -> str | None:
        """Return the recorded content hash, or None if never fully ingested."""
        ...

    async def write_episodes(self, episodes: Sequence[Episode]) -> None:
        """Idempotently write episodes with their sources and actors.

        An existing episode keeps its `ingested_at`, its recorded content hash
        and its temporal fields.
        """
        ...

    async def record_content_hash(self, episode_id: str, content_hash: str) -> None:
        """Record the content hash. Called last, after the write succeeded."""
        ...

    async def remove_episode(self, episode_id: str) -> None:
        """Delete an episode, its metadata and unused sources and actors."""
        ...

    async def write_episode_meta(self, meta: EpisodeMeta) -> None:
        """Replace the session, thread, subject reference and fact ids.

        Args:
            meta: Metadata for an episode in this tenant.

        Raises:
            ValueError: The episode does not exist in this tenant.
        """
        ...

    async def get_episode_meta(self, episode_id: str) -> EpisodeMeta | None:
        """Return the episode metadata, or None if none is stored."""
        ...

    async def episode_ids_by_subject(self, subject_ref: str) -> list[str]:
        """Return episode ids whose metadata has this subject reference."""
        ...

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Return the episode, or None if it does not exist."""
        ...

    async def get_episodes(self, ids: Sequence[str]) -> list[Episode]:
        """Return the requested episodes, ordered by id.

        Ids that no episode carries are left out. An empty list of ids returns
        an empty list.
        """
        ...

    async def list_episodes(
        self,
        *,
        limit: int,
        before: tuple[datetime, str] | None = None,
        session_id: str | None = None,
    ) -> list[Episode]:
        """Return newest episodes, optionally before a keyset cursor or in a session."""
        ...

    async def latest_before(
        self, before: datetime, limit: int, *, exclude_id: str | None = None
    ) -> Sequence[Episode]:
        """Return up to `limit` fully ingested episodes at or before `before`.

        The result is ordered oldest first. Episodes with the same
        `occurred_at` are ordered by `ingested_at`, then by id. This read is the
        `before` method of `PreviousEpisodes`, as the extraction pipelines call it.
        """
        ...
