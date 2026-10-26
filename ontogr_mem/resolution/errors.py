"""Errors raised by entity resolution."""

from ontogr_mem.errors import AgentMemoryError


class ResolutionError(AgentMemoryError):
    """Base class for entity-resolution errors."""


class NotFoundError(ResolutionError):
    """An entity or merge record does not exist in the tenant."""


class MergeRefusedError(ResolutionError):
    """A merge broke a rule such as family, uri or status.

    Attributes:
        reason: Stable reason code, such as ``not_active``.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        """Build the message from a reason code and optional detail.

        Args:
            reason: Stable reason code.
            detail: Extra context, such as the two entity ids.
        """
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class UnmergeBlockedError(ResolutionError):
    """A later applied merge touches the survivor or the absorbed entity."""


class SnapshotPurgedError(ResolutionError):
    """The merge record's snapshot was purged, so it cannot be undone."""


class EpisodeAlreadyResolvedError(ResolutionError):
    """Resolution already wrote mentions for the episode.

    Attributes:
        episode_id: Episode that already has mentions.
    """

    def __init__(self, episode_id: str) -> None:
        """Name the episode in the message.

        Args:
            episode_id: Episode that already has mentions.
        """
        super().__init__(f"episode {episode_id!r} already has mentions")
        self.episode_id = episode_id


class EntityHasFactsError(ResolutionError):
    """An entity still has a live fact edge, so deleting it would delete a fact.

    Attributes:
        entity_id: Entity that was not deleted.
    """

    def __init__(self, entity_id: str) -> None:
        """Name the entity in the message.

        Args:
            entity_id: Entity that was not deleted.
        """
        super().__init__(f"entity {entity_id!r} still has live facts")
        self.entity_id = entity_id
