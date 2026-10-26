"""Errors raised by temporal reconciliation."""

from ontogr_mem.errors import AgentMemoryError


class TemporalError(AgentMemoryError):
    """Base class for temporal errors."""


class RevisionConflict(TemporalError):  # noqa: N818
    """A compare-and-swap write used a stale revision.

    Attributes:
        key: Slot key or ledger chain that changed.
        base_revision: Revision the caller read before the write.
    """

    def __init__(self, key: str, base_revision: int) -> None:
        """Name the key and the stale revision.

        Args:
            key: Slot key or ledger chain that changed.
            base_revision: Revision the caller read before the write.
        """
        super().__init__(
            f"revision conflict on {key!r}: base revision {base_revision} is stale"
        )
        self.key = key
        self.base_revision = base_revision


class MissingEntityError(TemporalError):
    """A fact names an entity that has no node.

    Attributes:
        entity_id: Entity that does not exist in the tenant.
    """

    def __init__(self, entity_id: str) -> None:
        """Name the missing entity.

        Args:
            entity_id: Entity that does not exist in the tenant.
        """
        super().__init__(f"entity {entity_id!r} does not exist")
        self.entity_id = entity_id


class MissingEpisodeError(TemporalError):
    """A temporal state was written for an episode that does not exist.

    Attributes:
        episode_id: Episode that does not exist in the tenant.
    """

    def __init__(self, episode_id: str) -> None:
        """Name the missing episode.

        Args:
            episode_id: Episode that does not exist in the tenant.
        """
        super().__init__(f"episode {episode_id!r} does not exist")
        self.episode_id = episode_id


class MissingFactVersionError(TemporalError):
    """A commit expires a fact version that the store does not hold.

    Attributes:
        fact_id: Fact whose version is missing.
        version: Version number that is missing.
    """

    def __init__(self, fact_id: str, version: int) -> None:
        """Name the missing fact version.

        Args:
            fact_id: Fact whose version is missing.
            version: Version number that is missing.
        """
        super().__init__(f"fact {fact_id!r} has no version {version} to expire")
        self.fact_id = fact_id
        self.version = version


class EpisodeAlreadyIngestedError(TemporalError):
    """An episode already has observations.

    Attributes:
        episode_id: Episode that already has observations.
    """

    def __init__(self, episode_id: str) -> None:
        """Name the episode that is already ingested.

        Args:
            episode_id: Episode that already has observations.
        """
        super().__init__(
            f"episode {episode_id!r} already has observations; "
            "reset it before ingesting again"
        )
        self.episode_id = episode_id


class EpisodeMismatchError(TemporalError, ValueError):
    """A fact belongs to another episode than the one being ingested.

    Attributes:
        fact_id: Fact that names the other episode.
        episode_id: Episode the fact belongs to.
    """

    def __init__(self, fact_id: str, episode_id: str) -> None:
        """Name the fact and its episode.

        Args:
            fact_id: Fact that names the other episode.
            episode_id: Episode the fact belongs to.
        """
        super().__init__(f"fact {fact_id!r} is from episode {episode_id!r}")
        self.fact_id = fact_id
        self.episode_id = episode_id


class UnknownRelationError(TemporalError, ValueError):
    """A relation type id is not part of the view that should contain it.

    Attributes:
        relation_type_id: Id that was not found.
    """

    def __init__(self, relation_type_id: int) -> None:
        """Name the unknown relation type.

        Args:
            relation_type_id: Id that was not found.
        """
        super().__init__(f"relation type {relation_type_id} is not in the view")
        self.relation_type_id = relation_type_id
