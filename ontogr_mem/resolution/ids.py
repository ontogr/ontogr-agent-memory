"""Deterministic ids of resolved entities, mentions and aliases."""

from uuid import UUID

from ontogr_mem.data_models.ids import stable_id, uuid5_id


ENTITY_NAMESPACE = UUID("0f5e2b7a-6c41-4d8e-9a03-1b2c3d4e5f60")
"""Namespace of entity ids. It never changes, so ids survive a rebuild."""


def entity_id(episode_id: str, local_id: str) -> str:
    """Return the id of an entity created for an extracted entity.

    Args:
        episode_id: Episode the first mention came from.
        local_id: Extracted entity id, unique inside the episode.

    Returns:
        A UUIDv5 string over the episode and local id.
    """
    return uuid5_id(ENTITY_NAMESPACE, episode_id, local_id)


def mention_id(episode_id: str, local_id: str, resolver_version: str) -> str:
    """Return the id of a mention, which is also its decision id.

    Args:
        episode_id: Episode the mention came from.
        local_id: Extracted entity id, unique inside the episode.
        resolver_version: Version of the resolver that linked the mention.

    Returns:
        A deterministic hex id.
    """
    return stable_id("mention", episode_id, local_id, resolver_version)


def alias_id(entity_id: str, strict_key: str) -> str:
    """Return the id of an alias of an entity.

    Args:
        entity_id: Entity the name resolves to.
        strict_key: Normalized lookup key of the alias text.

    Returns:
        A deterministic hex id.
    """
    return stable_id("alias", entity_id, strict_key)
