"""The candidate entities the model may refer to.

A fact may only refer to an entity that was offered to the model. The speaker
of a message is always offered, under the local id `actor`.
"""

from collections.abc import Sequence

from ontogr_mem.data_models import (
    UNKNOWN_TYPE_ID,
    Episode,
    ExtractedEntity,
    Kind,
    Method,
    TypeView,
)


ACTOR_ID = "actor"


def with_actor(
    episode: Episode, entities: Sequence[ExtractedEntity], types: TypeView | None
) -> list[ExtractedEntity]:
    """Add the speaker as candidate `actor` for messages.

    The entity layer does not emit the speaker, so the speaker is always added.
    The speaker is never matched by name to another candidate. If a candidate
    already has the `actor` id, the list is returned unchanged. The same person
    can still appear under another id when the entity layer found the same name.
    """
    if episode.kind is not Kind.MESSAGE or any(
        entity.local_id == ACTOR_ID for entity in entities
    ):
        return list(entities)
    actor = episode.actor
    type_id = types.catalog.actor_type_id() if types else UNKNOWN_TYPE_ID
    speaker = ExtractedEntity(
        local_id=ACTOR_ID,
        name=actor.display_name or actor.id,
        type_id=type_id,
        type_label=types.catalog.entity_label(type_id) if types else "Actor",
        confidence=1.0,
        method=Method.RECORD,
        episode_id=episode.id,
        attributes={"actor_id": actor.id, "role": actor.role.value},
    )
    return [*entities, speaker]
