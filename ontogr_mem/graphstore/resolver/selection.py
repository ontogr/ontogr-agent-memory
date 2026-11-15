"""Select resolver rows using the same rules for each storage backend."""

from collections.abc import Iterable

from ontogr_mem.data_models.entities import Entity, EntityStatus, ResolutionState


def deduplication_entities(
    entities: Iterable[Entity],
    *,
    provisional: bool = False,
    sweep_limit: int = 0,
) -> list[Entity]:
    """Return copies of active entities selected for a deduplication run.

    Include entities whose revisions exceed their checkpoints and, when
    requested, checked provisional entities. Fill the sweep from the least
    recently checked remaining entities. Return the result in revision and
    entity-id order.
    """
    active = sorted(
        (entity for entity in entities if entity.status is EntityStatus.ACTIVE),
        key=lambda entity: entity.entity_id,
    )
    selected = {
        entity.entity_id: entity
        for entity in active
        if entity.revision > entity.last_checked_version
    }
    if provisional:
        selected |= {
            e.entity_id: e
            for e in active
            if e.resolution_state is ResolutionState.PROVISIONAL
        }
    rest = [e for e in active if e.entity_id not in selected]
    rest.sort(key=lambda e: (e.last_checked_version, e.entity_id))
    for entity in rest[: max(0, sweep_limit)]:
        selected[entity.entity_id] = entity
    return [
        entity.model_copy(deep=True)
        for entity in sorted(selected.values(), key=lambda e: (e.revision, e.entity_id))
    ]
