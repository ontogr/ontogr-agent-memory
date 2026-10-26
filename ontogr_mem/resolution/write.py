"""Turn one resolution plan into the port calls that store it.

The entity and its aliases are written first and mentions last. The resolver
writes mentions after every group, because one episode may span several groups.
"""

from ontogr_mem.data_models import Clock
from ontogr_mem.data_models.entities import (
    Alias,
    AliasSource,
    Entity,
    EntityStatus,
    Mention,
    Outcome,
    ResolutionState,
)
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.errors import NotFoundError
from ontogr_mem.resolution.grouping import Group
from ontogr_mem.resolution.ids import alias_id, entity_id, mention_id
from ontogr_mem.resolution.keys import loose_key, strict_key
from ontogr_mem.resolution.plan import Plan
from ontogr_mem.resolution.policy import TypeRelations
from ontogr_mem.resolution.resolver_store import ResolverStore


async def write_group(
    store: ResolverStore,
    group: Group,
    plan: Plan,
    *,
    settings: ResolverSettings,
    relations: TypeRelations,
    clock: Clock,
    incomplete: bool,
) -> tuple[Entity, list[Alias], list[Mention]]:
    """Write the entity and aliases of one group and return its rows.

    Returns:
        The stored entity, the aliases written with it, and the mentions that
        the caller will write once every group of the episode is decided.
    """
    if plan.entity_id is not None:
        stored = await store.get_entities([plan.entity_id])
        if not stored:
            raise NotFoundError(f"entity {plan.entity_id!r}")
        entity = _matched(stored[0], group, settings, relations, clock)
    else:
        entity = await _created(
            store, group, plan, settings, clock, incomplete=incomplete
        )

    aliases = await _aliases(store, entity.entity_id, group)
    written = await store.write_entity_batch([entity], aliases)
    return written[0], aliases, _mentions(entity.entity_id, group, plan, settings)


def _matched(
    entity: Entity,
    group: Group,
    settings: ResolverSettings,
    relations: TypeRelations,
    clock: Clock,
) -> Entity:
    type_label = entity.type_label
    if type_label.casefold() != group.type_label.casefold():
        type_label = relations.more_specific(type_label, group.type_label)
    descriptions = list(entity.descriptions)
    for description in group.descriptions:
        if description not in descriptions:
            descriptions.append(description)
    attributes = dict(entity.attributes)
    for name, value in group.attributes.items():
        attributes.setdefault(name, value)
    identity_attrs = dict(entity.identity_attrs)
    for name, value in group.identity.items():
        identity_attrs.setdefault(name, value)
    return entity.model_copy(
        update={
            "status": EntityStatus.ACTIVE,
            "type_label": type_label,
            "descriptions": descriptions,
            "attributes": attributes,
            "identity_attrs": identity_attrs,
            "uri": entity.uri or group.uri,
            "summary_stale": entity.summary_stale
            or len(descriptions) >= settings.summary_stale_after,
            "updated_at": clock.now(),
        }
    )


async def _created(
    store: ResolverStore,
    group: Group,
    plan: Plan,
    settings: ResolverSettings,
    clock: Clock,
    *,
    incomplete: bool,
) -> Entity:
    first = group.members[0]
    entity_id_value = entity_id(first.episode_id, first.local_id)
    now = clock.now()
    fresh = Entity(
        entity_id=entity_id_value,
        type_label=group.type_label,
        canonical_name=group.name,
        strict_key=group.strict,
        loose_key=group.loose,
        uri=group.uri,
        identity_attrs=dict(group.identity),
        descriptions=list(group.descriptions),
        attributes=dict(group.attributes),
        mention_count=0,
        resolution_state=(
            ResolutionState.PROVISIONAL
            if plan.outcome is Outcome.CREATED_PROVISIONAL
            else ResolutionState.RESOLVED
        ),
        candidates_incomplete=incomplete,
        candidate_ids=list(plan.candidates),
        created_at=now,
        updated_at=now,
    )
    existing = await store.get_entities([entity_id_value])
    if not existing:
        return fresh
    prior = existing[0]
    return prior.model_copy(
        update={
            "type_label": fresh.type_label,
            "canonical_name": fresh.canonical_name,
            "strict_key": fresh.strict_key,
            "loose_key": fresh.loose_key,
            "uri": prior.uri or fresh.uri,
            "identity_attrs": {**fresh.identity_attrs, **prior.identity_attrs},
            "descriptions": prior.descriptions or fresh.descriptions,
            "attributes": {**fresh.attributes, **prior.attributes},
            "resolution_state": fresh.resolution_state,
            "candidates_incomplete": (
                prior.candidates_incomplete or fresh.candidates_incomplete
            ),
            "candidate_ids": fresh.candidate_ids,
            "summary_stale": prior.summary_stale,
            "updated_at": now,
        }
    )


async def _aliases(store: ResolverStore, entity_id: str, group: Group) -> list[Alias]:
    existing = {alias.strict_key for alias in await store.aliases_of(entity_id)}
    first_episode_id = group.members[0].episode_id
    out: list[Alias] = []
    for form in group.surface_forms:
        key = strict_key(form)
        if not key or key in existing:
            continue
        existing.add(key)
        out.append(
            Alias(
                alias_id=alias_id(entity_id, key),
                entity_id=entity_id,
                text=form,
                strict_key=key,
                loose_key=loose_key(form),
                source=AliasSource.EXTRACTED,
                first_episode_id=first_episode_id,
            )
        )
    return out


def _mentions(
    entity_id: str, group: Group, plan: Plan, settings: ResolverSettings
) -> list[Mention]:
    return [
        Mention(
            mention_id=mention_id(
                member.episode_id, member.local_id, settings.resolver_version
            ),
            entity_id=entity_id,
            episode_id=member.episode_id,
            local_id=member.local_id,
            surface_form=member.name,
            type_label_at_extraction=member.type_label or "",
            resolver_version=settings.resolver_version,
            decision_method=plan.method,
            decision_outcome=plan.outcome,
            decision_score=plan.score,
            decision_reason=plan.reason,
            decision_candidates=list(plan.candidates),
        )
        for member in group.members
    ]
