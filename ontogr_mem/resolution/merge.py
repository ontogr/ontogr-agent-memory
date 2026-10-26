"""Merge and unmerge of canonical entities.

A merge never deletes: the absorbed entity becomes ``MERGED`` with a pointer to
the survivor, and the record keeps a snapshot plus what moved. An unmerge
restores that snapshot unless a later merge or a purged snapshot prevents it.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from ontogr_mem.data_models import Clock
from ontogr_mem.data_models.entities import (
    Alias,
    AliasSource,
    Entity,
    EntityStatus,
    EntityVerdict,
    Mention,
    MergeRecord,
    MergeState,
    VerdictKind,
    VerdictSource,
)
from ontogr_mem.data_models.ids import stable_id
from ontogr_mem.resolution.errors import (
    MergeRefusedError,
    NotFoundError,
    SnapshotPurgedError,
)
from ontogr_mem.resolution.policy import (
    ResolutionPolicy,
    TypeRel,
    TypeRelations,
    identity_conflict,
)
from ontogr_mem.resolution.resolver_store import (
    MergePlan,
    ResolverStore,
    UnmergePlan,
)
from ontogr_mem.resolution.verdict_ids import inputs_hash, verdict_id


def is_user(actor: str) -> bool:
    """Return whether an actor is a person rather than an automated caller."""
    return actor == "user" or actor.startswith("user:")


def survivor_order(entity: Entity) -> tuple[bool, int, datetime, int, str]:
    """Sort key: bound entity first, then most mentions, oldest, longer name, id."""
    return (
        not (entity.uri or entity.identity_attrs),
        -entity.mention_count,
        entity.created_at,
        -len(entity.canonical_name),
        entity.entity_id,
    )


def pick_survivor(entities: Iterable[Entity]) -> Entity:
    """Return the survivor of a group, independent of input order."""
    return min(entities, key=survivor_order)


def _require(entities: Sequence[Entity], entity_id: str) -> Entity:
    for entity in entities:
        if entity.entity_id == entity_id:
            return entity
    raise NotFoundError(f"entity {entity_id!r}")


def _check_allowed(
    policy: ResolutionPolicy,
    relations: TypeRelations,
    survivor: Entity,
    absorbed: Entity,
    actor: str,
) -> None:
    if survivor.entity_id == absorbed.entity_id:
        raise MergeRefusedError("same_entity", survivor.entity_id)
    for entity in (survivor, absorbed):
        if entity.status is not EntityStatus.ACTIVE:
            raise MergeRefusedError("not_active", entity.entity_id)
    if is_user(actor):
        return
    if survivor.uri and absorbed.uri and survivor.uri != absorbed.uri:
        raise MergeRefusedError("different_uri", f"{survivor.uri} vs {absorbed.uri}")
    if identity_conflict(survivor.identity_attrs, absorbed.identity_attrs):
        raise MergeRefusedError("identity_conflict", absorbed.entity_id)
    if (
        policy.relation(survivor.type_label, absorbed.type_label, relations)
        is TypeRel.DISTINCT
    ):
        raise MergeRefusedError(
            "different_family", f"{survivor.type_label} vs {absorbed.type_label}"
        )


async def build_merge_plan(
    store: ResolverStore,
    policy: ResolutionPolicy,
    relations: TypeRelations,
    survivor_id: str,
    absorbed_id: str,
    *,
    clock: Clock,
    actor: str,
    reason: str,
) -> MergePlan:
    """Build the rows one merge writes without changing the store."""
    entities = await store.get_entities([survivor_id, absorbed_id])
    survivor = _require(entities, survivor_id)
    absorbed = _require(entities, absorbed_id)
    _check_allowed(policy, relations, survivor, absorbed, actor)
    aliases = await store.aliases_of(absorbed_id)
    mentions = await store.mentions_of(absorbed_id)
    merge_id = stable_id(
        "merge", survivor_id, absorbed_id, actor, reason, str(len(mentions))
    )
    now = clock.now()
    snapshot = {
        "entity": absorbed.model_dump(mode="json"),
        "aliases": [alias.model_dump(mode="json") for alias in aliases],
    }
    descriptions_added = [
        d for d in absorbed.descriptions if d not in survivor.descriptions
    ]
    attribute_keys_added = [
        k for k in absorbed.attributes if k not in survivor.attributes
    ]
    identity_keys_added = [
        k for k in absorbed.identity_attrs if k not in survivor.identity_attrs
    ]
    delta: dict[str, Any] = {
        "type_label": survivor.type_label,
        "uri": survivor.uri,
        "descriptions_added": descriptions_added,
        "attribute_keys_added": attribute_keys_added,
        "identity_keys_added": identity_keys_added,
        "mention_count_added": len(mentions),
        "summary_stale": survivor.summary_stale,
        "candidate_ids": list(survivor.candidate_ids),
    }
    updated_survivor = survivor.model_copy(
        update={
            "type_label": relations.more_specific(
                survivor.type_label, absorbed.type_label
            ),
            "descriptions": survivor.descriptions + descriptions_added,
            "attributes": {**absorbed.attributes, **survivor.attributes},
            "identity_attrs": {
                **absorbed.identity_attrs,
                **survivor.identity_attrs,
            },
            "uri": survivor.uri or absorbed.uri,
            "mention_count": survivor.mention_count + len(mentions),
            "candidate_ids": [
                candidate
                for candidate in survivor.candidate_ids
                if candidate != absorbed_id
            ],
            "summary_stale": survivor.summary_stale or bool(absorbed.summary),
            "updated_at": now,
        }
    )
    updated_absorbed = absorbed.model_copy(
        update={
            "status": EntityStatus.MERGED,
            "merged_into": survivor_id,
            "mention_count": 0,
            "updated_at": now,
        }
    )
    record = MergeRecord(
        merge_id=merge_id,
        survivor_id=survivor_id,
        absorbed_id=absorbed_id,
        actor=actor,
        reason=reason,
        state=MergeState.APPLIED,
        snapshot=snapshot,
        survivor_delta=delta,
        moved_mention_ids=[mention.mention_id for mention in mentions],
        moved_alias_ids=[alias.alias_id for alias in aliases],
        created_at=now,
    )
    moved_mentions = tuple(
        mention.model_copy(update={"entity_id": survivor_id}) for mention in mentions
    )
    moved_aliases = tuple(
        alias.model_copy(
            update={
                "entity_id": survivor_id,
                "source": AliasSource.MERGED,
                "merge_id": merge_id,
            }
        )
        for alias in aliases
    )
    return MergePlan(
        record=record,
        survivor=updated_survivor,
        absorbed=updated_absorbed,
        mentions=moved_mentions,
        aliases=moved_aliases,
    )


def build_unmerge_plan(
    record: MergeRecord,
    survivor: Entity,
    absorbed: Entity,
    mentions: Sequence[Mention],
    aliases: Sequence[Alias],
    *,
    clock: Clock,
) -> UnmergePlan:
    """Build the rows that reverse an applied merge."""
    if record.snapshot_purged or record.snapshot is None:
        raise SnapshotPurgedError(record.merge_id)
    if record.state is not MergeState.APPLIED:
        raise MergeRefusedError("not_applied", record.merge_id)
    now = clock.now()
    delta = record.survivor_delta
    restored_survivor = survivor.model_copy(
        update={
            "type_label": delta["type_label"],
            "uri": delta["uri"],
            "descriptions": [
                d for d in survivor.descriptions if d not in delta["descriptions_added"]
            ],
            "attributes": {
                k: v
                for k, v in survivor.attributes.items()
                if k not in delta["attribute_keys_added"]
            },
            "identity_attrs": {
                k: v
                for k, v in survivor.identity_attrs.items()
                if k not in delta["identity_keys_added"]
            },
            "mention_count": max(
                0, survivor.mention_count - delta["mention_count_added"]
            ),
            "summary_stale": delta["summary_stale"],
            "candidate_ids": list(delta["candidate_ids"]),
            "updated_at": now,
        }
    )
    restored_absorbed = absorbed.model_copy(
        update={
            "status": EntityStatus.ACTIVE,
            "merged_into": None,
            "mention_count": len(mentions),
            "updated_at": now,
        }
    )
    lo, hi, verdict_key = verdict_id(survivor.entity_id, absorbed.entity_id)
    verdict = EntityVerdict(
        verdict_id=verdict_key,
        a_id=lo,
        b_id=hi,
        verdict=VerdictKind.DISTINCT,
        source=VerdictSource.USER,
        inputs_hash=inputs_hash(restored_survivor, restored_absorbed),
    )
    return UnmergePlan(
        record=record.model_copy(update={"state": MergeState.REVERTED}),
        survivor=restored_survivor,
        absorbed=restored_absorbed,
        mentions=tuple(
            mention.model_copy(update={"entity_id": absorbed.entity_id})
            for mention in mentions
        ),
        aliases=tuple(
            alias.model_copy(update={"entity_id": absorbed.entity_id, "merge_id": None})
            for alias in aliases
        ),
        verdict=verdict,
    )
