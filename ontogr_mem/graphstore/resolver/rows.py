"""Map resolver domain rows to the properties the graph stores.

The whole model row travels as one JSON property, so a read parses the
model back. The scalar properties beside it answer the indexed lookups.
"""

from typing import Any

from ontogr_mem.data_models.entities import (
    Alias,
    Entity,
    EntityVerdict,
    Mention,
    MergeRecord,
)
from ontogr_mem.resolution.keys import stripped_key


def entity_props(entity: Entity) -> dict[str, Any]:
    """Return the properties that store one entity."""
    props: dict[str, Any] = {
        "type_label": entity.type_label,
        "canonical_name": entity.canonical_name,
        "strict_key": entity.strict_key,
        "loose_key": entity.loose_key,
        "stripped_key": stripped_key(entity.canonical_name),
        "status": entity.status.value,
        "merged_into": entity.merged_into or "",
        "mention_count": entity.mention_count,
        "revision": entity.revision,
        "last_checked_version": entity.last_checked_version,
        "json": entity.model_dump_json(),
    }
    if entity.uri:
        props["uri"] = entity.uri
    return props


def entity_row(entity: Entity) -> dict[str, Any]:
    """Return the row that stores one entity."""
    return {
        "id": entity.entity_id,
        "props": entity_props(entity),
    }


def alias_row(alias: Alias) -> dict[str, Any]:
    """Return the row that stores one alias."""
    return {
        "id": alias.alias_id,
        "entity_id": alias.entity_id,
        "props": {
            "entity_id": alias.entity_id,
            "strict_key": alias.strict_key,
            "loose_key": alias.loose_key,
            # The text is a property of its own, so a fulltext index can cover
            # it. The read mapping keeps using the JSON property.
            "text": alias.text,
            "json": alias.model_dump_json(),
        },
    }


def mention_row(mention: Mention) -> dict[str, Any]:
    """Return the row that stores one mention."""
    return {
        "id": mention.mention_id,
        "entity_id": mention.entity_id,
        "props": {
            "entity_id": mention.entity_id,
            "episode_id": mention.episode_id,
            "local_id": mention.local_id,
            "json": mention.model_dump_json(),
        },
    }


def verdict_row(verdict: EntityVerdict) -> dict[str, Any]:
    """Return the row that stores one entity verdict."""
    return {
        "id": verdict.verdict_id,
        "props": {
            "a_id": verdict.a_id,
            "b_id": verdict.b_id,
            "superseded_by": verdict.superseded_by or "",
            "json": verdict.model_dump_json(),
        },
    }


def merge_row(record: MergeRecord) -> dict[str, Any]:
    """Return the row that stores one merge record."""
    return {
        "id": record.merge_id,
        "props": {
            "survivor_id": record.survivor_id,
            "absorbed_id": record.absorbed_id,
            "state": record.state.value,
            "seq": record.seq,
            "json": record.model_dump_json(),
        },
    }
