"""Canonical-name selection and cached-text removal for resolver stores."""

from collections.abc import Sequence

from ontogr_mem.data_models.entities import (
    Alias,
    AliasSource,
    Entity,
    Mention,
    MergeRecord,
)
from ontogr_mem.resolution.ids import alias_id
from ontogr_mem.resolution.keys import loose_key, strict_key


def pruned_entity_text(
    entity: Entity,
    mentions: Sequence[Mention],
    aliases: Sequence[Alias],
) -> tuple[Entity, list[Alias], list[Alias]]:
    """Return the cleared entity, supported aliases and missing canonical alias.

    Keep a mentioned canonical name. Otherwise select the most frequent
    surface form, breaking ties by the first mention id. With no mentions,
    preserve the canonical name and its aliases.
    """
    mentions_by_surface: dict[str, tuple[int, Mention]] = {}
    for mention in mentions:
        count, first = mentions_by_surface.get(mention.surface_form, (0, mention))
        mentions_by_surface[mention.surface_form] = (count + 1, first)

    canonical_name = entity.canonical_name
    canonical_key = strict_key(canonical_name)
    supported_keys = {strict_key(m.surface_form) for m in mentions}
    selected_mention: Mention | None = None
    if canonical_key not in supported_keys and mentions_by_surface:
        _, (_, selected_mention) = min(
            mentions_by_surface.items(),
            key=lambda item: (
                -item[1][0],
                item[1][1].mention_id,
            ),
        )
        canonical_name = selected_mention.surface_form
        canonical_key = strict_key(canonical_name)

    canonical_mention = selected_mention or next(
        (m for m in mentions if strict_key(m.surface_form) == canonical_key),
        None,
    )
    retained = [
        row
        for row in aliases
        if row.strict_key in supported_keys or row.strict_key == canonical_key
    ]
    new_aliases: list[Alias] = []
    if canonical_mention is not None and not any(
        row.strict_key == canonical_key for row in retained
    ):
        new_aliases.append(
            Alias(
                alias_id=alias_id(entity.entity_id, canonical_key),
                entity_id=entity.entity_id,
                text=canonical_name,
                strict_key=canonical_key,
                loose_key=loose_key(canonical_name),
                source=AliasSource.EXTRACTED,
                first_episode_id=canonical_mention.episode_id,
            )
        )

    updated = entity.model_copy(
        update={
            "canonical_name": canonical_name,
            "strict_key": strict_key(canonical_name),
            "loose_key": loose_key(canonical_name),
            "summary": "",
            "descriptions": [],
            "attributes": {},
            "summary_stale": True,
        }
    )
    return updated, retained, new_aliases


def purged_merge_snapshot(
    record: MergeRecord, *, entity_id: str | None, before_seq: int | None
) -> MergeRecord | None:
    """Return a purged record if its snapshot matches the supplied filters."""
    if record.snapshot is None:
        return None
    if entity_id is not None and entity_id not in (
        record.survivor_id,
        record.absorbed_id,
    ):
        return None
    if before_seq is not None and record.seq >= before_seq:
        return None
    return record.model_copy(
        update={
            "snapshot": None,
            "snapshot_purged": True,
            "survivor_delta": {},
            "replaced_verdicts": [],
        }
    )
