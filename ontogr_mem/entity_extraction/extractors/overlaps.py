"""Merge entity groups into one list and resolve the spans that collide."""

import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ontogr_mem.data_models import UNKNOWN_TYPE_ID, ExtractedEntity, Span
from ontogr_mem.data_models.text import normalize_name


log = logging.getLogger(__name__)

_TRIVIAL_EXTENSION_TOKENS = frozenset({"the", "a", "an"})


def resolve_overlaps(
    groups: Sequence[Sequence[ExtractedEntity]],
    *,
    text: str,
    suppress_overlaps: bool = True,
) -> list[ExtractedEntity]:
    """Merge entity groups into one list and resolve the spans that collide.

    Entities of the same name are merged. The first one keeps its type, method
    and confidence, and a later one only fills an Unknown type or a missing
    description. Their spans are joined.

    With `suppress_overlaps`, spans that collide across groups are resolved. A
    longer span that adds a content word replaces the fragment it contains
    (`Paris Observatory` over `Paris`). An article-only extension or an
    identical span keeps the earlier one. A span that only overlaps another
    one is kept as well, so neither group loses an entity. Entities within one
    group never suppress each other, and entities with no spans are always
    kept.

    Args:
        groups: The groups of entities, in priority order. Earlier groups win.
        text: The episode text, used to tell a content word from an article.
        suppress_overlaps: Whether colliding spans are resolved.

    Returns:
        The merged entities, in group order, with fresh local ids.
    """
    merged: dict[str, ExtractedEntity] = {}
    claimed: dict[Span, str] = {}
    for group in groups:
        fresh: dict[Span, str] = {}
        for ent in group:
            key = normalize_name(ent.name)
            if not key:
                continue
            if suppress_overlaps and ent.spans:
                admitted = _admit_spans(ent, key, claimed, merged, text)
                if not admitted:
                    continue
                ent = replace(ent, spans=tuple(admitted))  # noqa: PLW2901
            existing = merged.get(key)
            merged[key] = ent if existing is None else _combine(existing, ent)
            for span in ent.spans:
                fresh[span] = key
        # Claim only after the whole group is processed, so two entities from
        # the same group never suppress each other.
        claimed.update(fresh)
    return [replace(e, local_id=f"e{n}") for n, e in enumerate(merged.values())]


def _combine(existing: ExtractedEntity, ent: ExtractedEntity) -> ExtractedEntity:
    """Merge two entities of the same name."""
    changes: dict[str, Any] = {
        "spans": tuple(sorted(set(existing.spans) | set(ent.spans)))
    }
    if existing.type_id == UNKNOWN_TYPE_ID and ent.type_id != UNKNOWN_TYPE_ID:
        changes |= {"type_id": ent.type_id, "confidence": ent.confidence}
    if not existing.description and ent.description:
        changes["description"] = ent.description
    return replace(existing, **changes)


def _extends_meaningfully(text: str, outer: Span, inner: Span) -> bool:
    """Tell whether `outer` adds more than an article to `inner`."""
    (s, e), (cs, ce) = outer, inner
    extra = f"{text[s:cs]} {text[ce:e]}".casefold()
    return any(t not in _TRIVIAL_EXTENSION_TOKENS for t in re.findall(r"\w+", extra))


def _admit_spans(
    ent: ExtractedEntity,
    key: str,
    claimed: dict[Span, str],
    merged: dict[str, ExtractedEntity],
    text: str,
) -> list[Span]:
    """Return the spans of `ent` that no earlier group has claimed."""
    admitted: list[Span] = []
    for s, e in ent.spans:
        blocked = False
        nested: list[Span] = []
        for (cs, ce), owner_key in claimed.items():
            if owner_key == key or not (s < ce and cs < e):
                continue
            if cs <= s and e <= ce:
                blocked = True
                break
            if s <= cs and ce <= e:
                if not _extends_meaningfully(text, (s, e), (cs, ce)):
                    blocked = True
                    break
                nested.append((cs, ce))
        if blocked:
            log.debug(
                "dropping %r at %d-%d, overlaps an earlier entity", ent.name, s, e
            )
            continue
        for fragment in nested:
            owner_key = claimed.pop(fragment)
            owner = merged.get(owner_key)
            if owner is None:
                continue
            kept = tuple(sp for sp in owner.spans if sp != fragment)
            if kept:
                merged[owner_key] = replace(owner, spans=kept)
            else:
                del merged[owner_key]
        admitted.append((s, e))
    return admitted
