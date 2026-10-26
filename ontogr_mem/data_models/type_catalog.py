"""Type catalog and the view that extraction receives."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ontogr_mem.data_models.text import normalize_label


UNKNOWN_TYPE_ID = -1
"""Explicit low-confidence type for entities no catalog label fits."""
UNKNOWN_LABEL = "Unknown"


@dataclass(frozen=True, slots=True)
class EntityType:
    """An entity type."""

    id: int
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class RelationType:
    """A relation type with optional direction constraints.

    Empty ``source_types`` / ``target_types`` mean any entity type.
    """

    id: int
    label: str
    description: str = ""
    source_types: frozenset[int] = field(default_factory=frozenset)
    target_types: frozenset[int] = field(default_factory=frozenset)


class TypeCatalog:
    """Entity and relation types with dense ids ``0..n-1``.

    Dense ids make "out of range" well defined: an id outside the range is an
    error for that item, never a downgrade to a generic ``Entity``.
    """

    def __init__(
        self,
        entity_types: Sequence[EntityType],
        relation_types: Sequence[RelationType],
        actor_type_label: str = "Person",
    ) -> None:
        """Validate ids and build lookups."""
        for kind, items in (("entity", entity_types), ("relation", relation_types)):
            if [t.id for t in items] != list(range(len(items))):
                msg = f"{kind} type ids must be 0..n-1 in order"
                raise ValueError(msg)
        self.entity_types = tuple(entity_types)
        self.relation_types = tuple(relation_types)
        self.actor_type_label = actor_type_label
        self._by_label = {normalize_label(t.label): t for t in self.entity_types}
        self._rel_by_label = {normalize_label(t.label): t for t in self.relation_types}

    @property
    def version(self) -> str:
        """Content hash of the catalog; part of the status key."""
        blob = json.dumps(
            [self.entity_types_json(), self.relation_types_json()],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def has_entity_id(self, type_id: int) -> bool:
        """True for catalog ids and for the explicit Unknown id."""
        return type_id == UNKNOWN_TYPE_ID or 0 <= type_id < len(self.entity_types)

    def has_relation_id(self, type_id: int) -> bool:
        """True if ``type_id`` is a relation type id."""
        return 0 <= type_id < len(self.relation_types)

    def entity_label(self, type_id: int) -> str:
        """Label for an entity type id (``Unknown`` for the Unknown id)."""
        if type_id == UNKNOWN_TYPE_ID:
            return UNKNOWN_LABEL
        return self.entity_types[type_id].label

    def entity_by_label(self, label: str) -> EntityType | None:
        """Look up an entity type by normalized label."""
        return self._by_label.get(normalize_label(label))

    def relation_by_label(self, label: str) -> RelationType | None:
        """Look up a relation type by normalized label."""
        return self._rel_by_label.get(normalize_label(label))

    def actor_type_id(self) -> int:
        """Type id for message actors; Unknown if the catalog lacks the label."""
        found = self.entity_by_label(self.actor_type_label)
        return found.id if found else UNKNOWN_TYPE_ID

    def entity_types_json(self) -> list[dict[str, Any]]:
        """Entity types as plain data for prompts."""
        return [
            {"id": t.id, "label": t.label, "description": t.description}
            for t in self.entity_types
        ]

    def relation_types_json(self) -> list[dict[str, Any]]:
        """Relation types as plain data for prompts."""
        return [
            {
                "id": t.id,
                "label": t.label,
                "description": t.description,
                "source_types": sorted(t.source_types),
                "target_types": sorted(t.target_types),
            }
            for t in self.relation_types
        ]

    def direction_ok(self, rel: RelationType, source: int, target: int) -> bool:
        """Check the source/target entity types against the relation pattern.

        Unknown entities pass: their type cannot contradict the pattern.
        """
        return self._side_ok(rel.source_types, source) and self._side_ok(
            rel.target_types, target
        )

    @staticmethod
    def _side_ok(allowed: frozenset[int], type_id: int) -> bool:
        return not allowed or type_id == UNKNOWN_TYPE_ID or type_id in allowed


class TypeMode(StrEnum):
    """How the prompt and the direction check treat ``relation_type_id``.

    A problem with a proposed type never drops the fact: the type is cleared
    and the reason recorded, in every mode.

    * ``strict``: the prompt says the id must be one of the listed types and
      directions are checked.
    * ``preferred``: the prompt says to prefer a listed type; directions are
      checked.
    * ``open``: ids are range-checked but the catalog's direction patterns are
      advisory and not enforced.
    """

    STRICT = "strict"
    PREFERRED = "preferred"
    OPEN = "open"


@dataclass(frozen=True, slots=True)
class TypeView:
    """A catalog plus the mode."""

    catalog: TypeCatalog
    mode: TypeMode = TypeMode.PREFERRED

    @property
    def version(self) -> str:
        """Catalog version plus mode; part of the per-episode status key."""
        return f"{self.catalog.version}:{self.mode.value}"
