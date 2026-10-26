"""Mapping of structured episodes (records and events) to entities, no model."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ontogr_mem.data_models import (
    Dropped,
    DropReason,
    Episode,
    ExtractedEntity,
    ExtractionResult,
    Method,
    Status,
    TypeCatalog,
)


@dataclass(frozen=True, slots=True)
class FieldEntity:
    """An entity whose name is the value of `field`.

    Attributes:
        field: Key in the record content that holds the name.
        type_label: Catalog label of the entity type.
        attribute_fields: Other keys copied into the entity attributes.
    """

    field: str
    type_label: str
    attribute_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecordMapping:
    """How one table or event type maps to entities."""

    entities: tuple[FieldEntity, ...]


class RecordMapper:
    """Map structured content to entities without a model.

    A mapping is looked up by `"<source.system>/<metadata.table>"` and then by
    `source.system`. An episode with no mapping is skipped.
    """

    def __init__(self, mappings: Mapping[str, RecordMapping]) -> None:
        """Register mappings by key."""
        self._mappings = dict(mappings)

    def map(self, episode: Episode, types: TypeCatalog) -> ExtractionResult:
        """Build the result for one structured episode."""
        result = ExtractionResult(episode.id, Status.EMPTY)
        mapping = self._lookup(episode)
        data = episode.content
        if mapping is None or not isinstance(data, dict):
            result.status = Status.SKIPPED
            result.detail = "no record mapping" if mapping is None else "not a dict"
            return result
        for spec in mapping.entities:
            entity = self._entity(episode, spec, data, types, len(result.entities))
            if isinstance(entity, Dropped):
                result.dropped.append(entity)
            else:
                result.entities.append(entity)
        if result.entities:
            result.status = Status.OK
        return result

    def _lookup(self, episode: Episode) -> RecordMapping | None:
        system = episode.source.system
        table = episode.metadata.get("table")
        if table is not None and f"{system}/{table}" in self._mappings:
            return self._mappings[f"{system}/{table}"]
        return self._mappings.get(system)

    @staticmethod
    def _entity(
        episode: Episode,
        spec: FieldEntity,
        data: dict[str, Any],
        types: TypeCatalog,
        n: int,
    ) -> ExtractedEntity | Dropped:
        value = data.get(spec.field)
        if value is None or not str(value).strip():
            return Dropped({"field": spec.field}, DropReason.MISSING_FIELD)
        known = types.entity_by_label(spec.type_label)
        if known is None:
            return Dropped(
                {"field": spec.field, "type": spec.type_label},
                DropReason.TYPE_OUT_OF_RANGE,
            )
        return ExtractedEntity(
            local_id=f"r{n}",
            name=str(value).strip(),
            type_id=known.id,
            confidence=1.0,
            method=Method.RECORD,
            episode_id=episode.id,
            attributes={f: data[f] for f in spec.attribute_fields if f in data},
        )
