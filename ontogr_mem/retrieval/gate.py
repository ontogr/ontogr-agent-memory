"""Apply retrieval type filters to facts and entity cards."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.retrieval.scope import SearchScope
from ontogr_mem.temporal.query import FactView


class EntityReader(Protocol):
    """Read entities by id for retrieval filtering."""

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return matching entities, optionally following merge targets."""
        ...


_View = TypeVar("_View", bound=FactView)


class FactGate:
    """Apply a scope's type lists to fact views."""

    def __init__(self, scope: SearchScope, entities: EntityReader) -> None:
        """Keep the scope and the entity store."""
        self._scope = scope
        self._entities = entities

    async def admit(self, views: Sequence[_View]) -> list[_View]:
        """Return the views that pass, in their order."""
        if not self._scope.filters_facts or not views:
            return list(views)
        ids = list(
            dict.fromkeys(
                entity_id
                for view in views
                for entity_id in (view.subject_id, view.object_id)
            )
        )
        entities = await self._entities.get_entities(ids, follow_merged=False)
        by_id = {entity.entity_id: entity for entity in entities}
        missing = {
            entity.merged_into
            for entity in entities
            if entity.merged_into and entity.merged_into not in by_id
        }
        while missing:
            merged_entities = await self._entities.get_entities(
                sorted(missing), follow_merged=False
            )
            by_id.update({entity.entity_id: entity for entity in merged_entities})
            missing = {
                entity.merged_into
                for entity in merged_entities
                if entity.merged_into and entity.merged_into not in by_id
            }

        def type_id(entity_id: str) -> int | None:
            entity = by_id.get(entity_id)
            seen: set[str] = set()
            while entity is not None and entity.merged_into is not None:
                if entity.entity_id in seen:
                    return None
                seen.add(entity.entity_id)
                entity = by_id.get(entity.merged_into)
            return entity.ontology_type_id if entity is not None else None

        result: list[_View] = []
        for view in views:
            subject = type_id(view.subject_id)
            object_ = type_id(view.object_id)
            if self._scope.entity_types is not None and not (
                subject in self._scope.entity_types
                or object_ in self._scope.entity_types
            ):
                continue
            if (
                subject in self._scope.exclude_entity_types
                or object_ in self._scope.exclude_entity_types
            ):
                continue
            if (
                self._scope.relation_types is not None
                and view.relation_type_id not in self._scope.relation_types
            ):
                continue
            if view.relation_type_id in self._scope.exclude_relation_types:
                continue
            result.append(view)
        return result

    async def admit_entities(self, ids: Sequence[str]) -> list[str]:
        """Return entity ids whose own type passes the entity filters."""
        if not ids or (
            self._scope.entity_types is None and not self._scope.exclude_entity_types
        ):
            return list(ids)
        entities = await self._entities.get_entities(ids, follow_merged=False)
        types = {entity.entity_id: entity.ontology_type_id for entity in entities}
        return [
            entity_id
            for entity_id in ids
            if (
                self._scope.entity_types is None
                or types.get(entity_id) in self._scope.entity_types
            )
            and types.get(entity_id) not in self._scope.exclude_entity_types
        ]
