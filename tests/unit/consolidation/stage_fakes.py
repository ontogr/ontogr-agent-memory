"""In-memory ports for testing consolidation stage behavior without databases."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from ontogr_mem.consolidation.models import (
    ArchiveRecord,
    Proposal,
    SessionEntry,
    Watermark,
)
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.temporal.query import FactView


NOW = datetime(2030, 1, 1, tzinfo=UTC)


class MemoryStore:
    """Store consolidation documents by model and their public identifier."""

    def __init__(self, *documents: BaseModel) -> None:
        """Create the store with the given documents."""
        self.documents: dict[tuple[type[BaseModel], str], BaseModel] = {}
        for document in documents:
            self._save(document)

    async def list(self, model: type[Any]) -> builtins.list[Any]:
        """Return documents of ``model`` ordered by their identifier."""
        return [
            document
            for (document_model, _), document in sorted(
                self.documents.items(), key=lambda item: item[0][1]
            )
            if document_model is model
        ]

    async def get(self, model: type[Any], document_id: str) -> Any | None:
        """Return one document by its public identifier."""
        return self.documents.get((model, document_id))

    async def put(self, document: BaseModel) -> None:
        """Store one document by its public identifier."""
        self._save(document)

    async def find(
        self, model: type[Any], field: str, value: object
    ) -> builtins.list[Any]:
        """Return documents whose named field equals ``value``."""
        return [
            document
            for (document_model, _), document in self.documents.items()
            if document_model is model and getattr(document, field) == value
        ]

    def _save(self, document: BaseModel) -> None:
        if isinstance(document, Watermark):
            identifier = document.key
        elif isinstance(document, ArchiveRecord):
            identifier = document.target_id
        elif isinstance(document, Proposal):
            identifier = document.proposal_id
        elif isinstance(document, SessionEntry):
            identifier = document.entry_id
        else:
            raise TypeError(f"unsupported document: {type(document).__name__}")
        if not isinstance(identifier, str):
            raise TypeError(f"no identifier for {type(document).__name__}")
        self.documents[(type(document), identifier)] = document


class MemoryResolution:
    """In-memory entity port that records revision behavior."""

    def __init__(self, *entities: Entity) -> None:
        """Create the port with active entities."""
        self.entities = {entity.entity_id: entity for entity in entities}
        self.bump_revisions: list[bool] = []

    async def active_entities(self) -> list[Entity]:
        """Return active entities in id order."""
        return [
            self.entities[entity_id]
            for entity_id in sorted(self.entities)
            if self.entities[entity_id].status.value == "active"
        ]

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return stored entities that match ``ids``."""
        del follow_merged
        return [
            self.entities[entity_id] for entity_id in ids if entity_id in self.entities
        ]

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Store the entity and record the requested revision behavior."""
        self.bump_revisions.append(bump_revision)
        self.entities[entity.entity_id] = entity
        return entity


class MemoryTemporal:
    """Temporal port with a fixed set of current facts."""

    def __init__(self, facts: tuple[FactView, ...] = ()) -> None:
        """Create the port with current facts."""
        self.facts = facts

    async def current_facts(self) -> list[FactView]:
        """Return the current facts."""
        return list(self.facts)


def entity(entity_id: str, descriptions: list[str], *, stale: bool = True) -> Entity:
    """Build an active entity with the given descriptions."""
    return Entity(
        entity_id=entity_id,
        type_label="Thing",
        canonical_name=entity_id,
        strict_key=entity_id,
        loose_key=entity_id,
        descriptions=descriptions,
        summary_stale=stale,
        created_at=NOW,
        updated_at=NOW,
    )
