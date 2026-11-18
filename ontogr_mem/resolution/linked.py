"""A resolver with the type links of one ontology snapshot bound."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.ontology.links import TypeLinks
from ontogr_mem.resolution.config import Budget
from ontogr_mem.resolution.deduplication import ConsolidationReport


if TYPE_CHECKING:
    from ontogr_mem.resolution.resolver import EntityResolver


class LinkedResolver:
    """Entity resolution as consolidation stages see it, for one snapshot."""

    def __init__(self, resolver: "EntityResolver", links: TypeLinks) -> None:
        """Bind the ontology links to a resolver."""
        self._resolver = resolver
        self._links = links

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        return await self._resolver.active_entities()

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return entities by id; merged ones may be followed."""
        return await self._resolver.get_entities(ids, follow_merged=follow_merged)

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity."""
        return await self._resolver.save_entity(entity, bump_revision=bump_revision)

    async def archive(self, entity_id: str) -> None:
        """Mark an entity archived."""
        await self._resolver.archive(entity_id)

    async def delete(self, entity_id: str) -> None:
        """Delete an entity and its vector."""
        await self._resolver.delete(entity_id)

    async def deduplicate(
        self, *, plan_only: bool = False, budget: Budget | None = None
    ) -> ConsolidationReport:
        """Run one deduplication pass with the bound links."""
        return await self._resolver.deduplicate(
            self._links, plan_only=plan_only, budget=budget
        )
