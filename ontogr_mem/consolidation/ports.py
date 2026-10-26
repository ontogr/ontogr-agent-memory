"""What the consolidation stages need from the components they run over.

``EntityResolver`` and ``TemporalEngine`` satisfy these protocols. The stages reach
resolution and temporal state only through these verbs, never through a store.
"""

from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.resolution.config import Budget
from ontogr_mem.resolution.deduplication import ConsolidationReport
from ontogr_mem.temporal.query import FactView
from ontogr_mem.temporal.report import TemporalReport
from ontogr_mem.temporal.temporal_store import IncompleteEpisode


class ResolutionPort(Protocol):
    """Entity resolution as the consolidation stages see it."""

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        ...

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return the entities with these ids; merged ones may be followed."""
        ...

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity; ``bump_revision=False`` leaves its revision as is."""
        ...

    async def archive(self, entity_id: str) -> None:
        """Mark an entity archived without a revision bump."""
        ...

    async def delete(self, entity_id: str) -> None:
        """Delete an entity and its vector.

        Raises:
            EntityHasFactsError: If the entity still has a live fact edge.
        """
        ...

    async def deduplicate(
        self, *, plan_only: bool = False, budget: Budget | None = None
    ) -> ConsolidationReport:
        """Run one deduplication pass over the active entities."""
        ...


class TemporalPort(Protocol):
    """Temporal state as the consolidation stages see it."""

    async def current_facts(self) -> list[FactView]:
        """Return the current version of every fact, with statement and disputes."""
        ...

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return the episodes left pending by an interrupted ingest."""
        ...

    async def recover(self) -> TemporalReport:
        """Finish the pending episodes and report what could not be finished."""
        ...
