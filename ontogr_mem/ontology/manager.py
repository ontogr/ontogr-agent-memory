"""The ontology manager, which combines the store with the pure logic."""

from collections.abc import Sequence

from ontogr_mem.data_models import Clock
from ontogr_mem.ontology.apply import apply_changeset, check_catalog
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import ChangeSet, Channel, Ontology
from ontogr_mem.ontology.ontology_store import OntologyStore
from ontogr_mem.ontology.projection import Projection, project
from ontogr_mem.ontology.validate import ExtractedItems, ValidationReport, validate


class OntologyManager:
    """Owns a tenant's ontology and its history.

    Args:
        store: Storage for the tenant's ontology.
        clock: Source of the time stamped on change sets.
    """

    def __init__(self, store: OntologyStore, clock: Clock) -> None:
        """Wire the store and the clock."""
        self._store = store
        self._clock = clock

    async def catalog(self, version: int | None = None) -> Ontology:
        """Return a snapshot of the ontology; the head if `version` is None."""
        return await self._store.get(version)

    async def history(self) -> Sequence[ChangeSet]:
        """Return every committed change set, oldest first."""
        return await self._store.history()

    async def extraction_view(self) -> Projection:
        """Return the dense-id view of the head that extraction reads."""
        return project(await self._store.get())

    async def apply(self, change_set: ChangeSet) -> Ontology:
        """Check `change_set` against the head and commit it.

        Raises:
            InvalidChange: If the result breaks a rule.
            VersionConflict: If the head moved since `change_set.base_version`.
                Re-read the catalog and apply again.
        """
        head = await self._store.get()
        stamped = self._stamp(change_set)
        new = apply_changeset(head, stamped)
        await self._store.commit(new, stamped)
        return new

    async def load_catalog(self, catalog: Ontology) -> Ontology:
        """Load a whole user-built catalog into a tenant still at version 0.

        Types are matched by label, never by the ids in `catalog`.

        Raises:
            InvalidChange: If the tenant already has a catalog or `catalog` breaks
                a rule.
        """
        head = await self._store.get()
        if head.version != 0:
            raise InvalidChange(
                "tenant already has a catalog",
                hint="load_catalog is for a new tenant; use apply",
            )
        check_catalog(catalog)
        shaped = catalog.model_copy(
            update={
                "entities": sorted(
                    catalog.entities,
                    key=lambda e: (len(catalog.ancestors(e.type_id)), e.type_id),
                )
            }
        )
        change_set = diff(head, shaped).model_copy(
            update={
                "actor": "load",
                "reason": "load catalog",
                "channel": Channel.IMPORT,
            }
        )
        return await self.apply(change_set)

    async def validate_extraction(self, items: ExtractedItems) -> ValidationReport:
        """Check one episode's extraction against the head of the ontology."""
        return validate(await self._store.get(), items)

    def _stamp(self, change_set: ChangeSet) -> ChangeSet:
        if change_set.created_at is not None:
            return change_set
        return change_set.model_copy(update={"created_at": self._clock.now()})
