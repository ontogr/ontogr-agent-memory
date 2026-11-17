"""The ontology manager, which combines the store with the pure logic."""

from collections.abc import Sequence

from ontogr_mem.data_models import Clock
from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.declaration import OntologyDeclaration, check_declaration
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.errors import VersionConflict
from ontogr_mem.ontology.models import ChangeSet, Channel, Ontology
from ontogr_mem.ontology.ontology_store import OntologyStore
from ontogr_mem.ontology.projection import Projection, project
from ontogr_mem.ontology.validate import ExtractedItems, ValidationReport, validate


_DECLARE_ATTEMPTS = 3


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

    async def declare(
        self, declaration: OntologyDeclaration, *, only_if_empty: bool = False
    ) -> Ontology:
        """Apply a declaration and return the head, retrying up to three times.

        Args:
            declaration: Desired ontology.
            only_if_empty: Keep any ontology that already has a committed version.

        Raises:
            InvalidChange: The declaration does not apply to the stored ontology.
            VersionConflict: Another writer won every attempt.
        """
        check_declaration(declaration)
        for _ in range(_DECLARE_ATTEMPTS - 1):
            try:
                return await self._declare_once(declaration, only_if_empty)
            except VersionConflict:
                continue
        return await self._declare_once(declaration, only_if_empty)

    async def _declare_once(
        self, declaration: OntologyDeclaration, only_if_empty: bool
    ) -> Ontology:
        head = await self._store.get()
        if only_if_empty and head.version != 0:
            return head
        change_set = diff(head, build_target(head, declaration))
        if not change_set.ops:
            return head
        return await self.apply(
            change_set.model_copy(
                update={
                    "actor": "declare",
                    "reason": "declare ontology",
                    "channel": Channel.IMPORT,
                }
            )
        )

    async def validate_extraction(self, items: ExtractedItems) -> ValidationReport:
        """Check one episode's extraction against the head of the ontology."""
        return validate(await self._store.get(), items)

    def _stamp(self, change_set: ChangeSet) -> ChangeSet:
        if change_set.created_at is not None:
            return change_set
        return change_set.model_copy(update={"created_at": self._clock.now()})
