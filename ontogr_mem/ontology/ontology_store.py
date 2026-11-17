"""The ontology store port."""

from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.ontology.models import (
    ChangeSet,
    Ontology,
)


class OntologyStore(Protocol):
    """Stores one tenant's ontology versions."""

    async def get(self, version: int | None = None) -> Ontology:
        """Return a version of the ontology, or the head if `version` is None.

        The first access of a new tenant creates the genesis ontology, once, even
        under concurrent first access.

        Raises:
            MissingVersionError: If the version does not exist.
        """
        ...

    async def commit(self, new: Ontology, change_set: ChangeSet) -> None:
        """Store `new` as the next version and move the head to it.

        Raises:
            VersionConflict: If the head is no longer `change_set.base_version`.
                Nothing is written.
        """
        ...

    async def history(self) -> Sequence[ChangeSet]:
        """Return every committed change set, oldest first."""
        ...
