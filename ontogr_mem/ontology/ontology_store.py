"""The ontology store port."""

from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.ontology.models import (
    ChangeSet,
    Ontology,
    Proposal,
    ProposalStatus,
    Unmatched,
)


class OntologyStore(Protocol):
    """Stores one tenant's ontology versions, unmatched rows and proposals."""

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

    async def replace_unmatched(
        self, episode_id: str, rows: Sequence[Unmatched]
    ) -> None:
        """Replace the unmatched rows of one episode. Other episodes are untouched."""
        ...

    async def unmatched(self) -> Sequence[Unmatched]:
        """Return every stored unmatched row."""
        ...

    async def save_proposal(self, proposal: Proposal) -> None:
        """Store a proposal.

        Raises:
            ProposalDecidedError: If a decided proposal with the same id exists
                and `proposal` is pending.
        """
        ...

    async def get_proposal(self, proposal_id: str) -> Proposal | None:
        """Return the proposal, or None."""
        ...

    async def proposals(
        self, status: ProposalStatus | None = None
    ) -> Sequence[Proposal]:
        """Return the proposals with `status`, or all of them, ordered by id."""
        ...

    async def known_episode_ids(self) -> set[str]:
        """Return the ids of the fully ingested episodes of the tenant."""
        ...
