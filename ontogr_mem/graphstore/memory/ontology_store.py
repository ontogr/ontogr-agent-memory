"""In-memory twin of the ontology store."""

from collections.abc import Callable, Sequence

from ontogr_mem.ontology.errors import (
    MissingVersionError,
    ProposalDecidedError,
    VersionConflict,
)
from ontogr_mem.ontology.models import (
    ChangeSet,
    Ontology,
    Proposal,
    ProposalStatus,
    Unmatched,
)


class MemoryOntologyStore:
    """`OntologyStore` that keeps one tenant's ontology in memory.

    Args:
        ingested_episode_ids: Returns the ids of the fully ingested episodes.
    """

    def __init__(self, ingested_episode_ids: Callable[[], set[str]]) -> None:
        """Create a store holding only the genesis ontology."""
        genesis = Ontology.genesis()
        self._versions: dict[int, Ontology] = {0: genesis}
        self._head = 0
        self._change_sets: list[ChangeSet] = []
        self._unmatched: dict[str, list[Unmatched]] = {}
        self._proposals: dict[str, Proposal] = {}
        self._ingested = ingested_episode_ids

    async def get(self, version: int | None = None) -> Ontology:
        """Return a version of the ontology, or the head if `version` is None."""
        wanted = self._head if version is None else version
        found = self._versions.get(wanted)
        if found is None:
            msg = f"ontology version {wanted} does not exist"
            raise MissingVersionError(msg)
        return found

    async def commit(self, new: Ontology, change_set: ChangeSet) -> None:
        """Store `new` as the next version and move the head to it."""
        if change_set.base_version != self._head:
            raise VersionConflict(change_set.base_version)
        self._versions[new.version] = new
        self._change_sets.append(change_set)
        self._head = new.version

    async def history(self) -> Sequence[ChangeSet]:
        """Return every committed change set, oldest first."""
        return list(self._change_sets)

    async def replace_unmatched(
        self, episode_id: str, rows: Sequence[Unmatched]
    ) -> None:
        """Replace the unmatched rows of one episode."""
        by_id = {row.row_id: row for row in rows}
        if by_id:
            self._unmatched[episode_id] = list(by_id.values())
        else:
            self._unmatched.pop(episode_id, None)

    async def unmatched(self) -> Sequence[Unmatched]:
        """Return every stored unmatched row."""
        return [row for rows in self._unmatched.values() for row in rows]

    async def save_proposal(self, proposal: Proposal) -> None:
        """Store a proposal; a decided one is not overwritten by a pending save."""
        stored = self._proposals.get(proposal.proposal_id)
        if (
            stored is not None
            and stored.status is not ProposalStatus.PENDING
            and proposal.status is ProposalStatus.PENDING
        ):
            msg = f"proposal {proposal.proposal_id} is already {stored.status.value}"
            raise ProposalDecidedError(msg)
        self._proposals[proposal.proposal_id] = proposal

    async def get_proposal(self, proposal_id: str) -> Proposal | None:
        """Return the proposal, or None."""
        return self._proposals.get(proposal_id)

    async def proposals(
        self, status: ProposalStatus | None = None
    ) -> Sequence[Proposal]:
        """Return the proposals with `status`, or all of them, ordered by id."""
        return [
            p
            for _, p in sorted(self._proposals.items())
            if status is None or p.status is status
        ]

    async def known_episode_ids(self) -> set[str]:
        """Return the ids of the fully ingested episodes."""
        return set(self._ingested())
