"""The storage port of temporal reconciliation.

A slot is the unit of concurrency. Its revision guards the commit, so a writer
that read a stale state is refused instead of overwriting a newer fact. Each
standalone chain, such as the tenant chain, has its own head that is compared
on append.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ontogr_mem.data_models.episode import TemporalStatus
from ontogr_mem.data_models.fact_versions import Fact, Supersession
from ontogr_mem.data_models.ledger import LedgerEntry
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.ledger import GENESIS
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reconcile import Conflict
from ontogr_mem.temporal.verdicts import ConflictDecision, FactVerdict


@dataclass(frozen=True, slots=True)
class SlotState:
    """A slot as a writer read it.

    Attributes:
        slot_key: Slot identity.
        revision: Commit counter, starting at 0.
        ledger_seq: Sequence number of the last ledger entry of the slot.
        ledger_head: Checksum of the last ledger entry, or ``GENESIS``.
        facts: Every fact version of the slot, ordered by fact id and version.
        conflicts: Pair conflicts the last commit stored for the slot.
    """

    slot_key: str
    revision: int = 0
    ledger_seq: int = 0
    ledger_head: str = GENESIS
    facts: tuple[Fact, ...] = ()
    conflicts: tuple[Conflict, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeState:
    """Temporal processing state of one episode.

    Attributes:
        status: Where the episode stands in reconciliation.
        committed_hash: Content hash recorded with the commit, or ``None``.
    """

    status: TemporalStatus
    committed_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SlotCommit:
    """Everything one slot commit writes, all or nothing.

    Attributes:
        slot_key: Slot being committed.
        base_revision: Revision the writer read; a newer one refuses the write.
        new_facts: Fact versions to write.
        expired_facts: Fact versions the commit closes.
        supersessions: Audit rows for the closes.
        ledger: Ledger entries to append to the slot chain.
        conflicts: The slot's full set of pair conflicts after the commit. The
            store writes it when conflicts_changed is True.
        conflicts_changed: True when that set differs from the one the slot
            holds now. A commit that leaves the set alone leaves it False, and
            the store keeps what it has.
    """

    slot_key: str
    base_revision: int
    new_facts: tuple[Fact, ...] = ()
    expired_facts: tuple[Fact, ...] = ()
    supersessions: tuple[Supersession, ...] = ()
    ledger: tuple[LedgerEntry, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    conflicts_changed: bool = False

    @property
    def empty(self) -> bool:
        """True when the commit writes no fact, ledger row or conflict set."""
        return not (
            self.new_facts
            or self.expired_facts
            or self.supersessions
            or self.ledger
            or self.conflicts_changed
        )

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Ids that must exist before any fact is written, sorted."""
        return tuple(
            sorted(
                {
                    entity_id
                    for fact in self.new_facts
                    for entity_id in (fact.subject_id, fact.object_id)
                }
            )
        )


@dataclass(frozen=True, slots=True)
class SourceSelector:
    """Episodes of one source, matched on the source node.

    Attributes:
        system: Source system to match.
        ref: Item id in the source system, or ``None`` for any item.
        connector: Connector that mapped the item, or ``None`` for any.
    """

    system: str
    ref: str | None = None
    connector: str | None = None


@dataclass(frozen=True, slots=True)
class IncompleteEpisode:
    """An episode left ``PENDING`` by an interrupted ingest.

    Attributes:
        episode_id: Episode that never reached its commit marker.
        ingested_at: When the episode entered the system.
        content_hash: Content hash the episode was sealed with, or ``None``.
    """

    episode_id: str
    ingested_at: datetime
    content_hash: str | None


@dataclass(frozen=True, slots=True)
class TemporalSnapshot:
    """Every temporal row of the tenant, read at one point for the audit.

    Attributes:
        facts: Every fact version, current, expired and withdrawn.
        observations: Every observation still stored.
        supersessions: Every audit row of every slot.
        verdicts: Every stored verdict.
        decisions: Every stored decision.
        conflicts: Every stored pair conflict.
        slot_keys: Every slot key in the tenant, sorted.
    """

    facts: tuple[Fact, ...]
    observations: tuple[Observation, ...]
    supersessions: tuple[Supersession, ...]
    verdicts: tuple[FactVerdict, ...]
    decisions: tuple[ConflictDecision, ...]
    conflicts: tuple[Conflict, ...]
    slot_keys: tuple[str, ...]


class TemporalStore(Protocol):
    """Storage verbs temporal reconciliation needs."""

    async def load_slot(self, slot_key: str) -> SlotState:
        """Return the slot's revision, ledger head and every fact version.

        A slot that was never written reads as revision 0 with no facts.
        """
        ...

    async def append_observations(self, observations: Sequence[Observation]) -> int:
        """Write observations idempotently and return the number inserted.

        An observation whose id is already stored is skipped.
        """
        ...

    async def commit_slot(self, commit: SlotCommit) -> int:
        """Write a slot commit atomically and return the new revision.

        An empty commit writes nothing and returns its base revision.

        Raises:
            RevisionConflict: The slot moved since the writer read it.
            MissingEntityError: A fact names an entity that has no node.
        """
        ...

    async def query(self, query: TemporalQuery) -> list[FactView]:
        """Return the fact versions that satisfy `query`, ordered by fact id."""
        ...

    async def facts_of_episode(self, episode_id: str) -> list[Fact]:
        """Return every fact version in a slot that the episode touched.

        The episode's observations are found through their episode id, then
        their slots, then the fact edges of those slots.
        """
        ...

    async def observations_of_episode(self, episode_id: str) -> list[Observation]:
        """Return the episode's observations, ordered by observation id."""
        ...

    async def observations_of_slot(self, slot_key: str) -> list[Observation]:
        """Return the slot's observations, ordered by observation id."""
        ...

    async def supersessions_of_slot(self, slot_key: str) -> list[Supersession]:
        """Return the audit rows of a slot, ordered by supersession id."""
        ...

    async def ledger_entries(self, chain: str) -> list[LedgerEntry]:
        """Return the entries of a chain, ordered by sequence number."""
        ...

    async def slot_keys(self) -> list[str]:
        """Return every slot key in the tenant, sorted."""
        ...

    async def episode_state(self, episode_id: str) -> EpisodeState | None:
        """Return the temporal state of the episode, or ``None`` if it has none."""
        ...

    async def set_episode_pending(self, episode_id: str) -> None:
        """Mark the episode as having observations in flight.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        ...

    async def mark_committed(self, episode_id: str, content_hash: str) -> None:
        """Mark the episode committed with the content hash that was read.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        ...

    async def episode_has_observations(self, episode_id: str) -> bool:
        """True when the episode already has observations."""
        ...

    async def ledger_head(self, chain: str) -> tuple[int, str]:
        """Return the last ``(seq, checksum)`` of a standalone chain.

        Slot chains advance inside `commit_slot` and are read with `load_slot`.
        """
        ...

    async def put_verdicts(self, verdicts: Sequence[FactVerdict]) -> None:
        """Store verdicts, replacing any row with the same verdict id."""
        ...

    async def verdicts_for(self, fact_ids: Sequence[str]) -> list[FactVerdict]:
        """Return the verdicts that name any of the facts on either side."""
        ...

    async def put_decision(self, decision: ConflictDecision) -> None:
        """Store a decision, replacing any row with the same decision id."""
        ...

    async def decisions_for(self, fact_ids: Sequence[str]) -> list[ConflictDecision]:
        """Return the decisions that name any of the given facts."""
        ...

    async def conflicts_for(
        self, slot_keys: Sequence[str], fact_ids: Sequence[str]
    ) -> list[Conflict]:
        """Return the stored conflicts of the slots or that name the facts."""
        ...

    async def current_facts(self) -> list[Fact]:
        """Return the current version of every fact in the tenant."""
        ...

    async def facts_touching(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return the current versions with an entity in ``entity_ids`` on a side."""
        ...

    async def facts_by_ids(self, fact_ids: Sequence[str]) -> list[Fact]:
        """Return every version of each fact, ordered by fact id and version.

        Every status is returned, and no time rule applies. Fact ids that are
        not stored are left out.
        """
        ...

    async def facts_of_entities(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return every version of every fact with an entity on a side.

        An entity on the subject side or on the object side matches. Every
        status is returned, and no time rule applies.
        """
        ...

    async def observations_by_ids(self, obs_ids: Sequence[str]) -> list[Observation]:
        """Return the observations, ordered by observation id.

        Observation ids that are not stored are left out.
        """
        ...

    async def pending_slots(self) -> list[str]:
        """Return every pending slot key in the tenant, sorted.

        A slot is pending when one of its current facts shares an entity with
        a fact of another slot and that pair has no verdict. The answer is
        derived from the facts and the verdicts, so it never goes stale.
        """
        ...

    async def append_ledger(
        self, entries: Sequence[LedgerEntry], base_seq: int
    ) -> bool:
        """Append entries to a standalone chain when its head still has `base_seq`.

        Returns:
            True when the entries were appended, False when the head moved.
        """
        ...

    async def remove_episode_observations(self, episode_id: str) -> set[str]:
        """Delete the episode's observations and the audit rows they caused.

        Returns:
            The slot keys the removed observations were in, as a set.
        """
        ...

    async def scrub_episode(self, episode_id: str) -> int:
        """Blank the text copied from the episode into verdicts.

        Returns:
            The number of verdicts changed.
        """
        ...

    async def mark_retracted(self, episode_id: str) -> None:
        """Mark the episode retracted; its node stays.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        ...

    async def clear_episode_state(self, episode_id: str) -> None:
        """Clear the episode's temporal status and committed hash.

        The episode node and its content hash stay as they are.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        ...

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return every episode in ``PENDING`` state, sorted by episode id."""
        ...

    async def episodes_matching(self, selector: SourceSelector) -> list[str]:
        """Return the ids of episodes whose source matches, sorted."""
        ...

    async def snapshot(self) -> TemporalSnapshot:
        """Return every temporal row of the tenant."""
        ...
