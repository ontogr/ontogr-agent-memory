"""Graph-backed temporal store implementation.

A slot node is the compare-and-swap target. Each observation links to its slot.
Every fact version is a `FACT` edge between two canonical `Entity` nodes, so
the versions of one fact share their endpoints. Times are integer
microseconds, and the open end is stored as the open-end sentinel.

A slot commit is one statement. The statement checks that every entity and
expired version exists before creating a slot. It then locks the slot and checks
the revision before writing. A refused commit therefore writes nothing, on
either server.
"""

from collections.abc import Sequence

from ontogr_mem.data_models.fact_versions import Fact, Supersession
from ontogr_mem.data_models.ledger import LedgerEntry
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reconcile import Conflict
from ontogr_mem.temporal.temporal_store import (
    EpisodeState,
    IncompleteEpisode,
    SlotCommit,
    SlotState,
    SourceSelector,
    TemporalSnapshot,
)
from ontogr_mem.temporal.verdicts import ConflictDecision, FactVerdict

from .episode_state import EpisodeStateStore
from .ledger_store import LedgerStore
from .pending import pending_slots
from .slots import SlotStore
from .snapshot import SnapshotReader, snapshot
from .verdict_store import VerdictStore


class GraphTemporalStore:
    """`TemporalStore` over one tenant's graph.

    Args:
        store: Graph store that owns the tenant's graph.
        handle: Handle of the tenant's graph.
        tenant_keyed: Override the runner's backend scope when set.
    """

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        *,
        tenant_keyed: bool | None = None,
    ) -> None:
        """Bind the store to a tenant's graph."""
        runner = store.runner(handle)
        scope = (
            TenantScope.from_mode(tenant_keyed)
            if tenant_keyed is not None
            else runner.scope
        )
        self._reads = SnapshotReader(runner, scope)
        self._slots = SlotStore(runner, scope, handle.tenant, self._reads)
        self._episodes = EpisodeStateStore(runner, scope)
        self._ledger = LedgerStore(store, handle, runner, scope)
        self._verdicts = VerdictStore(runner, scope, handle.tenant)

    async def load_slot(self, slot_key: str) -> SlotState:
        """Return the slot's revision, ledger head and every fact version."""
        return await self._slots.load_slot(slot_key)

    async def append_observations(self, observations: Sequence[Observation]) -> int:
        """Write new observations and create the slots they mention."""
        return await self._slots.append_observations(observations)

    async def commit_slot(self, commit: SlotCommit) -> int:
        """Write a slot commit atomically and return the new revision."""
        return await self._slots.commit_slot(commit)

    async def query(self, query: TemporalQuery) -> list[FactView]:
        """Return the fact versions that satisfy `query`, ordered by fact id."""
        return await self._reads.query(query)

    async def facts_of_episode(self, episode_id: str) -> list[Fact]:
        """Return every fact version in a slot that the episode touched."""
        return await self._reads.facts_of_episode(episode_id)

    async def observations_of_episode(self, episode_id: str) -> list[Observation]:
        """Return the episode's observations, ordered by observation id."""
        return await self._reads.observations_of_episode(episode_id)

    async def observations_of_slot(self, slot_key: str) -> list[Observation]:
        """Return the slot's observations, ordered by observation id."""
        return await self._reads.observations_of_slot(slot_key)

    async def supersessions_of_slot(self, slot_key: str) -> list[Supersession]:
        """Return the audit rows of a slot, ordered by id."""
        return await self._reads.supersessions_of_slot(slot_key)

    async def ledger_entries(self, chain: str) -> list[LedgerEntry]:
        """Return the entries of a chain, ordered by sequence number."""
        return await self._ledger.ledger_entries(chain)

    async def slot_keys(self) -> list[str]:
        """Return every slot key, sorted."""
        return await self._reads.slot_keys()

    async def episode_state(self, episode_id: str) -> EpisodeState | None:
        """Return the temporal state of the episode, or ``None`` if it has none."""
        return await self._episodes.episode_state(episode_id)

    async def set_episode_pending(self, episode_id: str) -> None:
        """Mark the episode as having observations in flight."""
        return await self._episodes.set_episode_pending(episode_id)

    async def mark_committed(self, episode_id: str, content_hash: str) -> None:
        """Mark the episode committed with the content hash that was read."""
        return await self._episodes.mark_committed(episode_id, content_hash)

    async def mark_retracted(self, episode_id: str) -> None:
        """Mark the episode retracted; its node stays.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        return await self._episodes.mark_retracted(episode_id)

    async def clear_episode_state(self, episode_id: str) -> None:
        """Clear the episode's temporal status and committed hash.

        The episode node and its content hash stay as they are.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        return await self._episodes.clear_episode_state(episode_id)

    async def remove_episode_observations(self, episode_id: str) -> set[str]:
        """Delete the episode's observations and the audit rows they caused.

        Returns:
            The slot keys the removed observations were in.
        """
        return await self._slots.remove_episode_observations(episode_id)

    async def scrub_episode(self, episode_id: str) -> int:
        """Blank the text copied from the episode into verdicts.

        Returns:
            The number of verdicts changed.
        """
        return await self._verdicts.scrub_episode(episode_id)

    async def episode_has_observations(self, episode_id: str) -> bool:
        """True when the episode already has observations."""
        return await self._reads.episode_has_observations(episode_id)

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return every episode in ``PENDING`` state, sorted by episode id."""
        return await self._episodes.incomplete_episodes()

    async def episodes_matching(self, selector: SourceSelector) -> list[str]:
        """Return the ids of episodes whose source matches, sorted."""
        return await self._episodes.episodes_matching(selector)

    async def ledger_head(self, chain: str) -> tuple[int, str]:
        """Return the last ``(seq, checksum)`` of a standalone chain."""
        return await self._ledger.ledger_head(chain)

    async def append_ledger(
        self, entries: Sequence[LedgerEntry], base_seq: int
    ) -> bool:
        """Append entries to a standalone chain when its head still has `base_seq`."""
        return await self._ledger.append_ledger(entries, base_seq)

    async def put_verdicts(self, verdicts: Sequence[FactVerdict]) -> None:
        """Store verdicts, replacing any row with the same verdict id."""
        return await self._verdicts.put_verdicts(verdicts)

    async def verdicts_for(self, fact_ids: Sequence[str]) -> list[FactVerdict]:
        """Return the verdicts that name any of the facts on either side."""
        return await self._verdicts.verdicts_for(fact_ids)

    async def put_decision(self, decision: ConflictDecision) -> None:
        """Store a decision, replacing any row with the same decision id."""
        return await self._verdicts.put_decision(decision)

    async def decisions_for(self, fact_ids: Sequence[str]) -> list[ConflictDecision]:
        """Return the decisions that name any of the given facts."""
        return await self._verdicts.decisions_for(fact_ids)

    async def conflicts_for(
        self, slot_keys: Sequence[str], fact_ids: Sequence[str]
    ) -> list[Conflict]:
        """Return the stored conflicts of the slots or that name the facts."""
        return await self._reads.conflicts_for(slot_keys, fact_ids)

    async def current_facts(self) -> list[Fact]:
        """Return the current version of every fact in the tenant."""
        return await self._reads.current_facts()

    async def facts_touching(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return the current versions with an entity in ``entity_ids`` on a side."""
        return await self._reads.facts_touching(entity_ids)

    async def facts_by_ids(self, fact_ids: Sequence[str]) -> list[Fact]:
        """Return every version of each fact, ordered by fact id and version."""
        return await self._reads.facts_by_ids(fact_ids)

    async def facts_of_entities(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return every version of every fact with an entity on a side."""
        return await self._reads.facts_of_entities(entity_ids)

    async def observations_by_ids(self, obs_ids: Sequence[str]) -> list[Observation]:
        """Return the observations, ordered by observation id."""
        return await self._reads.observations_by_ids(obs_ids)

    async def pending_slots(self) -> list[str]:
        """Return every pending slot key in the tenant, sorted.

        A slot is pending when one of its current facts shares an entity with
        a fact of another slot and that pair has no verdict. The answer is
        derived from the facts and the verdicts, so it never goes stale.
        """
        return await pending_slots(self)

    async def snapshot(self) -> TemporalSnapshot:
        """Return every temporal row of the tenant."""
        return await snapshot(self._reads, self._verdicts)
