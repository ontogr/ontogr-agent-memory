"""In-memory twin of the temporal store.

The twin keeps the same shape as the graph adapters: slots guard revisions,
observations hang off their episode, and fact versions connect entity ids.
Episode temporal fields live in the episode store the twin is given, so the
two stores agree on what an episode's state is. One instance holds one tenant.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.data_models.episode import Episode, TemporalStatus
from ontogr_mem.data_models.fact_versions import Fact, FactStatus, Supersession
from ontogr_mem.data_models.ledger import LedgerEntry
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.graphstore.memory.episode_store import MemoryEpisodeStore
from ontogr_mem.graphstore.temporal.pending import pending_slots
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.temporal.errors import (
    MissingEntityError,
    MissingEpisodeError,
    MissingFactVersionError,
    RevisionConflict,
)
from ontogr_mem.temporal.ledger import GENESIS
from ontogr_mem.temporal.query import FactView, TemporalQuery, matches, view_of
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


class MemoryTemporalStore:
    """`TemporalStore` that keeps one tenant's temporal rows in memory.

    Args:
        episodes: Episode store of the same tenant. Temporal state of an episode
            is read from and written to it.
        resolver: Entity store of the same tenant. A commit naming an entity
            absent from it is refused.
    """

    def __init__(self, episodes: MemoryEpisodeStore, resolver: ResolverStore) -> None:
        """Create an empty store bound to the episodes of one tenant."""
        self._episodes = episodes
        self._resolver = resolver
        self._slots: dict[str, SlotState] = {}
        self._observations: dict[str, Observation] = {}
        self._facts: dict[tuple[str, int], Fact] = {}
        self._supersessions: dict[str, Supersession] = {}
        self._ledger: dict[str, list[LedgerEntry]] = {}
        self._chain_heads: dict[str, tuple[int, str]] = {}
        self._verdicts: dict[str, FactVerdict] = {}
        self._decisions: dict[str, ConflictDecision] = {}
        self._conflicts: dict[str, Conflict] = {}
        self._lock = asyncio.Lock()

    async def load_slot(self, slot_key: str) -> SlotState:
        """Return the slot's revision, ledger head, fact versions and conflicts."""
        state = self._slots.get(slot_key)
        if state is None:
            return SlotState(slot_key)
        return SlotState(
            slot_key=slot_key,
            revision=state.revision,
            ledger_seq=state.ledger_seq,
            ledger_head=state.ledger_head,
            facts=self._facts_of(slot_key),
            conflicts=self._conflicts_of(slot_key),
        )

    async def append_observations(self, observations: Sequence[Observation]) -> int:
        """Write new observations and create the slots they mention."""
        inserted = 0
        for observation in observations:
            if observation.obs_id in self._observations:
                continue
            self._observations[observation.obs_id] = observation
            self._slots.setdefault(
                observation.slot_key, SlotState(observation.slot_key)
            )
            inserted += 1
        return inserted

    async def commit_slot(self, commit: SlotCommit) -> int:
        """Write a slot commit atomically and return the new revision."""
        if commit.empty:
            return commit.base_revision
        async with self._lock:
            state = self._slots.get(commit.slot_key, SlotState(commit.slot_key))
            if state.revision != commit.base_revision:
                raise RevisionConflict(commit.slot_key, commit.base_revision)
            for entity_id in commit.entity_ids:
                if not await self._resolver.has_entity(entity_id):
                    raise MissingEntityError(entity_id)
            for fact in commit.expired_facts:
                if (fact.fact_id, fact.version) not in self._facts:
                    raise MissingFactVersionError(fact.fact_id, fact.version)

            for fact in commit.expired_facts:
                self._facts[(fact.fact_id, fact.version)] = fact
            for fact in commit.new_facts:
                self._facts[(fact.fact_id, fact.version)] = fact
            if commit.conflicts_changed:
                self._replace_conflicts(commit.slot_key, commit.conflicts)
            for row in commit.supersessions:
                self._supersessions[row.supersession_id] = row
            entries = sorted(commit.ledger, key=lambda entry: entry.seq)
            self._ledger.setdefault(commit.slot_key, []).extend(entries)
            ledger_seq, ledger_head = state.ledger_seq, state.ledger_head
            if entries:
                ledger_seq, ledger_head = entries[-1].seq, entries[-1].checksum
            revision = state.revision + 1
            self._slots[commit.slot_key] = SlotState(
                slot_key=commit.slot_key,
                revision=revision,
                ledger_seq=ledger_seq,
                ledger_head=ledger_head,
            )
            return revision

    async def query(self, query: TemporalQuery) -> list[FactView]:
        """Return the fact versions that satisfy `query`, ordered by fact id."""
        return [view_of(fact) for fact in self._sorted_facts() if matches(query, fact)]

    async def facts_of_episode(self, episode_id: str) -> list[Fact]:
        """Return every fact version in a slot that the episode touched."""
        slots = {
            observation.slot_key
            for observation in self._observations.values()
            if observation.episode_id == episode_id
        }
        return [fact for fact in self._sorted_facts() if fact.slot_key in slots]

    async def observations_of_episode(self, episode_id: str) -> list[Observation]:
        """Return the episode's observations, ordered by id."""
        return [
            observation
            for _, observation in sorted(self._observations.items())
            if observation.episode_id == episode_id
        ]

    async def observations_of_slot(self, slot_key: str) -> list[Observation]:
        """Return the slot's observations, ordered by id."""
        return [
            observation
            for _, observation in sorted(self._observations.items())
            if observation.slot_key == slot_key
        ]

    async def supersessions_of_slot(self, slot_key: str) -> list[Supersession]:
        """Return the audit rows of a slot, ordered by id."""
        return [
            row
            for _, row in sorted(self._supersessions.items())
            if row.slot_key == slot_key
        ]

    async def ledger_entries(self, chain: str) -> list[LedgerEntry]:
        """Return the entries of a chain, ordered by sequence number."""
        return sorted(self._ledger.get(chain, []), key=lambda entry: entry.seq)

    async def slot_keys(self) -> list[str]:
        """Return every slot key, sorted."""
        return sorted(self._slots)

    async def episode_state(self, episode_id: str) -> EpisodeState | None:
        """Return the temporal state of the episode, or ``None`` if it has none."""
        episode = await self._episodes.get_episode(episode_id)
        if episode is None or episode.temporal_status is None:
            return None
        return EpisodeState(episode.temporal_status, episode.temporal_committed_hash)

    async def set_episode_pending(self, episode_id: str) -> None:
        """Mark the episode as having observations in flight."""
        episode = await self._require_episode(episode_id)
        self._episodes.set_temporal_state(
            episode_id, TemporalStatus.PENDING, episode.temporal_committed_hash
        )

    async def mark_committed(self, episode_id: str, content_hash: str) -> None:
        """Mark the episode committed with the content hash that was read."""
        await self._require_episode(episode_id)
        self._episodes.set_temporal_state(
            episode_id, TemporalStatus.COMMITTED, content_hash
        )

    async def episode_has_observations(self, episode_id: str) -> bool:
        """Return True when the episode already has observations."""
        return any(o.episode_id == episode_id for o in self._observations.values())

    async def ledger_head(self, chain: str) -> tuple[int, str]:
        """Return the last ``(seq, checksum)`` of a standalone chain."""
        return self._chain_heads.get(chain, (0, GENESIS))

    async def append_ledger(
        self, entries: Sequence[LedgerEntry], base_seq: int
    ) -> bool:
        """Append entries to a standalone chain when its head still has `base_seq`."""
        if not entries:
            return True
        async with self._lock:
            chain = entries[0].chain
            head_seq, _ = await self.ledger_head(chain)
            if head_seq != base_seq:
                return False
            ordered = sorted(entries, key=lambda entry: entry.seq)
            self._ledger.setdefault(chain, []).extend(ordered)
            self._chain_heads[chain] = (ordered[-1].seq, ordered[-1].checksum)
            return True

    async def remove_episode_observations(self, episode_id: str) -> set[str]:
        """Delete the episode's observations and the audit rows they caused."""
        doomed = [o for o in self._observations.values() if o.episode_id == episode_id]
        for observation in doomed:
            del self._observations[observation.obs_id]
        self._supersessions = {
            key: row
            for key, row in self._supersessions.items()
            if row.source_episode_id != episode_id
        }
        return {o.slot_key for o in doomed}

    async def scrub_episode(self, episode_id: str) -> int:
        """Blank the text copied from the episode into verdicts."""
        changed = 0
        for key, verdict in list(self._verdicts.items()):
            if episode_id not in verdict.source_episode_ids:
                continue
            self._verdicts[key] = replace(
                verdict, reason_text="", quote_a="", quote_b=""
            )
            changed += 1
        return changed

    async def mark_retracted(self, episode_id: str) -> None:
        """Mark the episode retracted and keep its committed hash."""
        episode = await self._require_episode(episode_id)
        self._episodes.set_temporal_state(
            episode_id, TemporalStatus.RETRACTED, episode.temporal_committed_hash
        )

    async def clear_episode_state(self, episode_id: str) -> None:
        """Clear the episode's temporal status and committed hash."""
        await self._require_episode(episode_id)
        self._episodes.set_temporal_state(episode_id, None, None)

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return every episode in ``PENDING`` state, sorted by episode id."""
        return [
            IncompleteEpisode(episode.id, episode.ingested_at, episode.content_hash)
            for episode in self._episodes.all_episodes()
            if episode.temporal_status is TemporalStatus.PENDING
        ]

    async def episodes_matching(self, selector: SourceSelector) -> list[str]:
        """Return the ids of episodes whose source matches, sorted."""
        matched = []
        for episode in self._episodes.all_episodes():
            source = episode.source
            if source.system != selector.system:
                continue
            if selector.ref is not None and source.ref != selector.ref:
                continue
            if (
                selector.connector is not None
                and source.connector != selector.connector
            ):
                continue
            matched.append(episode.id)
        return matched

    async def snapshot(self) -> TemporalSnapshot:
        """Return every temporal row of the tenant."""
        return TemporalSnapshot(
            facts=tuple(self._sorted_facts()),
            observations=tuple(
                observation for _, observation in sorted(self._observations.items())
            ),
            supersessions=tuple(row for _, row in sorted(self._supersessions.items())),
            verdicts=tuple(verdict for _, verdict in sorted(self._verdicts.items())),
            decisions=tuple(
                decision for _, decision in sorted(self._decisions.items())
            ),
            conflicts=tuple(
                conflict for _, conflict in sorted(self._conflicts.items())
            ),
            slot_keys=tuple(sorted(self._slots)),
        )

    async def put_verdicts(self, verdicts: Sequence[FactVerdict]) -> None:
        """Store verdicts, replacing any row with the same verdict id."""
        for verdict in verdicts:
            self._verdicts[verdict.verdict_id] = verdict

    async def verdicts_for(self, fact_ids: Sequence[str]) -> list[FactVerdict]:
        """Return the verdicts that name any of the facts on either side."""
        wanted = set(fact_ids)
        return [
            verdict
            for _, verdict in sorted(self._verdicts.items())
            if verdict.fact_a_id in wanted or verdict.fact_b_id in wanted
        ]

    async def put_decision(self, decision: ConflictDecision) -> None:
        """Store a decision, replacing any row with the same decision id."""
        self._decisions[decision.decision_id] = decision

    async def decisions_for(self, fact_ids: Sequence[str]) -> list[ConflictDecision]:
        """Return the decisions that name any of the given facts."""
        wanted = set(fact_ids)
        return [
            decision
            for _, decision in sorted(self._decisions.items())
            if wanted & set(decision.fact_ids)
        ]

    async def conflicts_for(
        self, slot_keys: Sequence[str], fact_ids: Sequence[str]
    ) -> list[Conflict]:
        """Return the stored conflicts of the slots or that name the facts."""
        slots, wanted = set(slot_keys), set(fact_ids)
        return [
            conflict
            for _, conflict in sorted(self._conflicts.items())
            if conflict.slot_key in slots or wanted & set(conflict.fact_ids)
        ]

    async def current_facts(self) -> list[Fact]:
        """Return the current live version of every fact in the tenant."""
        return [
            fact
            for fact in self._sorted_facts()
            if fact.expired_at is None and fact.status is not FactStatus.WITHDRAWN
        ]

    async def facts_touching(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return the current live versions with an entity in ``entity_ids``."""
        wanted = set(entity_ids)
        return [
            fact
            for fact in await self.current_facts()
            if fact.subject_id in wanted or fact.object_id in wanted
        ]

    async def facts_by_ids(self, fact_ids: Sequence[str]) -> list[Fact]:
        """Return every version of each fact, ordered by fact id and version."""
        wanted = set(fact_ids)
        if not wanted:
            return []
        return [fact for fact in self._sorted_facts() if fact.fact_id in wanted]

    async def facts_of_entities(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return every version of every fact with an entity on a side."""
        wanted = set(entity_ids)
        if not wanted:
            return []
        return [
            fact
            for fact in self._sorted_facts()
            if fact.subject_id in wanted or fact.object_id in wanted
        ]

    async def observations_by_ids(self, obs_ids: Sequence[str]) -> list[Observation]:
        """Return the observations, ordered by observation id."""
        wanted = set(obs_ids)
        if not wanted:
            return []
        return [
            observation
            for _, observation in sorted(self._observations.items())
            if observation.obs_id in wanted
        ]

    async def pending_slots(self) -> list[str]:
        """Return every pending slot key in the tenant, sorted.

        A slot is pending when one of its current facts shares an entity with
        a fact of another slot and that pair has no verdict. The result is
        computed from the current facts and verdicts on each call.
        """
        return await pending_slots(self)

    def _replace_conflicts(self, slot_key: str, conflicts: Sequence[Conflict]) -> None:
        for held in self._conflicts_of(slot_key):
            del self._conflicts[held.conflict_id]
        for conflict in conflicts:
            self._conflicts[conflict.conflict_id] = conflict

    def _conflicts_of(self, slot_key: str) -> tuple[Conflict, ...]:
        return tuple(
            conflict
            for _, conflict in sorted(self._conflicts.items())
            if conflict.slot_key == slot_key
        )

    async def _require_episode(self, episode_id: str) -> Episode:
        episode = await self._episodes.get_episode(episode_id)
        if episode is None:
            raise MissingEpisodeError(episode_id)
        return episode

    def _sorted_facts(self) -> list[Fact]:
        return [
            fact
            for _, fact in sorted(
                self._facts.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]

    def _facts_of(self, slot_key: str) -> tuple[Fact, ...]:
        return tuple(fact for fact in self._sorted_facts() if fact.slot_key == slot_key)
