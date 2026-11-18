"""The temporal engine: ingest episodes and settle the slots they touch.

Ingestion records an episode as pending, writes its observations, reconciles
every slot they touch, appends one ledger entry for the episode, and marks it
committed with its content hash last. A crash before the last step leaves the
episode pending, and a second ingest of the same episode is refused.

Removing an episode deletes its observations and settles the slots again, so
facts that lost all their evidence become withdrawn tombstones. Resetting an
episode does the same, then clears its temporal state and removes its mentions,
so the episode can be ingested again.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from ontogr_mem.data_models.episode import Episode, TemporalStatus
from ontogr_mem.data_models.fact_versions import Fact
from ontogr_mem.data_models.facts import FactKind, Modality, TypedFact
from ontogr_mem.data_models.ledger import LedgerOp
from ontogr_mem.data_models.observation import NO_STAMP, Observation, ProvenanceStamp
from ontogr_mem.temporal.adapter import EntityRefResolver, observations_from_facts
from ontogr_mem.temporal.cardinality import DictRelationPolicy, RelationRules, rules_for
from ontogr_mem.temporal.closure import ClosureJudge, ClosureOutcome
from ontogr_mem.temporal.commit import plan_commit
from ontogr_mem.temporal.errors import (
    EpisodeAlreadyIngestedError,
    RevisionConflict,
)
from ontogr_mem.temporal.handler import (
    HandlerInputs,
    HandlerResult,
    Partner,
    slot_directives,
)
from ontogr_mem.temporal.ledger import TENANT_CHAIN, link
from ontogr_mem.temporal.pairs import pair_input
from ontogr_mem.temporal.query import FactView
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.temporal.reconcile import derive
from ontogr_mem.temporal.report import Outcome, TemporalReport
from ontogr_mem.temporal.settings import TemporalSettings
from ontogr_mem.temporal.temporal_store import (
    IncompleteEpisode,
    SlotCommit,
    SourceSelector,
    TemporalStore,
)
from ontogr_mem.temporal.timeutil import same_span
from ontogr_mem.temporal.trust import Trust
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    FactVerdict,
    VerdictKind,
    VerdictSource,
    pair_key,
)
from ontogr_mem.temporal.visibility import is_withdrawable


Clock = Callable[[], datetime]


class MentionRemover(Protocol):
    """Removes an episode's mentions from the resolution store.

    Resolution implements this, so the engine does not import it. Removing the
    mentions of an episode also deletes the entities that no longer have any.
    """

    async def remove_episode_mentions(self, episode_id: str) -> Sequence[str]:
        """Remove the episode's mentions and return the entities it deleted."""
        ...


class CommitObserver(Protocol):
    """Called around each slot commit.

    An observer that writes to another store (for example, a vector index) picks
    the call that fits its data. Before the commit, the graph does not refer to
    the data yet. After the commit, the data is written.
    """

    async def before_slot_commit(self, commit: SlotCommit) -> None:
        """Receive the commit that is about to be written."""
        ...

    async def after_slot_commit(
        self, commit: SlotCommit, observations: Mapping[str, Observation]
    ) -> None:
        """Receive the commit that was written, with the slot's observations."""
        ...


class TemporalEngine:
    """Reconciles observations into per-slot timelines and records episodes.

    The engine keeps no state beyond its configuration. Every write goes through
    the store, whose compare-and-swap refuses a slot that moved since it was read.

    Args:
        store: Temporal store of the tenant.
        policy: Relation rules that choose cardinality and slot keys.
        resolver_refs: Maps an episode-local entity ref to a stable entity id.
        clock: Returns the current time; stamps system time and ledger entries.
        hooks: Observers called before and after each slot commit, in order.
        settings: Retry and batch limits.
        trust: Trust rules and margin used to rank conflicting facts.
    """

    def __init__(
        self,
        store: TemporalStore,
        policy: DictRelationPolicy,
        resolver_refs: EntityRefResolver,
        clock: Clock,
        *,
        hooks: Iterable[CommitObserver] = (),
        settings: TemporalSettings | None = None,
        trust: Trust | None = None,
    ) -> None:
        """Bind the engine to a store and its configuration."""
        self._store = store
        self._policy = policy
        self._resolver_refs = resolver_refs
        self._clock = clock
        self._hooks = tuple(hooks)
        self._settings = settings or TemporalSettings()
        self._trust = trust or Trust()

    async def ingest(
        self,
        facts: Sequence[TypedFact],
        episode: Episode,
        stamp: ProvenanceStamp = NO_STAMP,
        *,
        unplaced: int | None = None,
        ontology_version: int | None = None,
    ) -> TemporalReport:
        """Turn the episode's typed facts into observations and settle them.

        An episode already committed with the same content hash is skipped.

        Raises:
            EpisodeAlreadyIngestedError: The episode has observations and is not
                committed with this content hash.
            MissingEpisodeError: The episode is not in the episode store.
        """
        content_hash = episode.content_hash
        if content_hash is None:
            msg = f"episode {episode.id!r} has no content hash to commit"
            raise ValueError(msg)
        report = TemporalReport()
        state = await self._store.episode_state(episode.id)
        if (
            state is not None
            and state.status is TemporalStatus.COMMITTED
            and state.committed_hash == content_hash
        ):
            report.bump(Outcome.EPISODE_SKIPPED)
            if state.skipped is not None:
                report.skipped.update(state.skipped)
            return report
        if await self._store.episode_has_observations(episode.id):
            raise EpisodeAlreadyIngestedError(episode.id)

        observations, skipped = await observations_from_facts(
            facts, episode, self._resolver_refs, self._policy, stamp
        )
        skipped_counts: Counter[str] = Counter()
        for item in skipped:
            skipped_counts[item.reason.value] += 1
        report.skipped.update(skipped_counts)
        await self._store.set_episode_pending(
            episode.id,
            unplaced=unplaced,
            ontology_version=ontology_version,
            skipped=(
                skipped_counts if skipped_counts or unplaced is not None else None
            ),
        )
        report.merge(await self.ingest_observations(observations))
        if report.ok:
            await self._commit_episode(episode.id, content_hash, report)
        return report

    async def ingest_observations(
        self, observations: Sequence[Observation]
    ) -> TemporalReport:
        """Store observations, then reconcile every slot they touch."""
        report = TemporalReport()
        size = self._settings.batch_size
        for start in range(0, len(observations), size):
            await self._store.append_observations(observations[start : start + size])
        report.bump(Outcome.OBSERVATION_SUBMITTED, len(observations))
        await self._settle(sorted({o.slot_key for o in observations}), report)
        return report

    async def record_verdicts(self, verdicts: Sequence[FactVerdict]) -> TemporalReport:
        """Store verdicts and reconcile every slot of the facts they name."""
        report = TemporalReport()
        if not verdicts:
            return report
        await self._store.put_verdicts(verdicts)
        fact_ids = {v.fact_a_id for v in verdicts} | {v.fact_b_id for v in verdicts}
        for slot_key in await self._slots_of_facts(fact_ids):
            await self._reconcile(slot_key, report)
        return report

    async def record_decision(self, decision: ConflictDecision) -> TemporalReport:
        """Store a person's decision and reconcile the slots of its facts."""
        report = TemporalReport()
        await self._store.put_decision(decision)
        for slot_key in await self._slots_of_facts(decision.fact_ids):
            await self._reconcile(slot_key, report)
        return report

    async def closure_pass(
        self, judge: ClosureJudge, fact_ids: Sequence[str], *, max_candidates: int
    ) -> TemporalReport:
        """Ask the judge about each new fact that has closure candidates."""
        report = TemporalReport()
        facts = await self._store.facts_by_ids(fact_ids)
        new_facts = [
            fact
            for fact in facts
            if fact.status.value == "active"
            and fact.expired_at is None
            and fact.valid_to is None
            and fact.modality is Modality.ACTUAL
            and fact.fact_kind is FactKind.STATE
        ]
        observations: dict[str, Mapping[str, Observation]] = {}
        for new in new_facts:
            neighbours = [
                fact
                for fact in await self._store.facts_touching([new.subject_id])
                if fact.subject_id == new.subject_id
                and fact.fact_id != new.fact_id
                and fact.status.value == "active"
                and fact.expired_at is None
                and fact.valid_to is None
                and fact.modality is Modality.ACTUAL
                and fact.fact_kind is FactKind.STATE
                and fact.slot_key != new.slot_key
                and not same_span(
                    fact.valid_from,
                    fact.valid_from_precision,
                    new.valid_from,
                    new.valid_from_precision,
                )
            ]
            new_rules = rules_for(
                self._policy, new.relation_type_id, new.modality, new.fact_kind
            )
            if new.relation_type_id is None:
                candidates = neighbours
            elif new_rules.judged:
                candidates = [
                    fact
                    for fact in neighbours
                    if fact.relation_type_id in (None, new.relation_type_id)
                ]
            else:
                candidates = [
                    fact for fact in neighbours if fact.relation_type_id is None
                ]
            candidates.sort(
                key=lambda fact: (-fact.valid_from.timestamp(), fact.fact_id)
            )
            candidates = candidates[:max_candidates]

            pairs = []
            for candidate in candidates:
                left = await self._partner(new, observations)
                right = await self._partner(candidate, observations)
                pairs.append(
                    pair_input(
                        pair_key(new.fact_id, candidate.fact_id),
                        left,
                        right,
                        self._trust,
                    )
                )
            stored = (
                await self._store.verdicts_for(
                    [new.fact_id, *(candidate.fact_id for candidate in candidates)]
                )
                if candidates
                else []
            )
            asked = []
            for pair in pairs:
                if any(
                    verdict.pair_key == pair.pair_key
                    and verdict.valid_for(pair.inputs_hash, judge.prompt_version)
                    for verdict in stored
                ):
                    report.bump(Outcome.JUDGE_CACHED)
                else:
                    asked.append(pair)
            if not asked:
                continue
            outcome = await judge.judge(
                pairs[0].a.statement,
                [(pair.b.fact_id, pair.b.statement) for pair in asked],
            )
            report.bump(Outcome.JUDGE_ASKED)
            _count_closure(outcome, report)
            if outcome.failed:
                continue
            by_id = {pair.b.fact_id: pair for pair in asked}
            verdicts = []
            for candidate_id in {pair.b.fact_id for pair in asked}:
                pair = by_id[candidate_id]
                if outcome.invalid_ids and candidate_id not in outcome.contradicted:
                    continue
                verdict = (
                    VerdictKind.CONTRADICTS
                    if candidate_id in outcome.contradicted
                    else VerdictKind.COMPATIBLE
                )
                verdicts.append(
                    FactVerdict(
                        pair_key=pair.pair_key,
                        fact_a_id=new.fact_id,
                        fact_b_id=candidate_id,
                        verdict=verdict,
                        inputs_hash=pair.inputs_hash,
                        prompt_version=judge.prompt_version,
                        source=VerdictSource.LLM,
                        reason_code="closure",
                        votes=(("1", verdict.value),),
                        decided_at=self._clock(),
                        source_episode_ids=pair.episode_ids,
                    )
                )
            report.merge(await self.record_verdicts(verdicts))
        return report

    async def reconcile_slot(self, slot_key: str) -> TemporalReport:
        """Recompute one slot from its observations and commit any difference."""
        report = TemporalReport()
        await self._reconcile(slot_key, report)
        return report

    async def reconcile_all(self) -> TemporalReport:
        """Recompute every slot in the tenant."""
        report = TemporalReport()
        await self._settle(await self._store.slot_keys(), report)
        return report

    async def _commit_episode(
        self, episode_id: str, content_hash: str, report: TemporalReport
    ) -> None:
        """Append the episode's ledger entry, then mark the episode committed.

        The entry comes first: a crash between the two leaves the episode pending
        rather than committed without an entry.
        """
        for _ in range(self._settings.max_retries + 3):
            head = await self._store.ledger_head(TENANT_CHAIN)
            entry = link(
                TENANT_CHAIN,
                head,
                self._clock(),
                LedgerOp.EPISODE_COMMITTED,
                episode_id,
                "engine",
                {"content_hash": content_hash},
            )
            if await self._store.append_ledger([entry], head[0]):
                break
        else:
            raise RevisionConflict(TENANT_CHAIN, head[0])
        await self._store.mark_committed(episode_id, content_hash)
        report.bump(Outcome.EPISODE_COMMITTED)

    async def current_facts(self) -> list[FactView]:
        """Return the current version of every fact, with statement and disputes."""
        return await FactReader(self._store).annotate(await self._store.current_facts())

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return the episodes left ``PENDING`` by an interrupted ingest."""
        return await self._store.incomplete_episodes()

    async def recover(self) -> TemporalReport:
        """Finish episodes left ``PENDING`` by an interrupted ingest.

        Roll-forward only: the observations were written before the crash, so
        settling their slots again and committing the episode is safe. An
        episode with no observations has nothing to settle; it is reported and
        left pending.
        """
        report = TemporalReport()
        for incomplete in await self._store.incomplete_episodes():
            observations = await self._store.observations_of_episode(
                incomplete.episode_id
            )
            if not observations or incomplete.content_hash is None:
                report.bump(Outcome.EPISODE_NOT_RECOVERABLE)
                continue
            slots = {o.slot_key for o in observations}
            settled = TemporalReport()
            await self._settle(
                sorted(slots | await self._partner_slots(slots)), settled
            )
            report.merge(settled)
            if settled.ok:
                await self._commit_episode(
                    incomplete.episode_id, incomplete.content_hash, report
                )
                report.bump(Outcome.EPISODE_RECOVERED)
        return report

    async def remove_episode(self, episode_id: str) -> TemporalReport:
        """Delete the episode's observations, settle its slots, and retract it.

        The observations go first, so the settle sees the slots as if the episode
        had never arrived. The retraction entry and the ``RETRACTED`` mark come
        last. An episode that is already retracted gets no second entry.
        """
        report = TemporalReport()
        slots = await self._store.remove_episode_observations(episode_id)
        report.bump(Outcome.SCRUBBED, await self._store.scrub_episode(episode_id))
        await self._settle(sorted(slots | await self._partner_slots(slots)), report)
        state = await self._store.episode_state(episode_id)
        if state is None or state.status is TemporalStatus.RETRACTED:
            return report
        await self._append_tenant(
            LedgerOp.EPISODE_RETRACTED,
            episode_id,
            {"content_hash": state.committed_hash},
        )
        await self._store.mark_retracted(episode_id)
        report.bump(Outcome.EPISODE_RETRACTED)
        return report

    async def retract_episodes(self, episode_ids: Iterable[str]) -> TemporalReport:
        """Remove several episodes in sorted id order, merging their reports."""
        report = TemporalReport()
        for episode_id in sorted(set(episode_ids)):
            report.merge(await self.remove_episode(episode_id))
        return report

    async def retract_source(self, selector: SourceSelector) -> TemporalReport:
        """Remove every episode whose source matches ``selector``."""
        return await self.retract_episodes(
            await self._store.episodes_matching(selector)
        )

    async def reset_episode(
        self, episode_id: str, mentions: MentionRemover
    ) -> TemporalReport:
        """Remove an episode's observations and mentions so it can be ingested again.

        Temporal goes first and mentions second. Once the observations are gone,
        the facts they backed are withdrawn, so the entities that only this
        episode supported have no live fact edge left and the mention removal
        can delete them. Unlike ``remove_episode``, the episode is not marked
        retracted and no retraction entry is written. A crash between the two
        steps leaves mentions without observations; the resolution guard refuses
        a rerun until this method runs again, and running it again is safe.
        """
        report = TemporalReport()
        slots = await self._store.remove_episode_observations(episode_id)
        report.bump(Outcome.SCRUBBED, await self._store.scrub_episode(episode_id))
        await self._settle(sorted(slots | await self._partner_slots(slots)), report)
        await self._store.clear_episode_state(episode_id)
        await mentions.remove_episode_mentions(episode_id)
        report.bump(Outcome.EPISODE_RESET)
        return report

    async def _append_tenant(
        self, op: LedgerOp, subject_id: str, payload: dict[str, object]
    ) -> None:
        """Append one entry to the tenant chain, retrying a lost head race."""
        for _ in range(self._settings.max_retries + 3):
            head = await self._store.ledger_head(TENANT_CHAIN)
            entry = link(
                TENANT_CHAIN, head, self._clock(), op, subject_id, "engine", payload
            )
            if await self._store.append_ledger([entry], head[0]):
                return
        raise RevisionConflict(TENANT_CHAIN, head[0])

    async def _partner_slots(self, slot_keys: set[str]) -> set[str]:
        """Return the slots of the other fact of every verdict about ``slot_keys``."""
        current = await self._store.current_facts()
        mine = {fact.fact_id for fact in current if fact.slot_key in slot_keys}
        named = {
            fact_id
            for verdict in await self._store.verdicts_for(sorted(mine))
            for fact_id in (verdict.fact_a_id, verdict.fact_b_id)
        }
        return {fact.slot_key for fact in current if fact.fact_id in named - mine}

    async def _settle(self, slot_keys: Sequence[str], report: TemporalReport) -> None:
        """Reconcile the given slots once each and report what each changed."""
        for slot_key in slot_keys:
            await self._reconcile(slot_key, report)

    async def _reconcile(self, slot_key: str, report: TemporalReport) -> None:
        """Recompute one slot and commit the difference, retrying lost races."""
        for attempt in range(self._settings.max_retries + 1):
            state = await self._store.load_slot(slot_key)
            observations = await self._store.observations_of_slot(slot_key)
            if not observations and not state.facts:
                return
            recorded_at = self._recorded_at(state.facts)
            first = observations[0] if observations else None
            fact = state.facts[0] if state.facts else None
            rules = (
                rules_for(
                    self._policy,
                    first.relation_type_id,
                    first.modality,
                    first.fact_kind,
                )
                if first is not None
                else rules_for(
                    self._policy, fact.relation_type_id, fact.modality, fact.fact_kind
                )
                if fact is not None
                else RelationRules()
            )
            cardinality = rules.cardinality
            handled = await self._handle(slot_key, state.facts, observations)
            if handled.rule_verdicts:
                await self._store.put_verdicts(handled.rule_verdicts)
            derived = derive(slot_key, observations, cardinality, handled.directives)
            commit = plan_commit(
                state, observations, derived, recorded_at, handled.conflicts
            )
            if commit.empty:
                report.bump(Outcome.SLOT_UNCHANGED)
                return
            for hook in self._hooks:
                await hook.before_slot_commit(commit)
            try:
                await self._store.commit_slot(commit)
            except RevisionConflict:
                if attempt < self._settings.max_retries:
                    report.bump(Outcome.RETRY)
                    continue
                report.bump(Outcome.COMMIT_FAILED)
                report.failures.append((slot_key, "revision_conflict"))
                return
            for hook in self._hooks:
                await hook.after_slot_commit(
                    commit, {o.obs_id: o for o in observations}
                )
            report.bump(Outcome.SLOT_COMMITTED)
            report.bump(Outcome.NEW_VERSION, len(commit.new_facts))
            report.bump(Outcome.SUPERSESSION, len(commit.supersessions))
            report.bump(Outcome.CONFLICT, len(derived.conflicts))
            report.bump(Outcome.UNMATCHED_END, len(derived.unmatched_ends))
            report.bump(Outcome.ORPHAN, len(derived.orphans))
            return

    async def _handle(
        self, slot_key: str, facts: Sequence[Fact], observations: Sequence[Observation]
    ) -> HandlerResult:
        """Decide the slot's directives from its stored verdicts and decisions."""
        current = [fact for fact in facts if is_withdrawable(fact)]
        entities = {e for f in current for e in (f.subject_id, f.object_id)}
        entities |= {e for o in observations for e in (o.subject_id, o.object_id)}
        partners: dict[str, Partner] = {}
        cache: dict[str, Mapping[str, Observation]] = {}
        for fact in await self._store.facts_touching(sorted(entities)):
            if fact.slot_key == slot_key:
                continue
            partners[fact.fact_id] = await self._partner(fact, cache)
        ids = [fact.fact_id for fact in current]
        return slot_directives(
            HandlerInputs(
                slot_key=slot_key,
                observations=observations,
                facts=facts,
                policy=self._policy,
                trust=self._trust,
                now=self._clock(),
                verdicts=await self._store.verdicts_for([*ids, *partners]),
                decisions=await self._store.decisions_for(ids),
                partners=partners,
            )
        )

    async def _partner(
        self, fact: Fact, cache: dict[str, Mapping[str, Observation]]
    ) -> Partner:
        """Pair a fact with the observations of its slot, read once per slot."""
        if fact.slot_key not in cache:
            cache[fact.slot_key] = {
                o.obs_id: o
                for o in await self._store.observations_of_slot(fact.slot_key)
            }
        return Partner(fact, cache[fact.slot_key])

    async def _slots_of_facts(self, fact_ids: Iterable[str]) -> list[str]:
        wanted = set(fact_ids)
        return sorted(
            {
                fact.slot_key
                for fact in await self._store.current_facts()
                if fact.fact_id in wanted
            }
        )

    def _recorded_at(self, facts: Sequence[Fact]) -> datetime:
        """Return a system time no earlier than every current slot version.

        A frozen or coarse clock can leave the clock at the latest version's
        recorded time. Committing at that same instant is correct: the new
        version supersedes the old one immediately, so a read at the clock's
        current time sees the committed belief.
        """
        now = self._clock()
        floor = max(
            (fact.recorded_at for fact in facts if fact.expired_at is None),
            default=None,
        )
        if floor is None:
            return now
        return max(now, floor)


def _count_closure(outcome: ClosureOutcome, report: TemporalReport) -> None:
    """Fold closure failures, usage, and contradictions into the report."""
    report.bump(Outcome.TOKENS_IN, outcome.input_tokens)
    report.bump(Outcome.TOKENS_OUT, outcome.output_tokens)
    for key, count in outcome.counts.items():
        report.counts[key] += count
    report.bump(Outcome.JUDGE_CONTRADICTS, len(outcome.contradicted))
