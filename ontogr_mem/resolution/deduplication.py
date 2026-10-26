"""Deferred deduplication: find duplicate entities, verify, then merge."""

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ontogr_mem.data_models import Usage
from ontogr_mem.data_models.entities import (
    Entity,
    EntityStatus,
    EntityVerdict,
    VerdictKind,
    VerdictSource,
    VerifyStatus,
)
from ontogr_mem.resolution.blocking import BlockIndex
from ontogr_mem.resolution.config import Budget
from ontogr_mem.resolution.errors import MergeRefusedError
from ontogr_mem.resolution.keys import (
    differ_only_in_digits,
    entity_text,
    stripped_key,
)
from ontogr_mem.resolution.merge import pick_survivor, survivor_order
from ontogr_mem.resolution.policy import (
    ResolutionMode,
    TypeRel,
    identity_agree,
    identity_conflict,
)
from ontogr_mem.resolution.scoring import PairSignals, needs_two_votes
from ontogr_mem.resolution.verdict_ids import inputs_hash, verdict_id
from ontogr_mem.resolution.verifier import (
    PairInput,
    PairOutcome,
)
from ontogr_mem.vectorstore import VectorKind


if TYPE_CHECKING:
    from ontogr_mem.resolution.resolver import EntityResolver


Pair = tuple[str, str]


@dataclass(frozen=True, slots=True)
class PlannedMerge:
    """One clique that a deduplication run would merge."""

    survivor_id: str
    absorbed_ids: tuple[str, ...]


@dataclass(slots=True)
class ConsolidationReport:
    """Counters and the plan of one deduplication run."""

    plan_only: bool = False
    selected: int = 0
    pairs_considered: int = 0
    pairs_skipped_fresh: int = 0
    pairs_rule_same: int = 0
    pairs_asked: int = 0
    pairs_rejected: int = 0
    pairs_merged: int = 0
    invalid_ids: int = 0
    parse_failed: int = 0
    errors: int = 0
    no_verdict: int = 0
    merges: list[str] = field(default_factory=list)
    plan: list[PlannedMerge] = field(default_factory=list)


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


def _vetoed(resolver: "EntityResolver", a: Entity, b: Entity) -> bool:
    if (
        resolver.policy.relation(a.type_label, b.type_label, resolver.relations)
        is TypeRel.DISTINCT
    ):
        return True
    if identity_conflict(a.identity_attrs, b.identity_attrs):
        return True
    if a.uri and b.uri and a.uri != b.uri:
        return True
    return differ_only_in_digits(a.canonical_name, b.canonical_name)


def _rule_same(
    resolver: "EntityResolver",
    a: Entity,
    b: Entity,
    keys: dict[str, set[str]],
) -> bool:
    if resolver.policy.relation(a.type_label, b.type_label, resolver.relations) not in (
        TypeRel.SAME,
        TypeRel.NESTED,
    ):
        return False
    if identity_agree(a.identity_attrs, b.identity_attrs):
        return True
    if (
        resolver.policy.mode(a.type_label) is ResolutionMode.AUTO_KEY
        and resolver.policy.mode(b.type_label) is ResolutionMode.AUTO_KEY
    ):
        return bool(keys.get(a.entity_id, set()) & keys.get(b.entity_id, set()))
    return False


def _cliques(
    entities: dict[str, Entity], same: set[Pair], cap: int
) -> list[list[Entity]]:
    adjacency: dict[str, set[str]] = {}
    for a, b in same:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen: set[str] = set()
    out: list[list[Entity]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component
        pending = sorted(
            (entities[i] for i in component if i in entities),
            key=survivor_order,
        )
        while pending:
            clique = [pending[0]]
            for candidate in pending[1:]:
                if len(clique) >= cap:
                    break
                if all(
                    _pair(candidate.entity_id, member.entity_id) in same
                    for member in clique
                ):
                    clique.append(candidate)
            if len(clique) > 1:
                out.append(clique)
            placed = {member.entity_id for member in clique}
            pending = [e for e in pending if e.entity_id not in placed]
    return out


def _stronger(left: float | None, right: float | None) -> float | None:
    """Return the stronger of two values, ignoring the missing one."""
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _merge(left: PairSignals, right: PairSignals) -> PairSignals:
    """Return the stronger value of each signal."""
    return PairSignals(
        cosine=_stronger(left.cosine, right.cosine),
        jaccard=_stronger(left.jaccard, right.jaccard),
    )


async def _candidates(
    resolver: "EntityResolver",
    entity: Entity,
    entities: dict[str, Entity],
    index: BlockIndex,
    vector: tuple[float, ...] | None,
) -> dict[str, PairSignals]:
    """Return the signals measured between this entity and every candidate."""
    found: dict[str, PairSignals] = {}
    keys = [entity.loose_key]
    if resolver.policy.strip_suffixes(entity.type_label):
        keys.append(stripped_key(entity.canonical_name))
    for key in keys:
        for item in index.candidates(key):
            other_id = item.split("#")[0]
            if other_id != entity.entity_id:
                signals = found.get(other_id, PairSignals())
                found[other_id] = replace(
                    signals,
                    jaccard=_stronger(
                        signals.jaccard,
                        index.similarity(item, entity.loose_key),
                    ),
                )
    if vector is not None:
        hits = await resolver.vectors.search(
            VectorKind.NODE,
            vector,
            k=resolver.settings.ann_top_k,
            min_score=resolver.settings.ann_floor,
        )
        for hit in hits:
            if hit.id != entity.entity_id and hit.id in entities:
                signals = found.get(hit.id, PairSignals())
                found[hit.id] = replace(
                    signals,
                    cosine=_stronger(signals.cosine, hit.score),
                )
    return found


async def run_deduplication(
    resolver: "EntityResolver", *, plan_only: bool, budget: Budget
) -> ConsolidationReport:
    """Run one deduplication pass over the tenant's active entities."""
    report = ConsolidationReport(plan_only=plan_only)
    selected, entities, index, keys = await _context(resolver, budget)
    report.selected = len(selected)
    if not selected:
        return report
    signals = await _pair_signals(resolver, selected, entities, index)
    known, to_ask = await _classify(resolver, report, signals, entities, keys)
    if plan_only:
        same = {pair for pair, verdict in known.items() if verdict is VerdictKind.SAME}
        report.plan = _plan(entities, same, resolver.settings.cluster_cap)
        return report
    await _ask(
        resolver,
        report,
        budget,
        signals=signals,
        entities=entities,
        to_ask=to_ask,
        known=known,
    )
    same = {pair for pair, verdict in known.items() if verdict is VerdictKind.SAME}
    report.plan = _plan(entities, same, resolver.settings.cluster_cap)
    await _apply(resolver, report)
    await _mark_checked(resolver, selected)
    return report


async def _context(
    resolver: "EntityResolver", budget: Budget
) -> tuple[list[Entity], dict[str, Entity], BlockIndex, dict[str, set[str]]]:
    """Return the selected entities, all active entities, the block index and keys."""
    store = resolver.store
    selected = await store.list_for_deduplication(0, False, budget.sweep)
    if not selected:
        return [], {}, BlockIndex(), {}
    rows = await store.blocking_rows()
    active = await store.get_entities([row.entity_id for row in rows])
    entities = {entity.entity_id: entity for entity in active}
    entities.update({entity.entity_id: entity for entity in selected})
    keys: dict[str, set[str]] = {}
    alias_rows = await asyncio.gather(
        *(store.aliases_of(entity.entity_id) for entity in selected)
    )
    for entity, aliases in zip(selected, alias_rows, strict=True):
        keys[entity.entity_id] = {entity.strict_key} | {
            alias.strict_key for alias in aliases
        }
    return selected, entities, BlockIndex.from_rows(rows), keys


async def _pair_signals(
    resolver: "EntityResolver",
    selected: list[Entity],
    entities: dict[str, Entity],
    index: BlockIndex,
) -> dict[Pair, PairSignals]:
    """Return the signals measured for every candidate pair."""
    signals: dict[Pair, PairSignals] = {}
    ordered = sorted(selected, key=lambda entity: entity.entity_id)
    embeddings = await resolver.vectors.gateway.embed(
        [entity_text(entity) for entity in ordered]
    )
    for entity, embedding in zip(ordered, embeddings, strict=True):
        found = await _candidates(resolver, entity, entities, index, embedding.vector)
        for other_id, value in found.items():
            other = entities.get(other_id)
            if other is None or other.status is not EntityStatus.ACTIVE:
                continue
            if _vetoed(resolver, entity, other):
                continue
            pair = _pair(entity.entity_id, other_id)
            signals[pair] = _merge(signals.get(pair, PairSignals()), value)
    return signals


async def _classify(
    resolver: "EntityResolver",
    report: ConsolidationReport,
    signals: dict[Pair, PairSignals],
    entities: dict[str, Entity],
    keys: dict[str, set[str]],
) -> tuple[dict[Pair, VerdictKind], list[Pair]]:
    """Use stored verdicts and rules, and return the pairs left to verify."""
    known: dict[Pair, VerdictKind] = {}
    to_ask: list[Pair] = []
    pairs = sorted(signals)
    stored_rows = await asyncio.gather(
        *(resolver.store.live_verdict(*pair) for pair in pairs)
    )
    for pair, stored in zip(pairs, stored_rows, strict=True):
        a, b = entities[pair[0]], entities[pair[1]]
        report.pairs_considered += 1
        if stored is not None and stored.inputs_hash == inputs_hash(a, b):
            report.pairs_skipped_fresh += 1
            known[pair] = stored.verdict
        elif _rule_same(resolver, a, b, keys):
            report.pairs_rule_same += 1
            known[pair] = VerdictKind.SAME
        else:
            to_ask.append(pair)
    return known, to_ask


async def _ask(
    resolver: "EntityResolver",
    report: ConsolidationReport,
    budget: Budget,
    *,
    signals: dict[Pair, PairSignals],
    entities: dict[str, Entity],
    to_ask: list[Pair],
    known: dict[Pair, VerdictKind],
) -> None:
    """Verify the pairs within budget and store the verdicts."""
    verifier = resolver.verifier
    if verifier is None or not to_ask:
        return
    take = to_ask[: max(0, budget.max_llm_pairs)]
    report.pairs_asked = len(take)
    inputs = [
        _pair_input(
            resolver,
            entities[a_id],
            entities[b_id],
            signals.get((a_id, b_id), PairSignals()),
        )
        for a_id, b_id in take
    ]
    outcomes, stats = await verifier.verify(inputs, Usage())
    report.invalid_ids += stats.invalid_ids
    new: dict[Pair, tuple[VerdictKind, VerdictSource, list[str]]] = {}
    for pair, outcome in zip(take, outcomes, strict=True):
        kind = _apply_outcome(report, known, pair, outcome)
        if kind is not None:
            new[pair] = (kind, VerdictSource.LLM, outcome.passes)
    await _store_verdicts(resolver, entities, new)


def _pair_input(
    resolver: "EntityResolver", a: Entity, b: Entity, signals: PairSignals
) -> PairInput:
    """Build one verifier input with the votes its signals need."""
    cross = (
        resolver.policy.relation(a.type_label, b.type_label, resolver.relations)
        is TypeRel.FAMILY
    )
    return PairInput(
        a,
        b,
        two_votes=needs_two_votes(signals, resolver.settings, cross),
        cosine=signals.cosine,
        jaccard=signals.jaccard,
    )


def _apply_outcome(
    report: ConsolidationReport,
    known: dict[Pair, VerdictKind],
    pair: Pair,
    outcome: PairOutcome,
) -> VerdictKind | None:
    """Record one verifier outcome and return the verdict to store, if any."""
    if outcome.status is VerifyStatus.OK and outcome.verdict is not None:
        known[pair] = outcome.verdict
        if outcome.verdict is VerdictKind.DISTINCT:
            report.pairs_rejected += 1
        return outcome.verdict
    if outcome.status is VerifyStatus.VOTE_DISAGREED:
        known[pair] = VerdictKind.UNCERTAIN
        return VerdictKind.UNCERTAIN
    if outcome.status is VerifyStatus.PARSE_FAILED:
        report.parse_failed += 1
    elif outcome.status is VerifyStatus.ERROR:
        report.errors += 1
    else:
        report.no_verdict += 1
    return None


async def _store_verdicts(
    resolver: "EntityResolver",
    entities: dict[str, Entity],
    new: dict[Pair, tuple[VerdictKind, VerdictSource, list[str]]],
) -> None:
    """Write the verdicts produced by this pass."""
    verdicts: list[EntityVerdict] = []
    for pair, (kind, source, passes) in sorted(new.items()):
        lo, hi, vid = verdict_id(*pair)
        verdicts.append(
            EntityVerdict(
                verdict_id=vid,
                a_id=lo,
                b_id=hi,
                verdict=kind,
                source=source,
                inputs_hash=inputs_hash(entities[lo], entities[hi]),
                passes=passes,
            )
        )
    if verdicts:
        await resolver.store.write_verdicts(verdicts)


async def _apply(resolver: "EntityResolver", report: ConsolidationReport) -> None:
    """Merge every planned clique."""
    for planned in report.plan:
        for absorbed_id in planned.absorbed_ids:
            try:
                record = await resolver.merge(
                    planned.survivor_id,
                    absorbed_id,
                    actor="resolver",
                    reason="deduplication: verified same",
                )
            except MergeRefusedError:
                report.pairs_rejected += 1
                continue
            report.merges.append(record.merge_id)
            report.pairs_merged += 1


async def _mark_checked(resolver: "EntityResolver", selected: list[Entity]) -> None:
    """Record that these entities have now been checked at their revision."""
    current = await resolver.store.get_entities(
        [entity.entity_id for entity in selected]
    )
    checked = [
        entity.model_copy(update={"last_checked_version": entity.revision})
        for entity in current
        if entity.status is EntityStatus.ACTIVE
    ]
    await asyncio.gather(
        *(resolver.store.save_entity(entity, bump_revision=False) for entity in checked)
    )


def _plan(entities: dict[str, Entity], same: set[Pair], cap: int) -> list[PlannedMerge]:
    cliques = _cliques(entities, same, cap)
    plan: list[PlannedMerge] = []
    for clique in cliques:
        survivor = pick_survivor(clique)
        plan.append(
            PlannedMerge(
                survivor_id=survivor.entity_id,
                absorbed_ids=tuple(
                    sorted(
                        e.entity_id for e in clique if e.entity_id != survivor.entity_id
                    )
                ),
            )
        )
    plan.sort(key=lambda item: item.survivor_id)
    return plan
