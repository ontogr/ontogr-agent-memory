"""The contradiction handler: pairs, verdicts and decisions into slot directives.

Work for one slot runs in this order. Same-slot pairs of a single-valued
relation are decided by the rules. Cross-slot pairs come from stored verdicts,
and a person's decision overrides any rule or verdict. The result is a value
that the engine passes to ``derive`` and commits. Nothing here orders facts
in time and nothing here writes to a store.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import Fact, FactStatus, SupersessionKind
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.temporal.anchor import anchor_observation_id
from ontogr_mem.temporal.cardinality import (
    ConflictPolicy,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.classify import (
    Classification,
    ConflictClass,
    Disposition,
    Side,
    apply_decision,
    classify,
    order,
)
from ontogr_mem.temporal.commit import assign_fact_ids
from ontogr_mem.temporal.reconcile import (
    Close,
    Conflict,
    ConflictKind,
    Derived,
    Run,
    SlotDirectives,
    derive,
)
from ontogr_mem.temporal.trust import Trust
from ontogr_mem.temporal.verdicts import (
    RULE_PROMPT_VERSION,
    ConflictDecision,
    FactFingerprint,
    FactVerdict,
    VerdictKind,
    VerdictSource,
    conflict_id_for,
    inputs_hash,
    latest_decision,
    pair_key,
)
from ontogr_mem.temporal.visibility import is_withdrawable


EPOCH = datetime(1, 1, 1, tzinfo=UTC)
"""Earliest instant, used where a verdict has no decision time."""


@dataclass(frozen=True, slots=True)
class Partner:
    """A fact in another slot, with the observations of its slot.

    Attributes:
        fact: The current version of the partner.
        observations: The observations of the partner's slot, by id.
    """

    fact: Fact
    observations: Mapping[str, Observation]


@dataclass(frozen=True, slots=True)
class HandlerInputs:
    """Everything the handler reads for one slot.

    Attributes:
        slot_key: Slot being decided.
        observations: Every observation of the slot.
        facts: The current version of every fact of the slot.
        policy: Relation rules.
        trust: Trust rules and margin.
        now: System time of the reconcile; stamps rule verdicts.
        verdicts: Stored verdicts that touch the slot's facts or partners.
        decisions: Person decisions about the slot's facts.
        partners: Facts in other slots that share an entity with this slot.
    """

    slot_key: str
    observations: Sequence[Observation]
    facts: Sequence[Fact]
    policy: DictRelationPolicy
    trust: Trust
    now: datetime
    verdicts: Sequence[FactVerdict] = ()
    decisions: Sequence[ConflictDecision] = ()
    partners: Mapping[str, Partner] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HandlerResult:
    """What the handler decided for one slot.

    Attributes:
        directives: Input for ``derive``.
        conflicts: Pair conflicts to store, sorted by conflict id.
        rule_verdicts: Verdicts the rules made, to store as inputs.
    """

    directives: SlotDirectives
    conflicts: tuple[Conflict, ...]
    rule_verdicts: tuple[FactVerdict, ...]


def side_of(
    fact_id: str,
    slot_key: str,
    obs_ids: tuple[str, ...],
    observations: Mapping[str, Observation],
    trust: Trust,
    *,
    valid_from: datetime | None,
    valid_from_prec: Precision | None,
    valid_to: datetime | None,
) -> Side:
    """Build the classifier's view of a fact from its evidence."""
    evidence = [observations[i] for i in obs_ids if i in observations]
    first_obs_id = anchor_observation_id(obs_ids, observations)
    first = observations.get(first_obs_id)
    start = valid_from.isoformat() if valid_from else ""
    # The fingerprint checks a stored verdict against the identity and first
    # statement. The end date is a verdict's effect, so including it would
    # invalidate the verdict when it is applied.
    fingerprint = FactFingerprint(
        slot_key=slot_key,
        subject_id=first.subject_id if first else "",
        relation_type_id=first.relation_type_id if first else None,
        object_id=first.object_id if first else "",
        valid_from=start,
        statement=first.statement if first else "",
    )
    return Side(
        fact_id=fact_id,
        slot_key=slot_key,
        first_obs_id=first_obs_id,
        obs_ids=obs_ids,
        valid_from=valid_from,
        valid_from_prec=valid_from_prec,
        valid_to=valid_to,
        rank=trust.fact_rank(evidence),
        corrects=any(o.kind is ObservationKind.CORRECTION for o in evidence),
        fp=fingerprint,
    )


def fact_side(
    fact: Fact, observations: Mapping[str, Observation], trust: Trust
) -> Side:
    """Return the classifier's view of a stored fact version."""
    return side_of(
        fact.fact_id,
        fact.slot_key,
        fact.evidence,
        observations,
        trust,
        valid_from=fact.valid_from,
        valid_from_prec=fact.valid_from_precision,
        valid_to=fact.valid_to,
    )


def _run_side(
    fact_id: str,
    slot_key: str,
    run: Run,
    observations: Mapping[str, Observation],
    trust: Trust,
) -> Side:
    return side_of(
        fact_id,
        slot_key,
        run.obs_ids,
        observations,
        trust,
        valid_from=run.valid_from,
        valid_from_prec=run.valid_from_precision,
        valid_to=run.valid_to,
    )


def pair_policy(a: RelationRules, b: RelationRules) -> ConflictPolicy:
    """Return the stricter of two relations' policies: ignore, review, auto."""
    if ConflictPolicy.IGNORE in (a.conflict_policy, b.conflict_policy):
        return ConflictPolicy.IGNORE
    if (
        a.conflict_policy is ConflictPolicy.AUTO
        and b.conflict_policy is ConflictPolicy.AUTO
    ):
        return ConflictPolicy.AUTO
    return ConflictPolicy.REVIEW


def _episodes(
    sides: Sequence[Side], observations: Mapping[str, Observation]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                observations[i].episode_id
                for s in sides
                for i in s.obs_ids
                if i in observations
            }
        )
    )


def judge_pending_slots(
    facts: Sequence[Fact],
    verdicts: Sequence[FactVerdict],
) -> set[str]:
    """Return the slots whose facts share an entity with an unjudged fact.

    A pair of facts is judged when a stored verdict names both of them. A
    slot is judge-pending while one of its facts shares an entity with a fact
    of another slot and that pair has no verdict. Nothing is stored: the answer
    comes from the facts and the verdicts as they are now.

    Args:
        facts: The current facts of the tenant.
        verdicts: The stored verdicts.

    Returns:
        The slot keys that still need the judge, unsorted.
    """
    judged = {frozenset((verdict.fact_a_id, verdict.fact_b_id)) for verdict in verdicts}
    by_entity: dict[str, list[Fact]] = {}
    for fact in facts:
        for entity in (fact.subject_id, fact.object_id):
            by_entity.setdefault(entity, []).append(fact)
    pending: set[str] = set()
    for bucket in by_entity.values():
        for index, one in enumerate(bucket):
            for other in bucket[index + 1 :]:
                if one.slot_key == other.slot_key:
                    continue
                if frozenset((one.fact_id, other.fact_id)) in judged:
                    continue
                pending.add(one.slot_key)
                pending.add(other.slot_key)
    return pending


def _same_slot_pairs(
    baseline: Derived,
    runs: Mapping[str, Run],
    mine: Mapping[str, Side],
) -> list[tuple[Side, Side, bool]]:
    """Collect replacement and equal-start pairs within a single-valued slot."""
    by_obs = {
        obs_id: fact_id for fact_id, run in runs.items() for obs_id in run.obs_ids
    }
    pairs: list[tuple[Side, Side, bool]] = []
    for fact_id, run in runs.items():
        if run.close_kind is SupersessionKind.REPLACEMENT and run.closed_by:
            later = by_obs.get(run.closed_by)
            if later and later != fact_id:
                pairs.append((mine[fact_id], mine[later], False))
    for tie in baseline.conflicts:
        if tie.kind is not ConflictKind.EQUAL_START:
            continue
        members = sorted({by_obs[obs_id] for obs_id in tie.obs_ids if obs_id in by_obs})
        pairs.extend(
            (mine[first], mine[second], True)
            for index, first in enumerate(members)
            for second in members[index + 1 :]
            if runs[first].value_id != runs[second].value_id
        )
    return pairs


def _cross_slot_pairs(
    inputs: HandlerInputs,
    mine: Mapping[str, Side],
    trust: Trust,
) -> list[tuple[FactVerdict, Side, Side, Partner]]:
    """Collect the newest current verdict for each local and partner fact pair."""
    newest: dict[
        tuple[str, str], tuple[datetime, str, FactVerdict, Side, Side, Partner]
    ] = {}
    for verdict in sorted(inputs.verdicts, key=lambda item: item.verdict_id):
        fact_ids = (verdict.fact_a_id, verdict.fact_b_id)
        mine_id = next(
            (fact_id for fact_id in fact_ids if fact_id in mine),
            None,
        )
        other_id = next(
            (fact_id for fact_id in fact_ids if fact_id not in mine),
            None,
        )
        if mine_id is None or other_id is None or other_id not in inputs.partners:
            continue
        partner = inputs.partners[other_id]
        local_side = mine[mine_id]
        partner_side = fact_side(partner.fact, partner.observations, trust)
        first, second = order(local_side, partner_side)
        if verdict.inputs_hash != inputs_hash(first.fp, second.fp):
            continue
        stamp = (verdict.decided_at or EPOCH, verdict.verdict_id)
        pair = (mine_id, other_id)
        held = newest.get(pair)
        if held is None or stamp > held[:2]:
            newest[pair] = (*stamp, verdict, local_side, partner_side, partner)
    return [
        (verdict, local_side, partner_side, partner)
        for _, _, verdict, local_side, partner_side, partner in sorted(
            newest.values(), key=lambda item: item[1]
        )
    ]


def _rule_verdict(
    first: Side,
    second: Side,
    observations: Mapping[str, Observation],
    now: datetime,
) -> FactVerdict:
    """Build the stored verdict for an exclusive-value pair."""
    a, b = order(first, second)
    return FactVerdict(
        pair_key=pair_key(a.fact_id, b.fact_id),
        fact_a_id=a.fact_id,
        fact_b_id=b.fact_id,
        verdict=VerdictKind.CONTRADICTS,
        inputs_hash=inputs_hash(a.fp, b.fp),
        prompt_version=RULE_PROMPT_VERSION,
        source=VerdictSource.RULE,
        kind="exclusive_value",
        reason_code="single_valued_slot",
        decided_at=now,
        source_episode_ids=_episodes([a, b], observations),
    )


def _baseline_sides(
    inputs: HandlerInputs, rules: RelationRules, obs: Mapping[str, Observation]
) -> tuple[Derived, dict[str, Run], dict[str, Side]]:
    """Derive baseline facts and assign their identities before pair decisions."""
    key = inputs.slot_key
    # Derive without directives to name the baseline facts and their ids. The
    # engine applies the directives in a later derive.
    baseline = derive(key, inputs.observations, rules.cardinality)
    current = [fact for fact in inputs.facts if is_withdrawable(fact)]
    ids = assign_fact_ids(key, baseline.runs, current)
    mine = {
        fid: _run_side(fid, key, run, obs, inputs.trust)
        for run, fid in zip(baseline.runs, ids, strict=True)
    }
    runs = dict(zip(ids, baseline.runs, strict=True))
    return baseline, runs, mine


def _pair_result(
    key: str,
    obs: Mapping[str, Observation],
    x: Side,
    y: Side,
    outcome: Classification,
    *,
    verdict: FactVerdict | None,
    owner: str,
) -> HandlerResult:
    """Return one pair's local directives and any owned conflict."""
    excluded: dict[str, FactStatus] = {}
    no_cut: list[frozenset[str]] = []
    closes: list[Close] = []
    conflicts: dict[str, Conflict] = {}
    a, b = order(x, y)
    if outcome.loser and outcome.loser_status is not None:
        loser = a if a.fact_id == outcome.loser else b
        if loser.slot_key == key:
            for obs_id in loser.obs_ids:
                if obs_id in obs:
                    excluded[obs_id] = outcome.loser_status
    if (
        outcome.close_earlier
        and verdict is not None
        and a.slot_key == key
        and b.slot_key != key
        and b.valid_from is not None
    ):
        closes.append(
            Close(
                anchor_obs_id=a.first_obs_id,
                at=b.valid_from,
                at_precision=b.valid_from_prec,
                cause_id=verdict.verdict_id,
            )
        )
    if outcome.disposition is Disposition.COEXIST and a.slot_key == b.slot_key == key:
        no_cut.append(frozenset(a.obs_ids) | frozenset(b.obs_ids))
    if owner == key and outcome.cls is not ConflictClass.UPDATE:
        pair = pair_key(a.fact_id, b.fact_id)
        conflict_id = conflict_id_for(pair)
        conflicts[conflict_id] = Conflict(
            conflict_id=conflict_id,
            slot_key=key,
            kind=(
                ConflictKind.UNCERTAIN
                if outcome.cls is ConflictClass.UNCERTAIN
                else ConflictKind.PAIR
            ),
            obs_ids=tuple(sorted({*a.obs_ids, *b.obs_ids})),
            episode_ids=_episodes([a, b], obs),
            reason=outcome.reason,
            cls=outcome.cls,
            relation_to_time=outcome.relation_to_time,
            disposition=outcome.disposition,
            fact_ids=(a.fact_id, b.fact_id),
            pair_key=pair,
        )
    return HandlerResult(
        SlotDirectives(excluded, tuple(no_cut), tuple(closes)),
        tuple(conflicts.values()),
        (),
    )


def _decide_pair(
    inputs: HandlerInputs,
    x: Side,
    y: Side,
    policy: ConflictPolicy,
    judged_uncertain: bool,
    *,
    confirmed_contradiction: bool = False,
) -> Classification | None:
    """Apply a person's decision before the classifier's rules."""
    person = latest_decision(inputs.decisions, (x.fact_id, y.fact_id))
    if person is not None:
        return apply_decision(x, y, person)
    return classify(
        x,
        y,
        policy=policy,
        margin=inputs.trust.margin,
        judged_uncertain=judged_uncertain,
        confirmed_contradiction=confirmed_contradiction,
    )


def _same_slot_results(
    inputs: HandlerInputs,
    rules: RelationRules,
    baseline: Derived,
    runs: Mapping[str, Run],
    mine: Mapping[str, Side],
    *,
    obs: Mapping[str, Observation],
) -> list[HandlerResult]:
    """Classify baseline replacements and ties, retaining every rule verdict."""
    results = []
    if not rules.single_valued:
        return results

    # Same-slot pairs of a single-valued relation represent a replacement or a
    # tie. Two values of a one-per-subject relation always contradict, so no
    # model call is needed.
    for x, y, is_tie in _same_slot_pairs(baseline, runs, mine):
        verdict = _rule_verdict(x, y, obs, inputs.now)
        outcome = _decide_pair(inputs, x, y, rules.conflict_policy, False)
        result = HandlerResult(SlotDirectives(), (), ())
        if outcome is not None:
            plain_tie = is_tie and outcome.disposition is Disposition.COEXIST
            result = _pair_result(
                inputs.slot_key,
                obs,
                x,
                y,
                outcome,
                verdict=None,
                owner="" if plain_tie else inputs.slot_key,
            )
        results.append(HandlerResult(result.directives, result.conflicts, (verdict,)))
    return results


def _cross_slot_results(
    inputs: HandlerInputs,
    rules: RelationRules,
    mine: Mapping[str, Side],
    obs: Mapping[str, Observation],
) -> list[HandlerResult]:
    """Classify the newest current verdicts against partner facts."""
    results = []

    # Cross-slot pairs come from stored verdicts. Only the newest answer about
    # the same two facts counts, so a later verdict replaces an older one.
    for v, x, y, partner in _cross_slot_pairs(inputs, mine, inputs.trust):
        if v.verdict not in (VerdictKind.CONTRADICTS, VerdictKind.UNCERTAIN):
            continue
        policy = pair_policy(rules, inputs.policy.rules(partner.fact.relation_type_id))
        outcome = _decide_pair(
            inputs,
            x,
            y,
            policy,
            v.verdict is VerdictKind.UNCERTAIN,
            confirmed_contradiction=v.verdict is VerdictKind.CONTRADICTS,
        )
        if outcome is not None:
            results.append(
                _pair_result(
                    inputs.slot_key,
                    obs,
                    x,
                    y,
                    outcome,
                    verdict=v,
                    owner=order(x, y)[0].slot_key,
                )
            )
    return results


def _combine_results(results: Sequence[HandlerResult]) -> HandlerResult:
    """Combine pair results in order, with later exclusions and conflicts winning."""
    excluded: dict[str, FactStatus] = {}
    conflicts: dict[str, Conflict] = {}
    for result in results:
        excluded.update(result.directives.excluded)
        conflicts.update(
            (conflict.conflict_id, conflict) for conflict in result.conflicts
        )

    return HandlerResult(
        directives=SlotDirectives(
            excluded=excluded,
            no_cut=tuple(
                group for result in results for group in result.directives.no_cut
            ),
            closes=tuple(
                close for result in results for close in result.directives.closes
            ),
        ),
        conflicts=tuple(sorted(conflicts.values(), key=lambda c: c.conflict_id)),
        rule_verdicts=tuple(
            verdict for result in results for verdict in result.rule_verdicts
        ),
    )


def slot_directives(inputs: HandlerInputs) -> HandlerResult:
    """Compute the directives, conflicts and verdicts for one slot.

    Args:
        inputs: The slot, its rules and the stored verdicts and decisions.

    Returns:
        The handler result. It is empty when the slot has no observations.
    """
    observations = list(inputs.observations)
    if not observations:
        return HandlerResult(SlotDirectives(), (), ())
    obs = {o.obs_id: o for o in observations}
    rules = inputs.policy.rules(observations[0].relation_type_id)
    baseline, runs, mine = _baseline_sides(inputs, rules, obs)
    results = _same_slot_results(inputs, rules, baseline, runs, mine, obs=obs)
    results.extend(_cross_slot_results(inputs, rules, mine, obs))
    return _combine_results(results)
