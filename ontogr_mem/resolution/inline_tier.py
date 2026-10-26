"""The inline verification tier.

The tier runs the rules without verdicts, asks the verifier about the ids the
rules could not settle, and hands the verdicts back to the resolver. The rules
it runs are the same ones the resolver runs later, so the ids it asks about
are the ids the resolver waits for.
"""

from collections.abc import Sequence

from ontogr_mem.data_models import Clock, Usage
from ontogr_mem.data_models.entities import (
    Entity,
    Outcome,
    VerdictKind,
    VerifyStatus,
)
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.decision import collect_matches
from ontogr_mem.resolution.grouping import Group
from ontogr_mem.resolution.policy import ResolutionPolicy, TypeRel, TypeRelations
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.resolution.scoring import Candidate, needs_two_votes
from ontogr_mem.resolution.verdicts import apply_verdicts
from ontogr_mem.resolution.verifier import PairInput, Verifier
from ontogr_mem.resolution.views import view_for


VERIFY_CAP = 3
"""Most candidates one group sends to the verifier."""


async def inline_verify(
    groups: Sequence[Group],
    candidates: Sequence[Sequence[Candidate]],
    *,
    store: ResolverStore,
    policy: ResolutionPolicy,
    settings: ResolverSettings,
    relations: TypeRelations,
    verifier: Verifier,
    incomplete: bool,
    clock: Clock,
) -> dict[int, dict[str, VerdictKind]]:
    """Dry-run the rules, verify what they could not decide, return the verdicts."""
    pairs: list[PairInput] = []
    owners: list[tuple[int, str]] = []
    for index, group in enumerate(groups):
        view = await view_for(store, group, candidates[index])
        collected = collect_matches(
            group,
            candidates[index],
            view,
            policy=policy,
            settings=settings,
            relations=relations,
        )
        plan = apply_verdicts(collected, {}, incomplete)
        if plan.outcome is Outcome.MATCHED or not plan.verify:
            continue
        available = dict(view.candidates)
        for entity, _ in view.alias_hits:
            available[entity.entity_id] = entity
        by_id = {candidate.entity_id: candidate for candidate in candidates[index]}
        for entity_id in plan.verify[:VERIFY_CAP]:
            entity = available.get(entity_id)
            if entity is None:
                continue
            candidate = by_id.get(entity_id)
            cross = (
                policy.relation(group.type_label, entity.type_label, relations)
                is TypeRel.FAMILY
            )
            two_votes = cross or (
                candidate is not None
                and needs_two_votes(candidate.signals, settings, cross)
            )
            pairs.append(
                PairInput(
                    _pseudo(index, group, clock),
                    entity,
                    two_votes=two_votes,
                    cosine=None if candidate is None else candidate.signals.cosine,
                    jaccard=None if candidate is None else candidate.signals.jaccard,
                )
            )
            owners.append((index, entity_id))
    outcomes, _ = await verifier.verify(pairs, Usage())
    found: dict[int, dict[str, VerdictKind]] = {}
    for (index, entity_id), outcome in zip(owners, outcomes, strict=True):
        if outcome.status is VerifyStatus.OK and outcome.verdict in (
            VerdictKind.SAME,
            VerdictKind.DISTINCT,
        ):
            found.setdefault(index, {})[entity_id] = outcome.verdict
    return found


def _pseudo(index: int, group: Group, clock: Clock) -> Entity:
    now = clock.now()
    return Entity(
        entity_id=f"new:{index}",
        type_label=group.type_label,
        canonical_name=group.name,
        strict_key=group.strict,
        loose_key=group.loose,
        summary=group.descriptions[0] if group.descriptions else "",
        identity_attrs=dict(group.identity),
        created_at=now,
        updated_at=now,
    )
