"""Decide one group, up to the point where a verdict is needed.

This is the first half of the decision. The uri, alias, key and fuzzy rules
can settle a group on their own; what they cannot settle is collected here as
the ids that still need a verdict and the reason they need one.
``ontogr_mem.resolution.verdicts`` holds the second half, the verdicts
applied to what this module collected.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ontogr_mem.data_models.entities import (
    DecisionMethod,
    Entity,
    EntityStatus,
    Outcome,
)
from ontogr_mem.resolution.blocking import has_high_entropy
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.grouping import Group
from ontogr_mem.resolution.keys import differ_only_in_digits
from ontogr_mem.resolution.plan import Plan
from ontogr_mem.resolution.policy import (
    ResolutionMode,
    ResolutionPolicy,
    TypeRel,
    TypeRelations,
    identity_agree,
    identity_conflict,
)
from ontogr_mem.resolution.scoring import Candidate
from ontogr_mem.resolution.views import DecisionView


_MATCHABLE = frozenset({EntityStatus.ACTIVE, EntityStatus.ARCHIVED})
"""Statuses a new mention may match; an archived match is restored."""

_Relation = Callable[[Entity], TypeRel]
_Veto = Callable[[Entity], bool]


@dataclass(frozen=True, slots=True)
class Collected:
    """What one group needs before its verdicts are known.

    Attributes:
        plan: The plan when a rule settled the group on its own.
        verify: The ids that no rule could settle.
        reason: Why the group needs a verdict.
    """

    plan: Plan | None
    verify: tuple[str, ...]
    reason: str


def collect_matches(
    group: Group,
    candidates: Sequence[Candidate],
    view: DecisionView,
    *,
    policy: ResolutionPolicy,
    settings: ResolverSettings,
    relations: TypeRelations,
) -> Collected:
    """Collect what the rules can say about one group, without asking a model."""
    mode = policy.mode(group.type_label)

    def relation(entity: Entity) -> TypeRel:
        return policy.relation(group.type_label, entity.type_label, relations)

    def vetoed(entity: Entity) -> bool:
        if identity_conflict(group.identity, entity.identity_attrs):
            return True
        return bool(group.uri and entity.uri and group.uri != entity.uri)

    by_uri = _uri_plan(group, view)
    if by_uri is not None:
        return Collected(plan=by_uri, verify=(), reason="")
    by_key, direct, family = _direct_plan(group, view, mode, relation, vetoed)
    if by_key is not None:
        return Collected(plan=by_key, verify=(), reason="")
    verify, reason = _verify_ids(direct, family, mode)
    by_fuzzy, verify, reason = _scan_candidates(
        group,
        candidates,
        view,
        direct=direct,
        verify=verify,
        reason=reason,
        mode=mode,
        settings=settings,
        relation=relation,
        vetoed=vetoed,
    )
    if by_fuzzy is not None:
        return Collected(plan=by_fuzzy, verify=(), reason="")
    return Collected(plan=None, verify=tuple(verify), reason=reason)


def _uri_plan(group: Group, view: DecisionView) -> Plan | None:
    """Return a plan for a bound uri, or None when no active uri entity exists."""
    if not group.uri:
        return None
    for entity in view.uri_entities:
        if entity.status in _MATCHABLE:
            return Plan(
                entity.entity_id, DecisionMethod.URI, Outcome.MATCHED, "uri_hit"
            )
    return None


def _direct_plan(
    group: Group,
    view: DecisionView,
    mode: ResolutionMode,
    relation: _Relation,
    vetoed: _Veto,
) -> tuple[Plan | None, list[tuple[Entity, bool]], list[Entity]]:
    """Decide from alias hits, or return the direct and family lists."""
    direct: list[tuple[Entity, bool]] = []
    family: list[Entity] = []
    for entity, canonical in view.alias_hits:
        kind = relation(entity)
        if kind in (TypeRel.SAME, TypeRel.NESTED) and not vetoed(entity):
            direct.append((entity, canonical))
        elif kind is TypeRel.FAMILY and not vetoed(entity):
            family.append(entity)
    if not direct:
        return None, direct, family
    agree = [
        (entity, canonical)
        for entity, canonical in direct
        if identity_agree(group.identity, entity.identity_attrs)
    ]
    if mode is ResolutionMode.AUTO_KEY and len(direct) == 1:
        entity, canonical = direct[0]
        return (
            Plan(
                entity.entity_id,
                DecisionMethod.KEY if canonical else DecisionMethod.ALIAS,
                Outcome.MATCHED,
                "key_hit" if canonical else "alias_hit",
            ),
            direct,
            family,
        )
    if len(agree) == 1:
        entity, canonical = agree[0]
        return (
            Plan(
                entity.entity_id,
                DecisionMethod.KEY if canonical else DecisionMethod.ALIAS,
                Outcome.MATCHED,
                "identity_attr",
            ),
            direct,
            family,
        )
    return None, direct, family


def _verify_ids(
    direct: Sequence[tuple[Entity, bool]],
    family: Sequence[Entity],
    mode: ResolutionMode,
) -> tuple[list[str], str]:
    """Return the ids that need verification and the reason for it."""
    verify = [entity.entity_id for entity, _ in direct]
    reason = ""
    if direct:
        reason = (
            "verify_always" if mode is ResolutionMode.VERIFY_ALWAYS else "ambiguous_key"
        )
    verify += [entity.entity_id for entity in family]
    if family and not reason:
        reason = "cross_type"
    return verify, reason


def _scan_candidates(
    group: Group,
    candidates: Sequence[Candidate],
    view: DecisionView,
    *,
    direct: Sequence[tuple[Entity, bool]],
    verify: Sequence[str],
    reason: str,
    mode: ResolutionMode,
    settings: ResolverSettings,
    relation: _Relation,
    vetoed: _Veto,
) -> tuple[Plan | None, list[str], str]:
    """Add fuzzy and verification candidates from the ranked list."""
    direct_ids = {entity.entity_id for entity, _ in direct}
    pending = list(verify)
    fuzzy: list[tuple[float, Entity]] = []
    for candidate in candidates:
        if candidate.entity_id in direct_ids or candidate.entity_id in pending:
            continue
        entity = view.candidates.get(candidate.entity_id)
        if entity is None or entity.status not in _MATCHABLE:
            continue
        kind = relation(entity)
        if (
            kind is TypeRel.DISTINCT
            or vetoed(entity)
            or differ_only_in_digits(group.name, entity.canonical_name)
        ):
            continue
        if _fuzzy_candidate(
            group, candidate, kind, direct=direct, mode=mode, settings=settings
        ):
            fuzzy.append((candidate.signals.jaccard or 0.0, entity))
        else:
            pending.append(entity.entity_id)
    if not fuzzy:
        return None, pending, reason
    fuzzy.sort(key=lambda item: (-item[0], item[1].entity_id))
    if len(fuzzy) == 1 or fuzzy[0][0] > fuzzy[1][0]:
        score, entity = fuzzy[0]
        return (
            Plan(
                entity.entity_id,
                DecisionMethod.FUZZY,
                Outcome.MATCHED,
                "fuzzy_jaccard",
                score=score,
            ),
            pending,
            reason,
        )
    pending += [entity.entity_id for _, entity in fuzzy]
    return None, pending, reason or "ambiguous_fuzzy"


def _fuzzy_candidate(
    group: Group,
    candidate: Candidate,
    kind: TypeRel,
    *,
    direct: Sequence[tuple[Entity, bool]],
    mode: ResolutionMode,
    settings: ResolverSettings,
) -> bool:
    """Return whether a candidate may match on Jaccard alone."""
    return (
        not direct
        and kind in (TypeRel.SAME, TypeRel.NESTED)
        and mode is ResolutionMode.AUTO_KEY
        and has_high_entropy(group.loose)
        and candidate.signals.jaccard is not None
        and candidate.signals.jaccard >= settings.fuzzy_jaccard
    )
