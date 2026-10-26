"""Turn typed facts into observations.

A typed fact is a sentence-level claim whose entity references are local to one
episode. Reconciliation wants an observation: a dated statement about a slot,
with stable entity ids and relation type ids. This module is the translation
and the only place that knows both shapes.

A fact becomes an observation only when it is a state, actual, typed, has
exactly two refs, both refs resolve and its interval is valid. Every other fact
is returned with a ``SkipReason`` and counted, never silently dropped.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ontogr_mem.data_models.episode import Episode
from ontogr_mem.data_models.facts import (
    FactKind,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
    TypedFact,
)
from ontogr_mem.data_models.ids import stable_id
from ontogr_mem.data_models.observation import (
    NO_STAMP,
    Observation,
    ObservationKind,
    ProvenanceStamp,
)
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.errors import EpisodeMismatchError
from ontogr_mem.temporal.timeutil import span_end


class EntityRefResolver(Protocol):
    """Maps an episode-local entity ref to a stable entity id."""

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the entity id, or ``None`` when the ref cannot be resolved."""
        ...


class SkipReason(StrEnum):
    """Why a fact did not become an observation.

    ``END_ONLY`` is a negated state that carries an end but no start ("did not
    live in Paris until 2022"). It says the state began later, so turning it
    into a stated end would close a real fact wrongly.
    """

    NO_EVIDENCE = "no_evidence"
    NOT_STATE = "not_state"
    NOT_ACTUAL = "not_actual"
    NO_RELATION = "no_relation"
    REF_COUNT = "ref_count"
    UNRESOLVED_REF = "unresolved_ref"
    END_ONLY = "end_only"
    STATED_END_UNDATED = "stated_end_undated"
    BAD_INTERVAL = "bad_interval"


@dataclass(frozen=True, slots=True)
class Skipped:
    """A fact that was not turned into an observation.

    Attributes:
        fact_id: Fact that was skipped.
        reason: Why it was skipped.
        detail: Extra context, such as the ref count.
    """

    fact_id: str
    reason: SkipReason
    detail: str = ""


def _slot_key(
    subject_id: str,
    relation_type_id: int,
    object_id: str,
    rules: RelationRules,
) -> str:
    """Build the slot identity for a relation's cardinality.

    A ``MANY`` slot adds the object, so values never compete. A ``ONE`` slot
    omits it, so a new object replaces the old one. An inverse slot is keyed on
    the object instead of the subject and carries a direction tag, so it cannot
    collide with a subject-keyed slot.
    """
    relation = str(relation_type_id)
    if rules.inverse:
        return stable_id("slot-inverse", object_id, relation)
    if rules.cardinality is Cardinality.MANY:
        return stable_id("slot", subject_id, relation, object_id)
    return stable_id("slot", subject_id, relation)


async def observations_from_facts(
    facts: Sequence[TypedFact],
    episode: Episode,
    resolver: EntityRefResolver,
    policy: DictRelationPolicy,
    stamp: ProvenanceStamp = NO_STAMP,
) -> tuple[list[Observation], list[Skipped]]:
    """Convert the typed facts of one episode into observations.

    Args:
        facts: Typed facts from the episode. The relation type id is taken from
            ``TypedFact.relation_type``, never from the fact's dense id.
        episode: Episode the facts came from.
        resolver: Resolves episode-local entity refs to entity ids.
        policy: Relation rules that choose the slot key.
        stamp: Extractor provenance copied onto each observation.

    Returns:
        The observations and the facts that were skipped, each with a reason.

    Raises:
        TypeError: If an item is an ``ExtractedFact`` rather than a
            ``TypedFact``. A dense relation id is only meaningful with the
            projection that issued it.
        EpisodeMismatchError: If a fact belongs to another episode. Converting
            it would stamp the wrong source and dates.
    """
    out: list[Observation] = []
    skipped: list[Skipped] = []
    for item in facts:
        if not isinstance(item, TypedFact):
            msg = (
                "temporal adapter input must be TypedFact; "
                "map the dense relation id to a stable id first"
            )
            raise TypeError(msg)
        fact = item.fact
        if fact.episode_id != episode.id:
            raise EpisodeMismatchError(fact.fact_id, fact.episode_id)
        if not fact.evidence_quote.strip():
            skipped.append(Skipped(fact.fact_id, SkipReason.NO_EVIDENCE))
        elif fact.fact_kind is not FactKind.STATE:
            skipped.append(Skipped(fact.fact_id, SkipReason.NOT_STATE))
        elif fact.modality is not Modality.ACTUAL:
            skipped.append(Skipped(fact.fact_id, SkipReason.NOT_ACTUAL))
        elif item.relation_type is None:
            skipped.append(Skipped(fact.fact_id, SkipReason.NO_RELATION))
        elif len(fact.entity_refs) != 2:
            skipped.append(
                Skipped(
                    fact.fact_id,
                    SkipReason.REF_COUNT,
                    str(len(fact.entity_refs)),
                )
            )
        else:
            converted = await _convert(
                item,
                episode,
                resolver,
                policy,
                stamp,
                relation_type_id=item.relation_type,
            )
            if isinstance(converted, Observation):
                out.append(converted)
            else:
                skipped.append(converted)
    return out, skipped


async def _convert(
    typed: TypedFact,
    episode: Episode,
    resolver: EntityRefResolver,
    policy: DictRelationPolicy,
    stamp: ProvenanceStamp,
    *,
    relation_type_id: int,
) -> Observation | Skipped:
    """Build one observation from a qualifying fact, or say why there is none.

    Args:
        typed: The qualifying fact.
        episode: Episode the fact came from.
        resolver: Resolves episode-local entity refs to entity ids.
        policy: Relation rules that choose the slot key.
        stamp: Extractor provenance copied onto the observation.
        relation_type_id: Stable relation type id of the fact.

    Returns:
        The observation, or the fact with the reason it was skipped. Nothing is
        raised for bad data.
    """
    fact = typed.fact
    subject = await resolver.entity_id_for_ref(fact.episode_id, fact.entity_refs[0])
    obj = await resolver.entity_id_for_ref(fact.episode_id, fact.entity_refs[1])
    if subject is None or obj is None:
        return Skipped(fact.fact_id, SkipReason.UNRESOLVED_REF)

    ended = fact.polarity is Polarity.NEGATED
    kind = ObservationKind.STATED_END if ended else ObservationKind.ASSERTION
    precision = fact.precision
    basis = fact.time_basis
    start = fact.valid_from if basis is not TimeBasis.NONE else None
    end: datetime | None = None
    try:
        if basis is not TimeBasis.NONE and start is None:
            # An affirmed end-only state ("until 2022") ends the value at the
            # period's exclusive end. A negated one says the state began
            # later, so it is not an end.
            if ended or fact.valid_to is None:
                return Skipped(fact.fact_id, SkipReason.END_ONLY)
            kind = ObservationKind.STATED_END
            start = span_end(fact.valid_to, precision or Precision.TIME)
        elif ended:
            if start is None:
                return Skipped(fact.fact_id, SkipReason.STATED_END_UNDATED)
        elif start is not None and fact.valid_to is not None:
            # An extracted end is cut to the start of its period. The stated
            # end covers the whole period, so it is the period's exclusive end.
            end = span_end(fact.valid_to, precision or Precision.TIME)
    except (ValueError, OverflowError) as exc:
        return Skipped(fact.fact_id, SkipReason.BAD_INTERVAL, str(exc))

    rules = policy.rules(relation_type_id)
    try:
        return Observation(
            obs_id=stable_id("obs", episode.id, fact.fact_id, kind.value),
            slot_key=_slot_key(subject, relation_type_id, obj, rules),
            subject_id=subject,
            object_id=obj,
            relation_type_id=relation_type_id,
            kind=kind,
            episode_id=episode.id,
            occurred_at=episode.occurred_at,
            ingested_at=episode.ingested_at,
            valid_from=start,
            valid_from_precision=precision if start is not None else None,
            valid_to=end,
            valid_to_precision=precision if end is not None else None,
            time_basis=basis,
            provenance=stamp,
            statement=fact.text,
            quote=fact.evidence_quote,
            inverse=rules.inverse,
        )
    except ValueError as exc:
        return Skipped(fact.fact_id, SkipReason.BAD_INTERVAL, str(exc))
