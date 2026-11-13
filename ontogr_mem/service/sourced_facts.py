"""Text-level checks for facts that a host extracted from its own source text.

A host sends the source text and the facts it read from it. These functions verify
each fact against that text without a language model and without any I/O.
`ground_host_facts` checks that the quote and both names appear in the text and
gives each distinct endpoint a local ref. `bind_host_facts` binds the grounded
facts with the shared fact binding, merges exact duplicates and assigns ids.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from ontogr_mem.data_models import Episode, ExtractedEntity, Method, TypeView
from ontogr_mem.data_models.facts import (
    Evidence,
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
)
from ontogr_mem.fact_extraction.binding import bind
from ontogr_mem.fact_extraction.ids import assign_ids
from ontogr_mem.fact_extraction.models import FactDropReason
from ontogr_mem.fact_extraction.parsing import RawFact
from ontogr_mem.fact_extraction.text import locate, normalize_text
from ontogr_mem.service.contracts import HostFact
from ontogr_mem.service.formatting import default_statement


@dataclass(frozen=True, slots=True)
class Drop:
    """A host fact that failed a check.

    Attributes:
        index: Position of the fact in the input list.
        reason: Why the fact was dropped.
        detail: Short explanation that never contains source text.
    """

    index: int
    reason: FactDropReason
    detail: str = ""

    def as_result(self) -> dict[str, str | int]:
        """Return the drop as the plain mapping reported to callers."""
        return {"index": self.index, "reason": self.reason.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A distinct subject or object across the grounded facts.

    Attributes:
        ref: Local id such as `h0`, assigned in first-occurrence order.
        name: Name as first written, without outer whitespace.
        type_label: Type label as first written, without outer whitespace.
    """

    ref: str
    name: str
    type_label: str


@dataclass(frozen=True, slots=True)
class GroundedFact:
    """A host fact that passed the text checks.

    Attributes:
        index: Position of the fact in the input list.
        fact: The host fact as it was sent.
        subject_ref: Ref of the subject endpoint.
        object_ref: Ref of the object endpoint.
    """

    index: int
    fact: HostFact
    subject_ref: str
    object_ref: str


@dataclass(frozen=True, slots=True)
class Grounded:
    """Outcome of `ground_host_facts`.

    Attributes:
        facts: Facts that passed, in input order.
        endpoints: The endpoints that the kept facts refer to, by ref.
        drops: Dropped facts, ordered by index.
    """

    facts: tuple[GroundedFact, ...]
    endpoints: tuple[Endpoint, ...]
    drops: tuple[Drop, ...]


@dataclass(frozen=True, slots=True)
class BindInput:
    """A grounded fact together with the ids the caller resolved for it.

    Attributes:
        index: Position of the fact in the original input list.
        fact: The host fact.
        subject_ref: Ref of the subject entity.
        object_ref: Ref of the object entity.
        relation_type_id: Dense relation type id chosen by the caller.
    """

    index: int
    fact: HostFact
    subject_ref: str
    object_ref: str
    relation_type_id: int


@dataclass(frozen=True, slots=True)
class BoundHostFacts:
    """Outcome of `bind_host_facts`.

    Attributes:
        facts: Bound facts with ids, in input order.
        drops: Dropped and merged facts, ordered by index.
    """

    facts: tuple[ExtractedFact, ...]
    drops: tuple[Drop, ...]


def _first_problem(
    episode_text: str, fact: HostFact
) -> tuple[FactDropReason, str] | None:
    """Return the reason and detail of the first failed text check, if any."""
    for name, value in (
        ("subject", fact.subject),
        ("subject_type", fact.subject_type),
        ("relation", fact.relation),
        ("object", fact.object),
        ("object_type", fact.object_type),
        ("evidence_quote", fact.evidence_quote),
    ):
        if not value.strip():
            return FactDropReason.MISSING_FIELD, name
    if locate(episode_text, fact.evidence_quote) is None:
        return FactDropReason.UNGROUNDED, "quote not found in episode"
    if locate(episode_text, fact.subject) is None:
        return FactDropReason.UNGROUNDED, "subject not found in episode"
    if locate(episode_text, fact.object) is None:
        return FactDropReason.UNGROUNDED, "object not found in episode"
    return None


def ground_host_facts(episode_text: str, facts: Sequence[HostFact]) -> Grounded:
    """Check host facts against the source text and give their endpoints refs.

    Each fact gets at most one drop, from the first check it fails: a blank
    required field, a quote that is not in the text, then a subject or object
    name that is not in the text. Among the facts that pass, a name that appears
    with more than one type drops every fact that uses it, whatever the input
    order. Case and whitespace differences are ignored when comparing text.

    Args:
        episode_text: The source text in NFC, as returned by `text_of`.
        facts: The host facts, in the order the host sent them.

    Returns:
        The facts that passed with their refs, the endpoints those refs name and
        every drop. A fact whose subject and object share a ref is kept.
    """
    drops: list[Drop] = []
    survivors: list[tuple[int, HostFact]] = []
    for index, fact in enumerate(facts):
        problem = _first_problem(episode_text, fact)
        if problem is None:
            survivors.append((index, fact))
        else:
            drops.append(Drop(index, *problem))

    types_by_name: dict[str, set[str]] = {}
    for _, fact in survivors:
        for name, type_label in (
            (fact.subject, fact.subject_type),
            (fact.object, fact.object_type),
        ):
            types_by_name.setdefault(normalize_text(name), set()).add(
                normalize_text(type_label)
            )
    ambiguous = {name for name, types in types_by_name.items() if len(types) > 1}

    refs: dict[tuple[str, str], str] = {}
    endpoints: list[Endpoint] = []
    grounded: list[GroundedFact] = []
    for index, fact in survivors:
        if {normalize_text(fact.subject), normalize_text(fact.object)} & ambiguous:
            drops.append(
                Drop(
                    index,
                    FactDropReason.AMBIGUOUS_ENTITY_REF,
                    "entity name used with more than one type",
                )
            )
            continue
        pair: list[str] = []
        for name, type_label in (
            (fact.subject, fact.subject_type),
            (fact.object, fact.object_type),
        ):
            key = (normalize_text(name), normalize_text(type_label))
            if key not in refs:
                refs[key] = f"h{len(refs)}"
                endpoints.append(Endpoint(refs[key], name.strip(), type_label.strip()))
            pair.append(refs[key])
        grounded.append(GroundedFact(index, fact, pair[0], pair[1]))

    return Grounded(
        tuple(grounded),
        tuple(endpoints),
        tuple(sorted(drops, key=lambda drop: drop.index)),
    )


def _iso(moment: datetime | None) -> str | None:
    """Return an ISO 8601 string for a datetime, or None."""
    return None if moment is None else moment.isoformat()


def bind_host_facts(
    episode: Episode,
    inputs: Sequence[BindInput],
    entities: Sequence[ExtractedEntity],
    view: TypeView | None,
) -> BoundHostFacts:
    """Bind grounded host facts to the episode and merge exact duplicates.

    Each input goes through the same binding as model-extracted facts. That
    binding clears a bad type or time instead of dropping the fact, but a host
    fact must be kept whole or not at all, so any problem it reports drops the
    input with the first reported reason. Kept facts get `Method.RECORD`.
    The binding requires a time phrase from the text for any stated time. A fact
    with `valid_from` or `valid_to` uses its evidence quote, which is already
    grounded in the episode, as that phrase. A fact without times has none.
    Inputs that are the same fact merge into the first one. The quote of a later
    duplicate becomes extra evidence when it differs from the kept quote.

    Args:
        episode: The episode that holds the source text.
        inputs: Grounded facts with their refs and relation type ids.
        entities: Entities whose `local_id` values are the refs used in `inputs`.
        view: The type view used to check relation types, or None.

    Returns:
        The kept facts with ids and every drop, ordered by input index.
    """
    drops: list[Drop] = []
    unique: dict[tuple[object, ...], tuple[int, ExtractedFact]] = {}
    extra: dict[tuple[object, ...], list[Evidence]] = {}

    for item in inputs:
        fact = item.fact
        statement = fact.statement
        raw = RawFact(
            text=(
                statement
                if statement and statement.strip()
                else default_statement(fact.subject, fact.relation, fact.object)
            ),
            entity_refs=(item.subject_ref, item.object_ref),
            relation_type_id=item.relation_type_id,
            fact_kind=FactKind.STATE,
            polarity=Polarity.AFFIRMED,
            modality=Modality.ACTUAL,
            evidence_quote=fact.evidence_quote,
            valid_from=_iso(fact.valid_from),
            valid_to=_iso(fact.valid_to),
            time_expression=(
                fact.evidence_quote if fact.valid_from or fact.valid_to else None
            ),
        )
        bound = bind(episode, entities, [raw], view, default_confidence=1.0)
        # `bind` records a drop for every raw fact it does not return, so an
        # empty `dropped` means exactly one fact came back.
        if bound.dropped:
            first = bound.dropped[0]
            drops.append(Drop(item.index, first.reason, first.detail))
            continue
        kept = replace(bound.facts[0], method=Method.RECORD)

        key = (
            normalize_text(kept.text),
            kept.entity_refs,
            kept.relation_type_id,
            kept.valid_from,
            kept.valid_to,
            kept.precision,
        )
        if key not in unique:
            unique[key] = (item.index, kept)
            extra[key] = []
            continue
        first_index, first_fact = unique[key]
        drops.append(
            Drop(
                item.index,
                FactDropReason.DUPLICATE_MERGED,
                f"same fact as input index {first_index}",
            )
        )
        quote = kept.evidence_quote
        if quote != first_fact.evidence_quote and all(
            quote != seen.quote for seen in extra[key]
        ):
            extra[key].append(Evidence(quote, kept.evidence_span))

    merged = [
        replace(fact, extra_evidence=tuple(extra[key]))
        for key, (_, fact) in unique.items()
    ]
    return BoundHostFacts(
        tuple(assign_ids(episode.id, merged)),
        tuple(sorted(drops, key=lambda drop: drop.index)),
    )
