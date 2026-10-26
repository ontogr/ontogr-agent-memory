"""Validate and bind a parsed reply to the candidates, the types and the episode.

Nothing is discarded silently: every dropped fact, and every cleared part of a
kept fact, gets a reason code in `dropped`.

`bind` turns the model's raw facts into `ExtractedFact` objects:

1. refs must come from the supplied candidate list;
2. the evidence quote must be found in the current episode;
3. an invalid relation type is cleared, and an unusable time is cleared;
4. exact duplicates are merged, and ids are assigned by code.

The ids come from `ids.py`, the relation type check from `type_check.py` and the
reason codes from `models.py`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ontogr_mem.data_models import (
    Episode,
    ExtractedEntity,
    Kind,
    Method,
    Span,
    TypeView,
)
from ontogr_mem.data_models.facts import Evidence, ExtractedFact
from ontogr_mem.fact_extraction.ids import assign_ids
from ontogr_mem.fact_extraction.models import FactDropped, FactDropReason
from ontogr_mem.fact_extraction.parsing import RawFact
from ontogr_mem.fact_extraction.text import locate, normalize_text, text_of
from ontogr_mem.fact_extraction.timeparse import resolve_time
from ontogr_mem.fact_extraction.type_check import check_relation_type


def asserted_by(episode: Episode) -> str | None:
    """Return the actor for messages and None for other kinds."""
    return episode.actor.id if episode.kind is Kind.MESSAGE else None


def _info(raw: RawFact) -> dict[str, Any]:
    """Return the raw item fields kept on a drop."""
    return {
        "text": raw.text,
        "entity_refs": list(raw.entity_refs),
        "relation_type_id": raw.relation_type_id,
        "relation_label": raw.relation_label,
    }


@dataclass(slots=True)
class Bound:
    """Outcome of binding."""

    facts: list[ExtractedFact]
    dropped: list[FactDropped]


def bind(
    episode: Episode,
    entities: Sequence[ExtractedEntity],
    raws: Sequence[RawFact],
    view: TypeView | None,
    *,
    default_confidence: float,
    previous: Sequence[Episode] = (),
) -> Bound:
    """Check each raw fact and return the kept facts with ids and the drops.

    Args:
        episode: The episode the facts were extracted from.
        entities: The candidate entities the model could refer to.
        raws: The parsed facts.
        view: The type view the model used, or None.
        default_confidence: Confidence used when the model gives none.
        previous: Earlier episodes shown as context, used only to explain an
            ungrounded quote.

    Returns:
        The kept facts with ids and every drop.
    """
    by_id = {entity.local_id: entity for entity in entities}
    dropped: list[FactDropped] = []
    # `kept` holds the first fact for each key, in model order. `merged` collects
    # the quotes of later exact duplicates.
    kept: dict[tuple[object, ...], ExtractedFact] = {}
    merged: dict[tuple[object, ...], list[Evidence]] = {}

    for raw in raws:
        info = _info(raw)
        fact = _bind_one(
            episode,
            raw,
            info,
            entities=by_id,
            view=view,
            default_confidence=default_confidence,
            previous=previous,
            dropped=dropped,
        )
        if fact is None:
            continue
        key: tuple[object, ...] = (
            normalize_text(fact.text),
            fact.entity_refs,
            fact.relation_type_id,
            fact.relation_label,
            fact.fact_kind,
            fact.polarity,
            fact.modality,
            fact.valid_from,
            fact.valid_to,
            fact.precision,
            fact.time_expression,
        )
        if key in kept:
            merged[key].append(Evidence(fact.evidence_quote, fact.evidence_span))
            dropped.append(FactDropped(info, FactDropReason.DUPLICATE_MERGED))
            continue
        kept[key] = fact
        merged[key] = []

    facts: list[ExtractedFact] = []
    for key, fact in kept.items():
        # Keep each distinct extra quote once; a quote equal to the fact's own
        # quote is not extra evidence.
        extra: list[Evidence] = []
        for evidence in merged[key]:
            if evidence.quote != fact.evidence_quote and all(
                evidence.quote != seen.quote for seen in extra
            ):
                extra.append(evidence)
        facts.append(replace(fact, extra_evidence=tuple(extra)))
    return Bound(assign_ids(episode.id, facts), dropped)


def _bind_one(
    episode: Episode,
    raw: RawFact,
    info: dict[str, Any],
    *,
    entities: Mapping[str, ExtractedEntity],
    view: TypeView | None,
    default_confidence: float,
    previous: Sequence[Episode],
    dropped: list[FactDropped],
) -> ExtractedFact | None:
    """Check one raw fact; return it, or None after recording a drop."""
    text = " ".join(raw.text.split())
    if not text:
        dropped.append(FactDropped(info, FactDropReason.EMPTY_TEXT))
        return None

    unknown = [ref for ref in raw.entity_refs if ref not in entities]
    if unknown:
        dropped.append(
            FactDropped(info, FactDropReason.UNKNOWN_ENTITY_REF, ", ".join(unknown))
        )
        return None

    episode_text = text_of(episode)
    span: Span | None = locate(episode_text, raw.evidence_quote)
    if span is None:
        elsewhere = any(
            locate(text_of(earlier), raw.evidence_quote) for earlier in previous
        )
        detail = (
            "quote found only in a previous episode"
            if elsewhere
            else "quote not found in episode"
        )
        dropped.append(FactDropped(info, FactDropReason.UNGROUNDED, detail))
        return None

    check = check_relation_type(raw.relation_type_id, raw.entity_refs, entities, view)
    if check.problem is not None:
        dropped.append(FactDropped(info, *check.problem))

    when = resolve_time(
        valid_from=raw.valid_from,
        valid_to=raw.valid_to,
        declared=raw.precision,
        expression=raw.time_expression,
        kind=raw.fact_kind,
        modality=raw.modality,
        episode_text=episode_text,
        reference=episode.occurred_at,
    )
    dropped.extend(
        FactDropped(info, reason, detail) for reason, detail in when.problems
    )

    label = (raw.relation_label or "").strip() or None
    return ExtractedFact(
        fact_id="",
        text=text,
        entity_refs=tuple(dict.fromkeys(raw.entity_refs)),
        relation_type_id=check.type_id,
        relation_label=label,
        fact_kind=raw.fact_kind,
        polarity=raw.polarity,
        modality=raw.modality,
        asserted_by=asserted_by(episode),
        valid_from=when.valid_from,
        valid_to=when.valid_to,
        precision=when.precision,
        time_expression=when.time_expression,
        time_basis=when.basis,
        # The quote is stored as the episode's own text at the matched span.
        evidence_quote=episode_text[span[0] : span[1]],
        evidence_span=span,
        confidence=raw.confidence if raw.confidence is not None else default_confidence,
        method=Method.LLM,
        episode_id=episode.id,
    )
