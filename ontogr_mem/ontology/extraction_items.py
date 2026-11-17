"""Turn extracted entities and facts into the items the ontology manager validates."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ontogr_mem.data_models.facts import ExtractedFact, TypedFact
from ontogr_mem.ontology.errors import DenseIdError
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import UNKNOWN_LABEL
from ontogr_mem.ontology.projection import Projection
from ontogr_mem.ontology.validate import EntityItem, ExtractedItems, RelationItem


# Reason codes for facts whose relation could not be stabilized. They match the
# fact extraction codes without making ontology depend on fact_extraction.
_UNKNOWN_RELATION = "unknown_relation"
_BAD_TYPE_ID = "bad_type_id"


@dataclass(frozen=True, slots=True)
class StabilizeResult:
    """Facts with a stable relation type, facts dropped, and drop counts.

    Attributes:
        typed: Kept facts in input order, each with its stable relation type or
            None.
        dropped: Facts whose relation could not be stabilized, in input order.
        counts: Number of drops per reason code.
    """

    typed: tuple[TypedFact, ...] = ()
    dropped: tuple[ExtractedFact, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)


def _is_label_link(fact: ExtractedFact) -> bool:
    """True for a fact whose label can name a relation between two refs."""
    refs = fact.entity_refs
    return (
        fact.relation_type_id is None
        and bool(fact.relation_label)
        and len(refs) == 2
        and refs[0] != refs[1]
    )


def _typed(
    fact: ExtractedFact, relation_type: int | None, view: Projection
) -> TypedFact:
    """Attach a stable relation type and the projection version to a fact."""
    return TypedFact(
        fact=fact,
        relation_type=relation_type,
        catalog_version=str(view.catalog_version),
    )


async def stabilize_facts(
    manager: OntologyManager,
    facts: Sequence[ExtractedFact],
    view: Projection,
) -> StabilizeResult:
    """Map each fact's relation to a stable type id.

    A dense id is mapped through `view`. A free-text label is checked by the
    manager in one validation pass per episode. Its endpoints are checked as
    Unknown, so a label that names a relation with patterns is dropped. A label
    without a stable type is dropped and counted.

    Args:
        manager: The ontology manager that checks labels.
        facts: The extracted facts, in their original order.
        view: The projection the facts' dense ids came from.

    Returns:
        The typed facts, the dropped facts and the drop counts.
    """
    outcomes: list[TypedFact | None] = [None] * len(facts)
    dropped_at: set[int] = set()
    counts: Counter[str] = Counter()
    label_links: dict[str, list[tuple[int, ExtractedFact]]] = {}

    for index, fact in enumerate(facts):
        if _is_label_link(fact):
            label_links.setdefault(fact.episode_id, []).append((index, fact))
            continue
        if fact.relation_type_id is None:
            outcomes[index] = _typed(fact, None, view)
            continue
        try:
            stable = view.stable_relation_id(fact.relation_type_id)
        except DenseIdError:
            dropped_at.add(index)
            counts[_BAD_TYPE_ID] += 1
        else:
            outcomes[index] = _typed(fact, stable, view)

    for group in label_links.values():
        refs = list(dict.fromkeys(ref for _, fact in group for ref in fact.entity_refs))
        entities = tuple(
            EntityItem(local_id=ref, raw_type=UNKNOWN_LABEL) for ref in refs
        )
        relations = tuple(
            RelationItem(
                source=fact.entity_refs[0],
                target=fact.entity_refs[1],
                raw_label=fact.relation_label,
            )
            for _, fact in group
        )
        report = await manager.validate_extraction(
            ExtractedItems(entities=entities, relations=relations)
        )
        for (index, fact), verdict in zip(group, report.relations, strict=True):
            if not verdict.kept or verdict.type_id is None:
                dropped_at.add(index)
                counts[_UNKNOWN_RELATION] += 1
            else:
                outcomes[index] = _typed(fact, verdict.type_id, view)

    typed = tuple(outcome for outcome in outcomes if outcome is not None)
    dropped = tuple(fact for index, fact in enumerate(facts) if index in dropped_at)
    return StabilizeResult(typed=typed, dropped=dropped, counts=dict(counts))
