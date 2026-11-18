"""Place relation labels on ontology types after entity types are known."""

from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.data_models import ExtractedEntity
from ontogr_mem.data_models.facts import ExtractedFact
from ontogr_mem.fact_extraction.type_check import check_relation_type
from ontogr_mem.ontology.models import Kind
from ontogr_mem.ontology.projection import Projection
from ontogr_mem.ontology.resolve import TypeResolver


def resolve_relation(projection: Projection, label: str) -> int | None:
    """Return the dense id of the active relation named by ``label``.

    Exact labels, aliases, and fuzzy matches use the ontology resolver. Ambiguous,
    unknown, and deprecated matches have no dense id.
    """
    found = TypeResolver(projection.ontology, Kind.RELATION).resolve(label).match
    if found is None:
        return None
    try:
        return projection.relation_ids.index(found.type_id)
    except ValueError:
        return None


def place_relations(
    facts: Sequence[ExtractedFact],
    entities: Sequence[ExtractedEntity],
    projection: Projection,
) -> list[ExtractedFact]:
    """Place relation labels when their entity types fit the relation pattern.

    Existing relation ids remain unchanged. An id-less fact with two distinct
    refs is placed from its label only when the endpoints pass the type check.
    Other facts retain their text and remain unplaced.
    """
    by_id = {entity.local_id: entity for entity in entities}
    placed: list[ExtractedFact] = []
    for fact in facts:
        label = (fact.relation_label or "").strip()[:128]
        if fact.relation_type_id is not None or not label or len(fact.entity_refs) != 2:
            placed.append(replace(fact, relation_label=label or None))
            continue
        if fact.entity_refs[0] == fact.entity_refs[1]:
            placed.append(replace(fact, relation_label=label))
            continue
        dense = resolve_relation(projection, label)
        if dense is None:
            placed.append(replace(fact, relation_label=label))
            continue
        check = check_relation_type(dense, fact.entity_refs, by_id, projection.view)
        if check.problem is not None:
            placed.append(replace(fact, relation_label=label))
            continue
        placed.append(replace(fact, relation_type_id=dense, relation_label=None))
    return placed
