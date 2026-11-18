"""Turn extracted entities and facts into the items the ontology manager validates."""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.data_models.facts import ExtractedFact, TypedFact
from ontogr_mem.ontology.errors import DenseIdError
from ontogr_mem.ontology.projection import Projection


@dataclass(frozen=True, slots=True)
class StabilizeResult:
    """Facts mapped to stable relation type ids.

    Attributes:
        typed: Facts in input order, each with its stable relation type or None.
    """

    typed: tuple[TypedFact, ...] = ()


def _typed(
    fact: ExtractedFact, relation_type: int | None, view: Projection
) -> TypedFact:
    """Attach a stable relation type and the projection version to a fact."""
    return TypedFact(
        fact=fact,
        relation_type=relation_type,
        catalog_version=str(view.catalog_version),
    )


def stabilize_facts(
    facts: Sequence[ExtractedFact], view: Projection
) -> StabilizeResult:
    """Map dense relation ids to stable ids while keeping unplaced facts.

    Args:
        facts: The extracted facts, in their original order.
        view: The projection the facts' dense ids came from.

    Returns:
        The facts with stable relation ids, in input order.
    """
    outcomes: list[TypedFact] = []
    for fact in facts:
        if fact.relation_type_id is None:
            outcomes.append(_typed(fact, None, view))
            continue
        try:
            stable = view.stable_relation_id(fact.relation_type_id)
        except DenseIdError:
            outcomes.append(_typed(fact, None, view))
        else:
            outcomes.append(_typed(fact, stable, view))
    return StabilizeResult(typed=tuple(outcomes))
