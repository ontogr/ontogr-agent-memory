"""Cardinality rules for active ontology relations."""

from ontogr_mem.ontology.models import Ontology, RelationKind, Status
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)


def relation_policy(ontology: Ontology) -> DictRelationPolicy:
    """Build stable-id cardinality rules from active ontology relations."""
    return DictRelationPolicy(
        {
            relation.type_id: RelationRules(Cardinality.ONE)
            for relation in ontology.relations
            if relation.status is Status.ACTIVE and relation.kind is RelationKind.SINGLE
        }
    )
