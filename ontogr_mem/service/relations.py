"""Relation rules and the default ontology for a new tenant."""

from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import Ontology, Status
from ontogr_mem.service.defaults import DEFAULT_CATALOG
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
            if relation.status is Status.ACTIVE and relation.exclusive
        }
    )


async def load_defaults(manager: OntologyManager) -> Ontology:
    """Apply the default catalog only to a tenant at genesis."""
    ontology = await manager.catalog()
    if ontology.version == 0:
        return await manager.apply(DEFAULT_CATALOG.model_copy(deep=True))
    return ontology
