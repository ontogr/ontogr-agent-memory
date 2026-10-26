"""Operations on relation definitions stored in the ontology."""

from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import (
    AddRelation,
    ChangeSet,
    Mode,
    Ontology,
    Relation,
    SetExclusive,
    Status,
)
from ontogr_mem.service.defaults import DEFAULT_CATALOG
from ontogr_mem.service.errors import InvalidArgumentError
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)


async def ensure_relation(
    manager: OntologyManager,
    label: str,
    *,
    single_valued: bool | None = None,
) -> Relation:
    """Find or declare a relation, optionally changing its exclusivity."""
    ontology = await manager.catalog()
    relation = ontology.relation_by_label(label)
    if relation is None:
        if ontology.relation_mode is Mode.STRICT:
            raise InvalidArgumentError(
                f"relation label is not in the ontology: {label}"
            )
        change = ChangeSet(
            base_version=ontology.version,
            ops=[AddRelation(label=label, exclusive=bool(single_valued))],
        )
        ontology = await manager.apply(change)
        relation = ontology.relation_by_label(label)
    elif single_valued is not None and relation.exclusive != single_valued:
        change = ChangeSet(
            base_version=ontology.version,
            ops=[SetExclusive(target=relation.type_id, exclusive=single_valued)],
        )
        ontology = await manager.apply(change)
        relation = ontology.relation(relation.type_id)
    if relation is None:
        raise InvalidArgumentError(f"relation label is not in the ontology: {label}")
    return relation


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
