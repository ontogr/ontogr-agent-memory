"""Per-tenant, versioned ontology with change sets.

The ontology is a versioned model of the types a tenant knows: entity types,
relation types and the patterns a relation may follow. A change set is the only way
to change it, and ``apply_changeset`` is the only commit path::

    from ontogr_mem.ontology import AddEntity, ChangeSet, apply_changeset

    change_set = ChangeSet(base_version=head.version, ops=[AddEntity(label="Gadget")])
    new = apply_changeset(head, change_set)

``OntologyManager`` owns the ontology of a tenant and its history; ``validate``
checks one episode's extraction against it.
"""

from ontogr_mem.ontology.apply import apply_changeset, check_catalog
from ontogr_mem.ontology.errors import (
    DenseIdError,
    InvalidChange,
    MissingVersionError,
    OntologyError,
    VersionConflict,
)
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import (
    AddAlias,
    AddEntity,
    AddPattern,
    AddRelation,
    ChangeSet,
    Channel,
    DeprecatePattern,
    DeprecateType,
    Entity,
    Kind,
    MergeInto,
    Ontology,
    Op,
    Origin,
    Pattern,
    PatternSpec,
    Reactivate,
    Reason,
    Ref,
    Relation,
    Rename,
    SetDescription,
    SetExclusive,
    Status,
)
from ontogr_mem.ontology.validate import (
    EntityItem,
    EntityValidation,
    ExtractedItems,
    RelationItem,
    RelationValidation,
    ValidationReport,
    validate,
)


__all__ = [
    "AddAlias",
    "AddEntity",
    "AddPattern",
    "AddRelation",
    "ChangeSet",
    "Channel",
    "DenseIdError",
    "DeprecatePattern",
    "DeprecateType",
    "Entity",
    "EntityItem",
    "EntityValidation",
    "ExtractedItems",
    "InvalidChange",
    "Kind",
    "MergeInto",
    "MissingVersionError",
    "Ontology",
    "OntologyError",
    "OntologyManager",
    "Op",
    "Origin",
    "Pattern",
    "PatternSpec",
    "Reactivate",
    "Reason",
    "Ref",
    "Relation",
    "RelationItem",
    "RelationValidation",
    "Rename",
    "SetDescription",
    "SetExclusive",
    "Status",
    "ValidationReport",
    "VersionConflict",
    "apply_changeset",
    "check_catalog",
    "validate",
]
