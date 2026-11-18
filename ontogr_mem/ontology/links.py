"""Type compatibility and identity names, read from one ontology snapshot."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from ontogr_mem.data_models import UNKNOWN_LABEL
from ontogr_mem.ontology.models import UNKNOWN_ID, Entity, Ontology
from ontogr_mem.ontology.names import normalize


def _canonical_entity_ids(by_id: Mapping[int, Entity]) -> dict[int, int]:
    canonical: dict[int, int] = {}
    for type_id in by_id:
        current, seen = type_id, {type_id}
        while True:
            target = by_id[current].merged_into
            if target is None or target not in by_id or target in seen:
                break
            seen.add(target)
            current = target
        canonical[type_id] = current
    return canonical


def _entity_indexes(
    ontology: Ontology,
) -> tuple[
    dict[int, int],
    dict[int, str],
    dict[int, frozenset[int]],
    dict[int, tuple[str, ...]],
    dict[str, int | None],
    dict[int, frozenset[int]],
]:
    by_id = {entity.type_id: entity for entity in ontology.entities}
    canonical = _canonical_entity_ids(by_id)
    parents = {
        type_id: (
            None
            if by_id[type_id].parent_id is None
            else canonical.get(by_id[type_id].parent_id, by_id[type_id].parent_id)
        )
        for type_id in set(canonical.values())
    }
    ancestors: dict[int, frozenset[int]] = {}
    identities: dict[int, tuple[str, ...]] = {}
    for type_id, parent_id in parents.items():
        chain: list[int] = []
        current = parent_id
        while current is not None and current not in chain and current != type_id:
            chain.append(current)
            current = parents.get(current)
        ancestors[type_id] = frozenset(chain)
        names: list[str] = []
        for member in (type_id, *chain):
            if member in by_id:
                names.extend(
                    name
                    for name in by_id[member].identity_attributes
                    if name not in names
                )
        identities[type_id] = tuple(names)
    labels = {type_id: by_id[type_id].label for type_id in parents}
    entity_labels: dict[str, int | None] = {}
    for entity in ontology.entities:
        for name in (entity.label, *entity.aliases):
            key = normalize(name)
            target = canonical[entity.type_id]
            entity_labels[key] = (
                None
                if key in entity_labels and entity_labels[key] != target
                else target
            )
    descendants: dict[int, set[int]] = {type_id: set() for type_id in parents}
    for type_id, ancestor_set in ancestors.items():
        for ancestor in ancestor_set:
            descendants.setdefault(ancestor, set()).add(type_id)
    return (
        canonical,
        labels,
        ancestors,
        identities,
        entity_labels,
        {key: frozenset(value) for key, value in descendants.items()},
    )


class TypeRel(Enum):
    """How two entity types relate when two entities share a name.

    Attributes:
        SAME: The types are one type.
        NESTED: One type is an ancestor of the other, or one is Unknown.
        DISTINCT: The types are siblings or unrelated.
    """

    SAME = "same"
    NESTED = "nested"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class TypeLinks:
    """Compatibility and identity facts of entity types in one ontology snapshot.

    Ids are followed through merges in the ontology file before they are
    compared, so a type merged into another compares as the other.
    """

    merged_to: Mapping[int, int]
    labels: Mapping[int, str]
    ancestor_sets: Mapping[int, frozenset[int]]
    identities: Mapping[int, tuple[str, ...]]
    entity_labels: Mapping[str, int | None] = field(default_factory=dict)
    relation_labels: Mapping[str, int | None] = field(default_factory=dict)
    relation_merged_to: Mapping[int, int] = field(default_factory=dict)
    descendants: Mapping[int, frozenset[int]] = field(default_factory=dict)

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> "TypeLinks":
        """Index the entity types of ``ontology`` once."""
        canonical, labels, ancestors, identities, entity_labels, descendants = (
            _entity_indexes(ontology)
        )
        relation_merged_to: dict[int, int] = {}
        relation_by_id = {item.type_id: item for item in ontology.relations}
        for type_id, relation in relation_by_id.items():
            current, seen = relation, {type_id}
            while (
                current.merged_into in relation_by_id
                and current.merged_into not in seen
            ):
                seen.add(current.merged_into)
                current = relation_by_id[current.merged_into]
            relation_merged_to[type_id] = current.type_id
        relation_labels: dict[str, int | None] = {}
        for relation in ontology.relations:
            for name in (relation.label, *relation.aliases):
                key = normalize(name)
                target = relation_merged_to[relation.type_id]
                if key in relation_labels and relation_labels[key] != target:
                    relation_labels[key] = None
                else:
                    relation_labels[key] = target
        return cls(
            canonical,
            labels,
            ancestors,
            identities,
            entity_labels,
            relation_labels,
            relation_merged_to,
            descendants,
        )

    def entity_family(self, label: str) -> frozenset[int] | None:
        """Return stored entity ids represented by a type label."""
        found = self.entity_labels.get(normalize(label))
        if found is None:
            return None
        wanted = {found, *self.descendants.get(found, frozenset())}
        return frozenset(i for i, target in self.merged_to.items() if target in wanted)

    def relation_family(self, label: str) -> frozenset[int] | None:
        """Return stored relation ids represented by a relation label."""
        found = self.relation_labels.get(normalize(label))
        if found is None:
            return None
        return frozenset(
            type_id
            for type_id, target in self.relation_merged_to.items()
            if target == found
        )

    def canonical(self, type_id: int) -> int:
        """Return the id that ``type_id`` stands for after merges in the file."""
        return self.merged_to.get(type_id, type_id)

    def label(self, type_id: int) -> str:
        """Return the current label of a type, or Unknown for an id this lacks."""
        return self.labels.get(self.canonical(type_id), UNKNOWN_LABEL)

    def relation(self, a: int, b: int) -> TypeRel:
        """Return how two types relate for a merge."""
        first, second = self.canonical(a), self.canonical(b)
        if first == second:
            return TypeRel.SAME
        if UNKNOWN_ID in (first, second):
            return TypeRel.NESTED
        if second in self.ancestor_sets.get(first, frozenset()) or first in (
            self.ancestor_sets.get(second, frozenset())
        ):
            return TypeRel.NESTED
        return TypeRel.DISTINCT

    def more_specific(self, a: int, b: int) -> int:
        """Return the canonical id of the more specific of two types.

        The descendant wins over its ancestor, a typed id wins over Unknown, and
        the first argument wins otherwise.
        """
        first, second = self.canonical(a), self.canonical(b)
        if first == UNKNOWN_ID:
            return second
        if second == UNKNOWN_ID:
            return first
        if first in self.ancestor_sets.get(second, frozenset()):
            return second
        return first

    def identity_names(self, type_id: int) -> tuple[str, ...]:
        """Return the identity names of a type and of its ancestors."""
        return self.identities.get(self.canonical(type_id), ())
