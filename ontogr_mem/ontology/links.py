"""Type compatibility and identity names, read from one ontology snapshot."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ontogr_mem.data_models import UNKNOWN_LABEL
from ontogr_mem.ontology.models import UNKNOWN_ID, Ontology


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

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> "TypeLinks":
        """Index the entity types of ``ontology`` once."""
        by_id = {entity.type_id: entity for entity in ontology.entities}
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
        for type_id in parents:
            chain: list[int] = []
            current = parents[type_id]
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
        return cls(canonical, labels, ancestors, identities)

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
