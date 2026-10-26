"""Relation rules: how the values of a relation relate to one another.

The temporal layer reads relation rules from a mapping and does not
import the ontology catalog. An undeclared relation is ``MANY`` with the
``REVIEW`` conflict policy: a new value never closes an old one by itself, and
disagreements are reported rather than hidden.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Cardinality(StrEnum):
    """Which side of a fact a new value replaces.

    ``ONE``: one object per subject, so a new object replaces the old one.
    ``ONE_PER_OBJECT``: one subject per object, so the slot is keyed on the
    object. ``MANY``: values coexist.
    """

    ONE = "one"
    ONE_PER_OBJECT = "one_per_object"
    MANY = "many"


class ConflictPolicy(StrEnum):
    """What may be done about a disagreement on this relation."""

    AUTO = "auto"
    REVIEW = "review"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class RelationRules:
    """Everything reconciliation needs to know about one relation type.

    Attributes:
        cardinality: Whether a new value replaces the active one.
        conflict_policy: What the contradiction handler may do with a
            disagreement.
    """

    cardinality: Cardinality = Cardinality.MANY
    conflict_policy: ConflictPolicy = ConflictPolicy.REVIEW

    @property
    def inverse(self) -> bool:
        """True when the slot is keyed on the object."""
        return self.cardinality is Cardinality.ONE_PER_OBJECT

    @property
    def single_valued(self) -> bool:
        """True when a new value cuts the active one."""
        return self.cardinality is not Cardinality.MANY


class DictRelationPolicy:
    """Policy backed by a mapping; undeclared relations get the defaults."""

    def __init__(self, declared: Mapping[int, RelationRules] | None = None) -> None:
        """Copy the declarations so later edits to the mapping have no effect.

        Args:
            declared: Rules keyed by relation type id.
        """
        self._declared = dict(declared or {})

    def rules(self, relation_type_id: int) -> RelationRules:
        """Return the declared rules, or the many/review defaults."""
        return self._declared.get(relation_type_id, RelationRules())
