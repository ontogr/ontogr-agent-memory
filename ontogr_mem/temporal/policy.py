"""Build a relation policy from the view extraction used.

The ontology stores only one cardinality flag: ``exclusive``. Everything else
is temporal behavior. The caller reads the ontology and passes the dense ids of
exclusive relations; this module maps them to ``ONE`` and leaves every other
relation ``MANY`` with the ``REVIEW`` conflict policy. Rules are keyed by the
stable relation type id, which is what the adapter reads from each fact.
"""

from collections.abc import Callable, Iterable

from ontogr_mem.data_models.type_catalog import TypeView
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.errors import UnknownRelationError


def from_type_view(
    view: TypeView,
    exclusive_relation_ids: Iterable[int] = (),
    stable_relation_id: Callable[[int], int] = lambda dense_id: dense_id,
) -> DictRelationPolicy:
    """Build a policy for a projection view.

    An exclusive relation is single-valued: one object per subject, so a new
    object replaces the active one. Every other relation keeps the default
    ``MANY`` cardinality and ``REVIEW`` conflict policy.

    Args:
        view: The catalog view whose dense relation ids the caller uses.
        exclusive_relation_ids: Dense ids of relations the ontology marks
            exclusive.
        stable_relation_id: Maps a dense id to the stable relation type id. Pass
            the projection's ``stable_relation_id``; the default keeps the ids as
            they are.

    Returns:
        A policy that returns ``ONE`` for each exclusive relation and ``MANY`` for
        every other relation, keyed by stable id.

    Raises:
        UnknownRelationError: If an id is not a relation type in the view.
    """
    declared: dict[int, RelationRules] = {}
    for dense_id in exclusive_relation_ids:
        if not view.catalog.has_relation_id(dense_id):
            raise UnknownRelationError(dense_id)
        declared[stable_relation_id(dense_id)] = RelationRules(
            cardinality=Cardinality.ONE
        )
    return DictRelationPolicy(declared)
