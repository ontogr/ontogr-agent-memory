"""Checking a proposed relation type against the facts that use it.

A typed link relates exactly two entities, source first. The check is shared by
the model path and the record path. A problem clears the type, never the fact.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from ontogr_mem.data_models import ExtractedEntity, TypeMode, TypeView
from ontogr_mem.fact_extraction.models import FactDropReason


@dataclass(frozen=True, slots=True)
class TypeCheck:
    """Outcome of checking a proposed relation type against its refs.

    `type_id` is None when it was cleared; `problem` says why.
    """

    type_id: int | None
    problem: tuple[FactDropReason, str] | None = None


def check_relation_type(
    type_id: int | None,
    refs: tuple[str, ...],
    entities: Mapping[str, ExtractedEntity],
    view: TypeView | None,
) -> TypeCheck:
    """Check a proposed `relation_type_id`; a problem clears it, never the fact.

    A typed link relates exactly two entities, source first. Checks, in order:
    number of refs, self link, id in the catalog, direction pattern.
    """
    if type_id is None:
        return TypeCheck(None)
    if view is None:
        return TypeCheck(None, (FactDropReason.TYPES_UNAVAILABLE, f"id {type_id}"))
    problem: tuple[FactDropReason, str] | None = None
    if len(refs) != 2:
        problem = (FactDropReason.BAD_TYPE_ID, "a typed link needs exactly two refs")
    elif refs[0] == refs[1]:
        problem = (FactDropReason.SELF_RELATION, refs[0])
    elif not view.catalog.has_relation_id(type_id):
        problem = (FactDropReason.BAD_TYPE_ID, f"relation type {type_id}")
    elif view.mode is not TypeMode.OPEN and not view.catalog.direction_ok(
        view.catalog.relation_types[type_id],
        entities[refs[0]].type_id,
        entities[refs[1]].type_id,
    ):
        problem = (FactDropReason.WRONG_DIRECTION, f"relation type {type_id}")
    return TypeCheck(None if problem else type_id, problem)
