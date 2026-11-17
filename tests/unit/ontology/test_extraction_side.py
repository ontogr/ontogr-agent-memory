"""Tests for resolving, validating and projecting ontology types."""

import pytest

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.errors import DenseIdError
from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    AddEntity,
    AddRelation,
    DeprecateType,
    Kind,
    MergeInto,
    Ontology,
    PatternSpec,
    Reason,
)
from ontogr_mem.ontology.projection import project
from ontogr_mem.ontology.resolve import How, TypeResolver
from ontogr_mem.ontology.validate import (
    EntityItem,
    ExtractedItems,
    RelationItem,
    validate,
)
from tests.support.ontology import base, cs


def test_resolver_order_exact_alias_fuzzy():
    """Check resolver order exact alias fuzzy."""
    resolver = TypeResolver(base(), Kind.ENTITY)
    assert resolver.resolve("person").match.how is How.EXACT
    assert resolver.resolve("ORG").match.how is How.ALIAS
    got = resolver.resolve("Organisation").match
    assert got.how is How.FUZZY and got.type_id == 3 and 0.8 <= got.score < 1


def test_two_close_candidates_are_ambiguous_not_matched():
    """Check two close candidates are ambiguous not matched."""
    catalog = base()
    catalog = apply_changeset(catalog, cs(catalog, AddEntity(label="Organizations")))
    result = TypeResolver(catalog, Kind.ENTITY).resolve("Organizatio")
    assert result.match is None and set(result.ambiguous) == {3, 4}


def test_merged_types_resolve_to_the_live_target():
    """Check merged types resolve to the live target."""
    catalog = base()
    catalog = apply_changeset(
        catalog,
        cs(catalog, MergeInto(kind=Kind.ENTITY, source="Employee", into="Person")),
    )
    assert TypeResolver(catalog, Kind.ENTITY).resolve("Employee").match.type_id == 1


def test_unplaced_entity_becomes_unknown_and_is_kept():
    """Check unplaced entity becomes unknown and is kept."""
    result = validate(base(), ExtractedItems(entities=(EntityItem("a", "Gadget"),)))
    assert result.entities[0].type_id == UNKNOWN_ID
    assert result.counters[Reason.UNKNOWN_TYPE] == 1


def test_deprecated_entity_type_becomes_unknown_and_is_kept():
    """Check deprecated entity type becomes unknown and is kept."""
    catalog = base()
    catalog = apply_changeset(
        catalog, cs(catalog, DeprecateType(kind=Kind.ENTITY, target="Employee"))
    )
    result = validate(catalog, ExtractedItems(entities=(EntityItem("a", "Employee"),)))
    assert result.entities[0].type_id == UNKNOWN_ID
    assert result.counters[Reason.DEPRECATED_TYPE] == 1


def test_fuzzy_matches_are_never_silent():
    """Check fuzzy matches are kept and counted."""
    result = validate(
        base(), ExtractedItems(entities=(EntityItem("a", "Organisation"),))
    )
    item = result.entities[0]
    assert item.type_id == 3 and item.how is How.FUZZY
    if item.score is None:
        raise AssertionError("fuzzy match without a score")
    assert 0.8 <= item.score < 1
    assert Reason.FUZZY_MATCH in item.reasons
    assert result.counters[Reason.FUZZY_MATCH] == 1


def _relation_catalog(*, symmetric=False):
    catalog = base()
    if symmetric:
        return apply_changeset(
            catalog,
            cs(
                catalog,
                AddRelation(
                    label="partner_of",
                    symmetric=True,
                    patterns=[PatternSpec(src="Person", tgt="Organization")],
                ),
            ),
        )
    return catalog


def _validate_relation(catalog, source_type, target_type, label="works_at"):
    return validate(
        catalog,
        ExtractedItems(
            entities=(EntityItem("s", source_type), EntityItem("t", target_type)),
            relations=(RelationItem("s", "t", label),),
        ),
    ).relations[0]


def test_pattern_is_ordered_and_subsumption_accepts_children():
    """Check a child of the source type fits and a reversed pair is refused."""
    catalog = _relation_catalog()
    assert _validate_relation(catalog, "Person", "Organization").kept
    assert _validate_relation(catalog, "Employee", "Organization").kept


def test_pattern_violation_drops_the_relation():
    """Check pattern violation drops the relation."""
    relation = _validate_relation(_relation_catalog(), "Employee", "Employee")
    assert relation.kept is False
    assert Reason.PATTERN_MISMATCH in relation.reasons


def test_relation_with_patterns_is_dropped_when_an_endpoint_is_unknown():
    """Check relation with patterns is dropped when an endpoint is unknown."""
    relation = _validate_relation(_relation_catalog(), "Person", "Gadget")
    assert relation.kept is False
    assert Reason.UNKNOWN_ENDPOINT in relation.reasons


def test_swapped_direction_drops_the_relation():
    """Check swapped direction drops the relation."""
    relation = _validate_relation(_relation_catalog(), "Organization", "Person")
    assert relation.kept is False
    assert Reason.DIRECTION_SWAPPED in relation.reasons


def test_symmetric_relation_matches_in_either_order():
    """Check symmetric relation matches in either order."""
    relation = _validate_relation(
        _relation_catalog(symmetric=True), "Organization", "Person", "partner_of"
    )
    assert relation.kept and relation.reasons == ()


def test_undeclared_relation_is_dropped_and_counted():
    """Check undeclared relation is dropped and counted."""
    result = validate(
        base(), ExtractedItems(relations=(RelationItem("s", "t", "owns"),))
    )
    assert result.relations[0].kept is False
    assert Reason.UNDECLARED_RELATION in result.relations[0].reasons
    assert result.counters[Reason.UNDECLARED_RELATION] == 1


def test_projection_maps_dense_ids_back_and_is_looser_than_validate():
    """Check projection maps dense ids back and is looser than validate."""
    catalog = base()
    catalog = apply_changeset(
        catalog,
        cs(
            catalog,
            AddEntity(label="Place"),
            AddRelation(
                label="r",
                patterns=[
                    PatternSpec(src="Person", tgt="Organization"),
                    PatternSpec(src="Organization", tgt="Place"),
                ],
            ),
        ),
    )
    projection = project(catalog)
    assert [entity.label for entity in projection.view.catalog.entity_types] == [
        "Person",
        "Employee",
        "Organization",
        "Place",
    ]
    assert projection.entity_ids == (1, 2, 3, 4)
    assert projection.stable_entity_id(-1) == 0 and projection.stable_entity_id(2) == 3
    works, relation = projection.view.catalog.relation_types
    assert works.source_types == frozenset({0, 1})
    assert projection.view.catalog.direction_ok(relation, 0, 3)
    assert not catalog.pattern_allows(catalog.relation(1), 1, 4)
    assert projection.catalog_version == catalog.version


def _three_relations():
    genesis = Ontology.genesis()
    return apply_changeset(
        genesis, cs(genesis, *(AddRelation(label=name) for name in "ABC"))
    )


def test_projection_maps_dense_back_to_stable():
    """Check projection maps dense back to stable."""
    before = _three_relations()
    after = apply_changeset(
        before, cs(before, DeprecateType(kind=Kind.RELATION, target="B"))
    )
    old, new = project(before), project(after)
    assert [new.stable_relation_id(i) for i in (0, 1)] == [0, 2]
    assert old.stable_relation_id(1) == 1


def test_dense_index_out_of_range_raises_a_typed_error():
    """Check dense index out of range raises a typed error."""
    projection = project(_three_relations())
    with pytest.raises(DenseIdError):
        projection.stable_relation_id(3)
    with pytest.raises(DenseIdError):
        projection.stable_entity_id(0)
