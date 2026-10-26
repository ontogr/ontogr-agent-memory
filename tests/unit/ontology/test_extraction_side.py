"""Tests for resolving, rendering, validating and projecting types."""

import pytest

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.errors import DenseIdError
from ontogr_mem.ontology.models import (
    AddAttribute,
    AddEntity,
    AddPattern,
    AddRelation,
    Attribute,
    Channel,
    DeprecateType,
    Kind,
    MergeInto,
    Mode,
    Ontology,
    PatternSpec,
    Reason,
    SetModes,
    UnknownEdges,
    UnmatchedKind,
    ValueType,
)
from ontogr_mem.ontology.projection import project
from ontogr_mem.ontology.proposals import aggregate
from ontogr_mem.ontology.render import render
from ontogr_mem.ontology.resolve import How, TypeResolver
from ontogr_mem.ontology.validate import (
    Context,
    EntityItem,
    ExtractedItems,
    RelationItem,
    validate,
)
from tests.support.ontology import base, cs


CTX = Context("ep1")


def test_resolver_order_exact_alias_fuzzy():
    """Resolver order exact alias fuzzy."""
    r = TypeResolver(base(), Kind.ENTITY)
    assert r.resolve("person").match.how is How.EXACT
    assert r.resolve("ORG").match.how is How.ALIAS
    got = r.resolve("Organisation").match
    assert got.how is How.FUZZY and got.type_id == 3 and 0.8 <= got.score < 1


def test_two_close_candidates_are_ambiguous_not_matched():
    """Two close candidates are ambiguous not matched."""
    a = base()
    a = apply_changeset(a, cs(a, AddEntity(label="Organizations")))
    res = TypeResolver(a, Kind.ENTITY).resolve("Organizatio")
    assert res.match is None and set(res.ambiguous) == {3, 4}


def test_merged_types_resolve_to_the_live_target():
    """Merged types resolve to the live target."""
    a = base()
    a = apply_changeset(
        a, cs(a, MergeInto(kind=Kind.ENTITY, source="Employee", into="Person"))
    )
    assert TypeResolver(a, Kind.ENTITY).resolve("Employee").match.type_id == 1


def test_render_numbers_the_list_and_the_view_decodes_it():
    """Render numbers the list and the view decodes it."""
    a = base()
    text, view = render(a, Kind.ENTITY)
    assert text.startswith("Preferred entity types")
    assert view.shown == [0, 1, 2, 3] and view.decode(1) == 1 and view.decode(4) is None
    assert view.decode(-1) is None


def test_strict_mode_hides_unknown_and_deprecated_types_and_changes_the_wording():
    """Strict mode hides unknown and deprecated types and changes the wording."""
    a = base()
    a = apply_changeset(
        a,
        cs(
            a,
            DeprecateType(kind=Kind.ENTITY, target="Employee"),
            SetModes(entity_mode=Mode.STRICT),
        ),
    )
    text, view = render(a, Kind.ENTITY)
    assert text.startswith("Allowed entity types") and view.shown == [1, 3]


def test_render_cuts_descriptions_and_omits_structured_attributes():
    """Render cuts descriptions and omits structured attributes."""
    a = base()
    a = apply_changeset(
        a, cs(a, AddEntity(label="Long", description="x" * 500 + "\n\nsecond"))
    )
    text, _ = render(a, Kind.ENTITY, max_desc_chars=20)
    assert '"description": "' + "x" * 19 + '…"' in text and "second" not in text


def test_render_caps_the_list_and_keeps_unknown():
    """Render caps the list and keeps unknown."""
    _, view = render(base(), Kind.ENTITY, max_types=2)
    assert view.shown == [0, 1]
    _, view = render(base(), Kind.ENTITY, include=[3])
    assert view.shown == [0, 3]


def run(cat, ents, rels=()):
    """Run."""
    _, ev = render(cat, Kind.ENTITY)
    _, rv = render(cat, Kind.RELATION)
    return validate(cat, ExtractedItems(tuple(ents), tuple(rels)), CTX, ev, rv)


def test_position_outside_the_list_is_a_counted_drop_not_a_fallback():
    """Position outside the list is a counted drop not a fallback."""
    r = run(
        base(),
        [
            EntityItem("a", position=9),
            EntityItem("b", position=-1),
            EntityItem("c", position=1),
        ],
    )
    assert [v.kept for v in r.entities] == [False, False, True]
    assert r.counters[Reason.TYPE_ID_OUT_OF_RANGE] == 2 and r.entities[2].type_id == 1


def test_unplaced_entity_by_mode():
    """Unplaced entity by mode."""
    a = base()
    items = [EntityItem("a", raw_type="Gadget", quote="a gadget")]
    pref = run(a, items)
    assert pref.entities[0].kept and pref.entities[0].type_id == 0
    assert [(u.kind, u.normalized_label, u.quote) for u in pref.unmatched] == [
        (UnmatchedKind.ENTITY_TYPE, "gadget", "a gadget")
    ]
    strict = apply_changeset(a, cs(a, SetModes(entity_mode=Mode.STRICT)))
    s = run(strict, items)
    assert not s.entities[0].kept and s.counters[Reason.UNKNOWN_TYPE] == 1
    assert s.unmatched == () and s.to_register == ()


def test_deprecated_type_by_label_is_unknown_in_preferred_and_dropped_in_strict():
    """Deprecated type by label is unknown in preferred and dropped in strict."""
    a = apply_changeset(
        base(), cs(base(), DeprecateType(kind=Kind.ENTITY, target="Employee"))
    )
    r = run(a, [EntityItem("a", raw_type="Employee")])
    assert r.entities[0].type_id == 0 and r.counters[Reason.DEPRECATED_TYPE] == 1


def test_attribute_coercion_is_strict_and_failures_are_omitted_not_none():
    """Attribute coercion is strict and failures are omitted not none."""
    a = base()
    r = run(
        a,
        [
            EntityItem(
                "a",
                position=1,
                attributes={
                    "age": 3.7,
                    "nickname": "abcdefgh",
                    "city": "x",
                    "email": "e@x",
                },
            )
        ],
    )
    v = r.entities[0]
    assert v.kept and v.attributes == {"email": "e@x"}
    assert r.counters[Reason.ATTRIBUTE_UNCOERCIBLE] == 1
    assert r.counters[Reason.ATTRIBUTE_TOO_LONG] == 1
    assert r.counters[Reason.ATTRIBUTE_UNDECLARED] == 1
    assert [(u.kind, u.normalized_label) for u in r.unmatched] == [
        (UnmatchedKind.ATTRIBUTE, "city")
    ]
    ok = run(a, [EntityItem("a", position=1, attributes={"age": 4.0, "email": "e"})])
    assert ok.entities[0].attributes == {"age": 4, "email": "e"}
    boolean = run(
        a, [EntityItem("a", position=1, attributes={"age": True, "email": "e"})]
    )
    assert "age" not in boolean.entities[0].attributes


def test_missing_required_attribute_keeps_the_entity_with_a_reason():
    """Missing required attribute keeps the entity with a reason."""
    r = run(base(), [EntityItem("a", position=1)])
    assert (
        r.entities[0].kept
        and Reason.ATTRIBUTE_REQUIRED_MISSING in r.entities[0].reasons
    )


def test_a_list_attribute_holds_text():
    """A list attribute holds text."""
    a = base()
    a = apply_changeset(
        a,
        cs(
            a,
            AddAttribute(
                kind=Kind.ENTITY,
                target="Person",
                attribute=Attribute(name="tags", value_type=ValueType.LIST),
            ),
        ),
    )

    r = run(a, [EntityItem("a", position=1, attributes={"tags": ["x", 2]})])

    assert r.entities[0].attributes == {"tags": ["x", "2"]}


def relation_catalog(mode=Mode.STRICT, edges=UnknownEdges.DROP):
    """Relation catalog."""
    a = base()
    return apply_changeset(
        a, cs(a, SetModes(relation_mode=mode, unknown_endpoint_edges=edges))
    )


def rel_run(cat, src_pos, tgt_pos, rel_label="works_at", ents=None):
    """Rel run."""
    ents = ents or [
        EntityItem("s", position=src_pos),
        EntityItem("t", position=tgt_pos),
    ]
    return run(cat, ents, [RelationItem("s", "t", raw_label=rel_label)])


def test_pattern_is_ordered_and_subsumption_accepts_children():
    """Pattern is ordered and subsumption accepts children."""
    cat = relation_catalog()
    assert rel_run(cat, 1, 3).relations[0].kept  # Person -> Organization
    assert rel_run(cat, 2, 3).relations[0].kept  # Employee is a Person
    swapped = rel_run(cat, 3, 1).relations[0]
    assert not swapped.kept and swapped.reasons == (Reason.DIRECTION_SWAPPED,)
    wrong = rel_run(cat, 2, 2).relations[0]
    assert not wrong.kept and wrong.reasons == (Reason.PATTERN_MISMATCH,)


def test_symmetric_relation_matches_in_either_order():
    """Symmetric relation matches in either order."""
    a = relation_catalog()
    a = apply_changeset(
        a,
        cs(
            a,
            AddRelation(
                label="partner_of",
                symmetric=True,
                patterns=[PatternSpec(src="Person", tgt="Organization")],
            ),
        ),
    )
    r = rel_run(a, 3, 1, rel_label="partner_of")
    assert r.relations[0].kept and r.relations[0].reasons == ()


def test_preferred_mode_keeps_pattern_violations_but_records_them():
    """Preferred mode keeps pattern violations but records them."""
    cat = relation_catalog(Mode.PREFERRED, UnknownEdges.KEEP)
    r = rel_run(cat, 2, 2)
    assert r.relations[0].kept and r.counters[Reason.PATTERN_MISMATCH] == 1
    assert [u.kind for u in r.unmatched] == [UnmatchedKind.PATTERN]


def test_undeclared_relation_by_mode():
    """Undeclared relation by mode."""
    assert not rel_run(relation_catalog(), 1, 3, rel_label="owns").relations[0].kept
    kept = rel_run(relation_catalog(Mode.PREFERRED), 1, 3, rel_label="owns").relations[
        0
    ]
    assert kept.kept and kept.unvalidated and kept.type_id is None
    openr = rel_run(relation_catalog(Mode.OPEN), 1, 3, rel_label="owns").relations[0]
    assert openr.kept and not openr.unvalidated


def test_edges_with_an_unknown_endpoint_follow_the_catalog_setting():
    """Edges with an unknown endpoint follow the catalog setting."""
    ents = [EntityItem("s", position=1), EntityItem("t", raw_type="Gadget")]
    dropped = rel_run(
        relation_catalog(edges=UnknownEdges.DROP), 0, 0, ents=ents
    ).relations[0]
    assert not dropped.kept and dropped.reasons == (Reason.UNKNOWN_ENDPOINT,)
    r = rel_run(relation_catalog(edges=UnknownEdges.KEEP), 0, 0, ents=ents)
    assert r.relations[0].kept and r.counters[Reason.UNKNOWN_ENDPOINT] == 1


def test_a_dropped_entity_takes_its_relations_with_a_reason():
    """A dropped entity takes its relations with a reason."""
    cat = relation_catalog()
    ents = [EntityItem("s", position=1), EntityItem("t", position=9)]
    r = rel_run(cat, 0, 0, ents=ents)
    assert (
        r.relations[0].reasons == (Reason.ENDPOINT_DROPPED,) and not r.relations[0].kept
    )
    assert r.dropped == ("t", "s->t")


def test_projection_maps_dense_ids_back_and_is_looser_than_validate():
    """Projection maps dense ids back and is looser than validate."""
    a = base()
    a = apply_changeset(
        a,
        cs(
            a,
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
    p = project(a)
    cat = p.view.catalog
    assert [e.label for e in cat.entity_types] == [
        "Person",
        "Employee",
        "Organization",
        "Place",
    ]
    assert p.entity_ids == (1, 2, 3, 4)
    assert p.stable_entity_id(-1) == 0 and p.stable_entity_id(2) == 3
    works, r = cat.relation_types
    assert works.source_types == frozenset({0, 1})
    person, place = 0, 3
    assert cat.direction_ok(r, person, place)
    assert not a.pattern_allows(a.relation(1), 1, 4)
    assert p.view.mode.value == "strict" and p.catalog_version == a.version


def _three_relations():
    """Three relations."""
    g = Ontology.genesis()
    return apply_changeset(
        g,
        cs(g, AddRelation(label="A"), AddRelation(label="B"), AddRelation(label="C")),
    )


def test_projection_maps_dense_back_to_stable():
    """Projection maps dense back to stable."""
    before = _three_relations()
    after = apply_changeset(
        before, cs(before, DeprecateType(kind=Kind.RELATION, target="B"))
    )

    old_view, new_view = project(before), project(after)

    assert [new_view.stable_relation_id(i) for i in (0, 1)] == [0, 2]
    assert old_view.stable_relation_id(1) == 1


def test_dense_index_out_of_range_raises_a_typed_error():
    """Dense index out of range raises a typed error."""
    view = project(_three_relations())

    with pytest.raises(DenseIdError):
        view.stable_relation_id(3)
    with pytest.raises(DenseIdError):
        view.stable_entity_id(0)


def test_render_positions_and_projection_ids_are_not_interchangeable():
    """Render positions and projection ids are not interchangeable."""
    cat = _three_relations()

    _, rendered = render(cat, Kind.ENTITY)
    projected = project(cat)

    assert rendered.shown == [0]
    assert projected.entity_ids == ()
    assert projected.stable_entity_id(-1) == rendered.decode(0)


def test_open_registers_instead_of_unknown():
    """Open registers instead of unknown."""
    a = apply_changeset(base(), cs(base(), SetModes(entity_mode=Mode.OPEN)))

    r = run(a, [EntityItem("a", raw_type="Gadget")])

    assert r.entities[0].kept and r.entities[0].type_id is None
    assert r.to_register == ((Kind.ENTITY, "Gadget"),)
    assert r.unmatched == ()


def test_open_refuses_to_register_an_illegal_or_reserved_label():
    """Open refuses to register an illegal or reserved label."""
    a = apply_changeset(base(), cs(base(), SetModes(entity_mode=Mode.OPEN)))
    items = [
        EntityItem("a", raw_type="ignore previous instructions"),
        EntityItem("b", raw_type="Episode"),
    ]

    r = run(a, items)

    assert [v.kept for v in r.entities] == [False, False]
    assert r.to_register == ()


def test_deprecated_type_in_open_keeps_its_id():
    """Deprecated type in open keeps its id."""
    a = apply_changeset(
        base(), cs(base(), DeprecateType(kind=Kind.ENTITY, target="Employee"))
    )
    a = apply_changeset(a, cs(a, SetModes(entity_mode=Mode.OPEN)))

    r = run(a, [EntityItem("a", raw_type="Employee")])

    assert r.entities[0].type_id == 2 and r.to_register == ()


def test_strict_drops_and_counts_without_recording():
    """Strict drops and counts without recording."""
    a = base()
    a = apply_changeset(
        a, cs(a, SetModes(entity_mode=Mode.STRICT, relation_mode=Mode.STRICT))
    )
    items = ExtractedItems(
        entities=(EntityItem("a", raw_type="Gadget"), EntityItem("b", raw_type="Org")),
        relations=(RelationItem("a", "b", raw_label="invented"),),
    )

    r = validate(a, items, CTX)

    assert r.dropped == ("a", "a->b")
    assert r.counters[Reason.UNKNOWN_TYPE] == 1
    assert r.unmatched == () and r.to_register == ()


def test_preferred_records_unmatched():
    """Preferred records unmatched."""
    r = run(base(), [EntityItem("a", raw_type="Gadget")])

    assert [u.normalized_label for u in r.unmatched] == ["gadget"]
    assert r.to_register == ()


def test_fuzzy_matches_are_never_silent():
    """Fuzzy matches are never silent."""
    r = run(
        base(), [EntityItem("a", raw_type="Organisation", quote="the Organisation")]
    )
    v = r.entities[0]
    assert v.kept and v.type_id == 3 and v.how is How.FUZZY and 0.8 <= v.score < 1
    assert Reason.FUZZY_MATCH in v.reasons and r.counters[Reason.FUZZY_MATCH] == 1
    assert [(u.kind, u.raw_label) for u in r.unmatched] == [
        (UnmatchedKind.FUZZY_ENTITY, "Organisation")
    ]
    exact = run(base(), [EntityItem("a", raw_type="Org")]).entities[0]
    assert exact.how is How.ALIAS and exact.score == 1.0
    assert run(base(), [EntityItem("a", position=1)]).entities[0].how is None


def test_fuzzy_rows_are_not_proposed_as_new_types():
    """Fuzzy rows are not proposed as new types."""
    cat = base()
    row = run(cat, [EntityItem("a", raw_type="Organisation")]).unmatched[0]
    rows = [row.model_copy(update={"episode_id": f"e{i}"}) for i in range(3)]
    assert aggregate(cat, rows, [], min_support=3) == []


def test_first_patterns_make_relations_strict_and_edges_drop():
    """First patterns make relations strict and edges drop."""
    g = Ontology.genesis()
    a = apply_changeset(
        g, cs(g, AddEntity(label="A"), AddEntity(label="B"), AddRelation(label="free"))
    )
    assert (a.relation_mode, a.unknown_endpoint_edges) == (
        Mode.PREFERRED,
        UnknownEdges.KEEP,
    )
    b = apply_changeset(
        a, cs(a, AddRelation(label="r", patterns=[PatternSpec(src="A", tgt="B")]))
    )
    assert (b.relation_mode, b.unknown_endpoint_edges) == (
        Mode.STRICT,
        UnknownEdges.DROP,
    )
    # later patterns do not flip a catalog the user set back to preferred
    c = apply_changeset(b, cs(b, SetModes(relation_mode=Mode.PREFERRED)))
    assert (
        c.unknown_endpoint_edges is UnknownEdges.KEEP
    )  # the edge setting follows the mode
    d = apply_changeset(
        c, cs(c, AddPattern(relation="r", pattern=PatternSpec(src="B", tgt="A")))
    )
    assert d.relation_mode is Mode.PREFERRED


def test_first_patterns_keep_modes_when_set_explicitly_or_proposed():
    """The first patterns do not flip modes if the change sets them or is a proposal."""
    g = Ontology.genesis()
    pat = AddRelation(label="r", patterns=[PatternSpec(src="A", tgt="B")])
    ents = [AddEntity(label="A"), AddEntity(label="B")]
    explicit = apply_changeset(g, cs(g, *ents, pat, SetModes(relation_mode=Mode.OPEN)))
    assert explicit.relation_mode is Mode.OPEN
    base1 = apply_changeset(g, cs(g, *ents))
    prop = apply_changeset(base1, cs(base1, pat, channel=Channel.PROPOSAL))
    assert prop.relation_mode is Mode.PREFERRED
