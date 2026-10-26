"""Tests for applying change sets to the catalog."""

import pytest

from ontogr_mem.ontology.apply import apply_changeset, check_catalog
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.errors import InvalidChange, VersionConflict
from ontogr_mem.ontology.models import (
    AddAlias,
    AddAttribute,
    AddEntity,
    AddPattern,
    AddRelation,
    AddTable,
    Attribute,
    Channel,
    DeprecateAttribute,
    DeprecatePattern,
    DeprecateType,
    Kind,
    MergeInto,
    Mode,
    Ontology,
    Owner,
    PatternSpec,
    Reactivate,
    Rename,
    SetDescription,
    SetExclusive,
    SetModes,
    Status,
    TableMapping,
    ValueType,
)
from tests.support.ontology import base, cs


def test_genesis_holds_only_unknown_at_id_zero():
    """Genesis holds only unknown at id zero."""
    g = Ontology.genesis()
    assert [(e.type_id, e.label) for e in g.entities] == [(0, "Unknown")]
    assert g.version == 0 and g.relations == []


def test_json_round_trip_keeps_version_and_hash():
    """Json round trip keeps version and hash."""
    cat = base()
    back = Ontology.model_validate_json(cat.model_dump_json())
    assert back == cat and back.version == 1 and back.content_hash == cat.content_hash


def test_identical_content_hashes_the_same_whatever_the_version():
    """Identical content hashes the same whatever the version."""
    a = base()
    b = apply_changeset(
        a,
        cs(
            a,
            SetDescription(
                kind=Kind.ENTITY, target="Person", description=a.entity(1).description
            ),
        ),
    )
    assert b.version == 2 and b.content_hash == a.content_hash


def test_stale_base_version_raises_version_conflict():
    """Stale base version raises version conflict."""
    a = base()
    with pytest.raises(VersionConflict):
        apply_changeset(a, a_cs := cs(a).model_copy(update={"base_version": 0}))
    assert a_cs.base_version == 0


@pytest.mark.parametrize(
    "label", ["Entity", "unknown", "Chunk", "1abc", "has space", "a__b", "x" * 65]
)
def test_illegal_or_reserved_entity_labels_are_refused(label):
    """Illegal or reserved entity labels are refused."""
    a = base()
    with pytest.raises(InvalidChange):
        apply_changeset(a, cs(a, AddEntity(label=label)))


def test_names_colliding_after_normalization_are_refused():
    """Names colliding after normalization are refused."""
    a = base()
    with pytest.raises(InvalidChange, match="collides"):
        apply_changeset(a, cs(a, AddEntity(label="person")))
    with pytest.raises(InvalidChange, match="collides"):
        apply_changeset(
            a, cs(a, AddAlias(kind=Kind.ENTITY, target="Organization", alias="PERSON"))
        )


def test_reserved_attribute_name_is_refused():
    """Reserved attribute name is refused."""
    a = base()
    with pytest.raises(InvalidChange, match="reserved attribute"):
        apply_changeset(
            a,
            cs(
                a,
                AddAttribute(
                    kind=Kind.ENTITY, target="Person", attribute=Attribute(name="uuid")
                ),
            ),
        )


def test_retyping_an_attribute_is_refused_and_the_same_one_is_a_no_op():
    """Retyping an attribute is refused and the same one is a no op."""
    a = base()
    same = Attribute(name="age", value_type=ValueType.INTEGER)
    b = apply_changeset(
        a, cs(a, AddAttribute(kind=Kind.ENTITY, target="Person", attribute=same))
    )
    assert b.content_hash == a.content_hash
    with pytest.raises(InvalidChange, match="different definition"):
        apply_changeset(
            a,
            cs(
                a,
                AddAttribute(
                    kind=Kind.ENTITY,
                    target="Person",
                    attribute=Attribute(name="age", value_type=ValueType.STRING),
                ),
            ),
        )


def test_additive_attribute_on_an_existing_type_is_allowed():
    """Additive attribute on an existing type is allowed."""
    a = base()
    b = apply_changeset(
        a,
        cs(
            a,
            AddAttribute(
                kind=Kind.ENTITY, target="Person", attribute=Attribute(name="city")
            ),
        ),
    )
    assert [p.name for p in b.entity(1).properties][-1] == "city"


def test_unknown_cannot_be_edited_or_removed():
    """Unknown cannot be edited or removed."""
    a = base()
    for op in (
        Rename(kind=Kind.ENTITY, target=0, new_label="Other"),
        DeprecateType(kind=Kind.ENTITY, target=0),
        SetDescription(kind=Kind.ENTITY, target=0, description="x"),
        MergeInto(kind=Kind.ENTITY, source=0, into="Person"),
    ):
        with pytest.raises(InvalidChange, match="Unknown"):
            apply_changeset(a, cs(a, op))


def test_hierarchy_cycle_and_depth_are_refused():
    """Hierarchy cycle and depth are refused."""
    a = base()
    # a cycle can only be built by hand: Person's parent becomes Employee
    broken = a.model_copy(deep=True)
    broken.entity(1).parent_id = 2

    with pytest.raises(InvalidChange, match="cycle"):
        check_catalog(broken)
    cur = a
    for i in range(7):
        cur = apply_changeset(
            cur,
            cs(
                cur,
                AddEntity(label=f"L{i}", parent="Employee" if i == 0 else f"L{i - 1}"),
            ),
        )
    with pytest.raises(InvalidChange, match="deeper"):
        apply_changeset(cur, cs(cur, AddEntity(label="TooDeep", parent="L6")))


def test_pattern_endpoint_must_exist():
    """Pattern endpoint must exist."""
    a = base()
    with pytest.raises(InvalidChange, match="no entity type"):
        apply_changeset(
            a,
            cs(
                a,
                AddPattern(
                    relation="works_at", pattern=PatternSpec(src="Person", tgt="Nope")
                ),
            ),
        )


def test_active_pattern_cannot_end_at_a_deprecated_type():
    """Active pattern cannot end at a deprecated type."""
    a = base()
    with pytest.raises(InvalidChange, match="deprecate the pattern first"):
        apply_changeset(
            a, cs(a, DeprecateType(kind=Kind.ENTITY, target="Organization"))
        )
    b = apply_changeset(
        a,
        cs(
            a,
            DeprecatePattern(
                relation="works_at",
                pattern=PatternSpec(src="Person", tgt="Organization"),
            ),
            DeprecateType(kind=Kind.ENTITY, target="Organization"),
        ),
    )
    assert b.entity(3).status is Status.DEPRECATED


def test_strict_over_an_empty_kind_is_refused():
    """Strict over an empty kind is refused."""
    g = Ontology.genesis()
    with pytest.raises(InvalidChange, match="no active types"):
        apply_changeset(g, cs(g, SetModes(entity_mode=Mode.STRICT)))
    a = base()
    assert (
        apply_changeset(a, cs(a, SetModes(entity_mode=Mode.STRICT))).entity_mode
        is Mode.STRICT
    )


def test_deprecated_type_is_kept_and_can_be_reactivated():
    """Deprecated type is kept and can be reactivated."""
    a = base()
    b = apply_changeset(a, cs(a, DeprecateType(kind=Kind.ENTITY, target="Employee")))
    assert b.entity(2).status is Status.DEPRECATED and len(b.entities) == len(
        a.entities
    )
    c = apply_changeset(b, cs(b, Reactivate(kind=Kind.ENTITY, target="Employee")))
    assert c.entity(2).status is Status.ACTIVE


def test_rename_keeps_the_old_label_as_an_alias():
    """Rename keeps the old label as an alias."""
    a = base()
    b = apply_changeset(
        a, cs(a, Rename(kind=Kind.ENTITY, target="Organization", new_label="Company"))
    )
    e = b.entity(3)
    assert (e.label, sorted(e.aliases)) == ("Company", ["Org", "Organization"])


def test_merge_collapses_chains_and_refuses_cycles():
    """Merge collapses chains and refuses cycles."""
    a = base()
    b = apply_changeset(a, cs(a, AddEntity(label="Staff"), AddEntity(label="Worker")))
    c = apply_changeset(
        b, cs(b, MergeInto(kind=Kind.ENTITY, source="Staff", into="Worker"))
    )
    d = apply_changeset(
        c, cs(c, MergeInto(kind=Kind.ENTITY, source="Worker", into="Employee"))
    )
    assert d.entity(4).merged_into == 2 and d.entity(5).merged_into == 2
    assert d.follow(Kind.ENTITY, 4) == 2
    with pytest.raises(InvalidChange, match="cycle|not active"):
        apply_changeset(
            d, cs(d, MergeInto(kind=Kind.ENTITY, source="Employee", into="Staff"))
        )


def test_a_merge_pointer_to_a_type_that_is_not_there_is_refused():
    """A merge pointer to a type that is not there is refused."""
    a = base()
    b = apply_changeset(a, cs(a, AddEntity(label="Staff"), AddEntity(label="Worker")))
    c = apply_changeset(
        b, cs(b, MergeInto(kind=Kind.ENTITY, source="Staff", into="Worker"))
    )
    broken = c.model_copy(deep=True)
    worker = next(e for e in broken.entities if e.label == "Worker")
    worker.merged_into = 99

    with pytest.raises(InvalidChange, match="bad merge pointer"):
        check_catalog(broken)


def test_relation_by_label_matches_normalized_aliases_but_not_deprecated_types():
    """Relation lookup folds spaces and underscores and skips retired rows."""
    catalog = base()
    catalog = apply_changeset(
        catalog,
        cs(catalog, AddAlias(kind=Kind.RELATION, target="works_at", alias="employs")),
    )
    relation = catalog.relation_by_label("EMPLOYS")

    assert relation is not None
    assert relation.type_id == 0
    assert catalog.relation_by_label("Org") is None
    deprecated = catalog.model_copy(deep=True)
    deprecated.relations[0].status = Status.DEPRECATED

    assert deprecated.relation_by_label("works_at") is None


def test_apply_changeset_rejects_a_merge_that_follows_a_dangling_pointer():
    """A dangling target pointer raises InvalidChange, not an internal error."""
    a = base()
    b = apply_changeset(a, cs(a, AddEntity(label="Staff"), AddEntity(label="Worker")))
    broken = b.model_copy(deep=True)
    worker = next(e for e in broken.entities if e.label == "Worker")
    worker.merged_into = 99

    with pytest.raises(InvalidChange, match="bad merge pointer"):
        apply_changeset(
            broken,
            cs(
                broken,
                MergeInto(kind=Kind.ENTITY, source="Staff", into="Worker"),
            ),
        )


def test_table_mapping_rules():
    """Table mapping rules."""
    a = base()
    tbl = Attribute(name="hr_id", owner=Owner.TABLE)
    b = apply_changeset(
        a,
        cs(
            a,
            AddAttribute(kind=Kind.ENTITY, target="Person", attribute=tbl),
            channel=Channel.TABLE,
        ),
    )
    assert (
        b.entity(1).properties[-1].structured
        and not b.entity(1).properties[-1].extractable
    )
    m = TableMapping(source="hr.people", entity_id=1, columns={"id": "hr_id"})
    c = apply_changeset(b, cs(b, AddTable(table=m)))
    with pytest.raises(InvalidChange, match="mapping names"):
        apply_changeset(c, cs(c, DeprecateType(kind=Kind.ENTITY, target="Person")))
    with pytest.raises(InvalidChange, match="table mapping uses"):
        apply_changeset(
            c,
            cs(c, DeprecateAttribute(kind=Kind.ENTITY, target="Person", name="hr_id")),
        )
    with pytest.raises(InvalidChange, match="share a source"):
        apply_changeset(
            c, cs(c, AddTable(table=m.model_copy(update={"columns": {"x": "hr_id"}})))
        )
    with pytest.raises(InvalidChange, match="not table-owned"):
        apply_changeset(
            b,
            cs(
                b,
                AddTable(
                    table=TableMapping(source="t", entity_id=1, columns={"c": "age"})
                ),
            ),
        )


def test_a_proposal_cannot_add_table_owned_attributes():
    """A proposal cannot add table owned attributes."""
    a = base()
    with pytest.raises(InvalidChange, match="proposal cannot add table-owned"):
        apply_changeset(
            a,
            cs(
                a,
                AddAttribute(
                    kind=Kind.ENTITY,
                    target="Person",
                    attribute=Attribute(name="x", owner=Owner.TABLE),
                ),
                channel=Channel.PROPOSAL,
            ),
        )


def test_diff_replayed_on_the_old_catalog_gives_the_new_content():
    """Diff replayed on the old catalog gives the new content."""
    a = base()
    b = apply_changeset(
        a,
        cs(
            a,
            AddEntity(
                label="Place",
                description="A location.",
                properties=[Attribute(name="lat", value_type=ValueType.FLOAT)],
            ),
            AddRelation(
                label="located_in",
                symmetric=False,
                patterns=[PatternSpec(src="Organization", tgt="Place")],
            ),
            Rename(kind=Kind.ENTITY, target="Organization", new_label="Company"),
            AddAlias(kind=Kind.ENTITY, target="Person", alias="Human"),
            SetDescription(
                kind=Kind.RELATION, target="works_at", description="Employment."
            ),
            AddAttribute(
                kind=Kind.ENTITY, target="Person", attribute=Attribute(name="city")
            ),
            DeprecateAttribute(kind=Kind.ENTITY, target="Person", name="nickname"),
            AddPattern(
                relation="works_at", pattern=PatternSpec(src="Employee", tgt="Company")
            ),
            SetModes(relation_mode=Mode.STRICT),
        ),
    )
    c = apply_changeset(a, diff(a, b))
    assert c.content_hash == b.content_hash
    assert diff(b, b).ops == []


_GOLD_EDGES = {
    "LIVES_IN": True,
    "WORKS_AT": True,
    "MANAGES": False,
    "LIKES": False,
    "CONSIDERING_JOINING": False,
    "RELOCATING_TO": False,
}


def _excl(cat):
    """Excl."""
    return {r.label: r.exclusive for r in cat.relations}


def test_exclusive_flag_registers_diffs_and_round_trips():
    """Exclusive flag registers diffs and round trips."""
    g = Ontology.genesis()
    a = apply_changeset(
        g,
        cs(g, *(AddRelation(label=k, exclusive=v) for k, v in _GOLD_EDGES.items())),
    )
    assert {
        r.label: r.exclusive for r in a.relations if r.label != "Unknown"
    } == _GOLD_EDGES
    b = apply_changeset(a, cs(a, SetExclusive(target="LIKES", exclusive=True)))
    assert _excl(b)["LIKES"] and not _excl(a)["LIKES"]
    replay = apply_changeset(a, diff(a, b))
    assert replay.content_hash == b.content_hash
    assert _excl(Ontology.model_validate_json(b.model_dump_json()))["LIKES"]


def test_set_exclusive_refuses_unknown_and_missing_relation():
    """Set exclusive refuses unknown and missing relation."""
    a = base()
    with pytest.raises(InvalidChange):
        apply_changeset(a, cs(a, SetExclusive(target="Unknown", exclusive=True)))
    with pytest.raises(InvalidChange):
        apply_changeset(a, cs(a, SetExclusive(target="nope", exclusive=True)))
