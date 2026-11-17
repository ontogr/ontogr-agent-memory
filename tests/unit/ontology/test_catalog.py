"""Tests for applying change sets to the catalog."""

import pytest

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.errors import InvalidChange, VersionConflict
from ontogr_mem.ontology.models import (
    AddAlias,
    AddEntity,
    AddPattern,
    AddRelation,
    DeprecatePattern,
    DeprecateType,
    Kind,
    MergeInto,
    Ontology,
    PatternSpec,
    Reactivate,
    RelationKind,
    Rename,
    SetDescription,
    SetIdentityAttributes,
    SetParent,
    SetRelationKind,
    SetSymmetric,
    Status,
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
        apply_changeset(broken, cs(broken))
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
        apply_changeset(broken, cs(broken))


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


def test_diff_replayed_on_the_old_catalog_gives_the_new_content():
    """Diff replayed on the old catalog gives the new content."""
    a = base()
    b = apply_changeset(
        a,
        cs(
            a,
            AddEntity(label="Place", description="A location."),
            AddRelation(
                label="located_in",
                symmetric=False,
                patterns=[PatternSpec(src="Organization", tgt="Place")],
            ),
            Rename(kind=Kind.ENTITY, target="Organization", new_label="Company"),
            SetParent(target="Employee", parent="Organization"),
            SetIdentityAttributes(target="Person", names=["email", "external_id"]),
            AddAlias(kind=Kind.ENTITY, target="Person", alias="Human"),
            SetDescription(
                kind=Kind.RELATION, target="works_at", description="Employment."
            ),
            SetSymmetric(target="works_at", symmetric=True),
            SetRelationKind(target="works_at", kind=RelationKind.SINGLE),
            AddPattern(
                relation="works_at", pattern=PatternSpec(src="Employee", tgt="Company")
            ),
        ),
    )
    c = apply_changeset(a, diff(a, b))
    assert c.content_hash == b.content_hash
    assert diff(b, b).ops == []


def test_parent_can_change_and_be_cleared():
    """Change a parent and then clear it."""
    a = base()
    b = apply_changeset(a, cs(a, SetParent(target="Employee", parent="Organization")))
    assert b.entity(2).parent_id == 3
    c = apply_changeset(b, cs(b, SetParent(target="Employee", parent=None)))
    assert c.entity(2).parent_id is None


def test_parent_change_refuses_unknown_and_cycles():
    """Reject Unknown and cycles in a parent change."""
    a = base()
    with pytest.raises(InvalidChange, match="Unknown"):
        apply_changeset(a, cs(a, SetParent(target="Employee", parent="Unknown")))
    with pytest.raises(InvalidChange, match="cycle"):
        apply_changeset(a, cs(a, SetParent(target="Person", parent="Employee")))


@pytest.mark.parametrize("names", [["bad-name"], ["Email", "email"]])
def test_identity_attributes_must_be_legal_and_unique(names):
    """Reject invalid identity attribute names."""
    a = base()
    with pytest.raises(InvalidChange):
        apply_changeset(a, cs(a, SetIdentityAttributes(target="Person", names=names)))


_GOLD_EDGES = {
    "LIVES_IN": True,
    "WORKS_AT": True,
    "MANAGES": False,
    "LIKES": False,
    "CONSIDERING_JOINING": False,
    "RELOCATING_TO": False,
}


def _relation_kinds(cat):
    return {r.label: r.kind for r in cat.relations}


def test_relation_kind_registers_diffs_and_round_trips():
    """Replay and serialize changes to a relation kind."""
    g = Ontology.genesis()
    a = apply_changeset(
        g,
        cs(
            g,
            *(
                AddRelation(
                    label=k, kind=RelationKind.SINGLE if v else RelationKind.MANY
                )
                for k, v in _GOLD_EDGES.items()
            ),
        ),
    )
    assert {r.label: r.kind for r in a.relations if r.label != "Unknown"} == {
        k: RelationKind.SINGLE if v else RelationKind.MANY
        for k, v in _GOLD_EDGES.items()
    }
    b = apply_changeset(
        a, cs(a, SetRelationKind(target="LIKES", kind=RelationKind.JUDGED))
    )
    assert _relation_kinds(b)["LIKES"] is RelationKind.JUDGED
    replay = apply_changeset(a, diff(a, b))
    assert replay.content_hash == b.content_hash
    assert (
        _relation_kinds(Ontology.model_validate_json(b.model_dump_json()))["LIKES"]
        is RelationKind.JUDGED
    )


def test_set_relation_kind_refuses_unknown_and_missing_relation():
    """Reject a kind change for an absent relation."""
    a = base()
    with pytest.raises(InvalidChange):
        apply_changeset(
            a, cs(a, SetRelationKind(target="Unknown", kind=RelationKind.SINGLE))
        )
    with pytest.raises(InvalidChange):
        apply_changeset(
            a, cs(a, SetRelationKind(target="nope", kind=RelationKind.SINGLE))
        )


def test_active_entities_cannot_be_reparented_to_retired_types():
    """A new parent must be active after the change set applies."""
    head = base()
    with pytest.raises(InvalidChange, match="parent must be active"):
        apply_changeset(
            head,
            cs(
                head,
                DeprecateType(kind=Kind.ENTITY, target="Organization"),
                SetParent(target="Employee", parent="Organization"),
                DeprecatePattern(
                    relation="works_at",
                    pattern=PatternSpec(src="Person", tgt="Organization"),
                ),
            ),
        )
