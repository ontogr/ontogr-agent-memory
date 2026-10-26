"""Tests for lookup keys, resolution policy and fuzzy blocking."""

import pytest

from ontogr_mem.resolution.blocking import (
    BlockIndex,
    BlockingRow,
    has_high_entropy,
    jaccard,
    shingles,
)
from ontogr_mem.resolution.keys import (
    differ_only_in_digits,
    loose_key,
    strict_key,
    stripped_key,
)
from ontogr_mem.resolution.policy import (
    ResolutionMode,
    ResolutionPolicy,
    TypePolicy,
    TypeRel,
    TypeRelations,
)


def test_strict_key_folds_case_and_space_but_keeps_punctuation_and_marks():
    """The strict key keeps diacritics and punctuation."""
    assert strict_key("  Café   NYC ") == "café nyc"
    assert strict_key("U.S.A.") != strict_key("USA")


def test_loose_key_drops_marks_and_punctuation_but_never_goes_empty():
    """The loose key is the blocking form and never becomes empty."""
    assert loose_key("Café, N.Y.C.") == "cafe n y c"
    assert loose_key("東京") == "東京"
    assert loose_key("???") == "???"


def test_loose_key_collides_c_plus_plus_with_c_which_is_why_it_never_auto_merges():
    """A loose-key collision is why it may only propose candidates."""
    assert loose_key("C++") == loose_key("C")
    assert strict_key("C++") != strict_key("C")


def test_stripped_key_drops_one_article_and_one_legal_suffix():
    """Only a leading article and one trailing suffix are removed."""
    assert stripped_key("The Apple Inc.") == "apple"
    assert stripped_key("Apple Holdings Inc") == "apple holdings"
    assert stripped_key("Inc") == "inc"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Q3 2023", "Q3 2024", True),
        ("Q3 2023", "q3 2023", False),
        ("Apple", "Apples", False),
    ],
)
def test_digit_only_difference(a, b, expected):
    """Names that differ only in digit runs are a hard veto."""
    assert differ_only_in_digits(a, b) is expected


def test_entropy_floor_blocks_short_single_token_names():
    """Short and repetitive names never take part in fuzzy matching."""
    assert not has_high_entropy("Joe")
    assert not has_high_entropy("aaaaaaa")
    assert has_high_entropy("Joe Michaels")


def test_jaccard_hand_computed_value_for_a_near_name():
    """The Jaccard value of a near name is the hand-computed one."""
    a, b = shingles("joe michaels"), shingles("joe michael")
    assert round(jaccard(a, b), 3) == 0.889


def test_block_index_finds_a_variant_and_is_sorted():
    """Candidates come back sorted."""
    ix = BlockIndex()
    ix.add("b", "joe michaels")
    ix.add("a", "joe michaels")
    ix.add("c", "completely different")
    assert ix.candidates("joe michaels") == ["a", "b"]


def test_block_index_from_rows_indexes_aliases_and_the_stripped_key():
    """A row adds the canonical key, one item per alias and the stripped key."""
    ix = BlockIndex.from_rows(
        [
            BlockingRow(
                entity_id="e1",
                loose_key="apple inc",
                stripped_key="apple",
                alias_keys=("zebra quux",),
            )
        ]
    )

    assert ix.candidates("zebra quux") == ["e1#0"]
    assert "e1#s" in ix.candidates("apple")


def test_policy_defaults_person_verify_always_others_auto_key():
    """People are verified and every other type may merge on a key."""
    p = ResolutionPolicy()
    assert p.mode("Person") is ResolutionMode.VERIFY_ALWAYS
    assert p.mode("City") is ResolutionMode.AUTO_KEY
    assert p.mode("Widget") is ResolutionMode.AUTO_KEY


def test_policy_override_beats_family_default():
    """A per-type setting wins over the family default."""
    p = ResolutionPolicy(types={"Person": TypePolicy(mode=ResolutionMode.AUTO_KEY)})
    assert p.mode("person") is ResolutionMode.AUTO_KEY


def test_type_relations_keep_distinct_and_family_apart():
    """Type relation, case folding, nesting and specificity agree."""
    p = ResolutionPolicy()
    rel = TypeRelations({"Employee": "Person"})
    assert p.relation("Java", "City") is TypeRel.DISTINCT
    assert p.relation("Company", "Organization") is TypeRel.FAMILY
    assert p.relation("Company", "company") is TypeRel.SAME
    assert p.relation("Entity", "City") is TypeRel.NESTED
    assert p.relation("Employee", "Person", rel) is TypeRel.NESTED
    assert rel.more_specific("Person", "Employee") == "Employee"


def test_type_relations_survive_a_parent_cycle():
    """A parent cycle ends the ancestor walk."""
    rel = TypeRelations({"A": "B", "B": "A"})
    assert rel.ancestors("A") == ["b", "a"]


def test_policy_and_ancestry_ignore_label_separators():
    """Policy lookup and caller-supplied ancestry share label normalization."""
    policy = ResolutionPolicy(
        types={
            "Data_Model": TypePolicy(
                mode=ResolutionMode.VERIFY_ALWAYS,
                identity_attrs=("email",),
                strip_suffixes=True,
            )
        },
        families={"Data Model": "domain"},
    )
    relations = TypeRelations({"Staff Member": "People"})

    assert policy.mode("data model") is ResolutionMode.VERIFY_ALWAYS
    assert policy.family("data_model") == "domain"
    assert policy.identity_attrs("data model") == ("email",)
    assert policy.strip_suffixes("data_model")
    assert relations.ancestors("staff_member") == ["people"]
    assert policy.relation("Staff_Member", "People", relations) is TypeRel.NESTED
    assert relations.more_specific("People", "Staff_Member") == "Staff_Member"


def test_unknown_label_comparison_uses_canonical_label_normalization():
    """Unknown remains vague when written with label separators."""
    assert ResolutionPolicy().relation("U_nknown", "City") is TypeRel.NESTED
