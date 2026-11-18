"""Tests for type metadata carried on resolution groups."""

from dataclasses import replace

import pytest

from ontogr_mem.data_models import ExtractedEntity, Method
from ontogr_mem.resolution.grouping import Group, build_groups
from tests.support.type_world import WORLD


def ex(
    local_id: str,
    name: str,
    label: str,
    *,
    attributes: dict[str, str] | None = None,
    proposed_type: str = "",
) -> ExtractedEntity:
    """Build an ontology-stamped extracted entity for grouping tests."""
    return replace(
        ExtractedEntity(
            local_id=local_id,
            name=name,
            type_id=0,
            type_label=label,
            confidence=1,
            method=Method.RECORD,
            episode_id="ep",
        ),
        ontology_type_id=WORLD.ids[label],
        proposed_type=proposed_type,
        attributes=attributes or {},
    )


def groups(entities: list[ExtractedEntity]) -> list[Group]:
    """Group entities against the shared ontology world."""
    return build_groups(entities, WORLD.links)


class TestGroupTypes:
    """Type triple handling when entities collapse into one group."""

    def test_group_carries_canonical_id_and_label_of_first_member(self):
        """A group uses the canonical id and label of its first member."""
        [group] = groups([ex("e1", "Acme", "Company")])
        assert group.ontology_type_id == WORLD.ids["Company"]
        assert group.type_label == "Company"

    def test_child_member_replaces_group_id_and_label(self):
        """A later child type makes the group take the child's id and label."""
        parent = ex("e1", "Deal", "Contract")
        child = ex("e2", "Deal", "PurchaseOrder")
        [group] = groups([parent, child])
        assert group.ontology_type_id == WORLD.ids["PurchaseOrder"]
        assert group.type_label == "PurchaseOrder"

    def test_unrelated_sibling_stays_in_its_own_group(self):
        """Two same-name mentions of sibling types form two groups."""
        company = ex("e1", "Acme", "Company")
        city = ex("e2", "Acme", "City")
        result = groups([company, city])
        assert len(result) == 2
        assert {group.type_label for group in result} == {"Company", "City"}

    def test_unknown_member_joins_a_typed_group(self):
        """An Unknown mention joins a typed group and keeps its type."""
        typed = ex("e1", "Acme", "Company")
        unknown = ex("e2", "Acme", "Unknown")
        [group] = groups([typed, unknown])
        assert group.ontology_type_id == WORLD.ids["Company"]
        assert group.type_label == "Company"

    def test_typed_member_promotes_unknown_group_and_clears_proposed_type(self):
        """A typed member gives an Unknown group its id and no proposed type."""
        unknown = ex("e1", "Acme", "Unknown", proposed_type="Startup")
        typed = ex("e2", "Acme", "Company")
        [group] = groups([unknown, typed])
        assert group.ontology_type_id == WORLD.ids["Company"]
        assert group.type_label == "Company"
        assert group.proposed_type == ""

    def test_entity_without_stamped_type_is_refused(self):
        """A mention without an ontology type id cannot be grouped."""
        extracted = replace(ex("e1", "Acme", "Company"), ontology_type_id=None)
        with pytest.raises(ValueError, match="ontology type"):
            groups([extracted])

    def test_different_identity_values_form_two_groups(self):
        """Two people with one name and different emails form two groups."""
        first = ex("e1", "Ana", "Person", attributes={"email": "ana@one.org"})
        second = ex("e2", "Ana", "Person", attributes={"email": "ana@two.org"})
        result = groups([first, second])
        assert len(result) == 2
        assert [len(group.members) for group in result] == [1, 1]

    def test_mention_without_identity_joins_group_with_one(self):
        """A mention with no email joins the first group that has one."""
        with_email = ex("e1", "Ana", "Person", attributes={"email": "ana@one.org"})
        without = ex("e2", "Ana", "Person")
        [group] = groups([with_email, without])
        assert len(group.members) == 2
        assert group.identity == {"email": "ana@one.org"}
