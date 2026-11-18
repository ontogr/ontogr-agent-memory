"""Tests for type metadata carried on resolution groups."""

from dataclasses import replace

from ontogr_mem.data_models import ExtractedEntity, Method
from ontogr_mem.resolution.grouping import build_groups
from ontogr_mem.resolution.policy import ResolutionPolicy, TypeRelations


def ex(local_id: str, name: str, label: str) -> ExtractedEntity:
    """Build an extracted entity for grouping tests."""
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=0,
        type_label=label,
        confidence=1,
        method=Method.RECORD,
        episode_id="ep",
    )


class TestGroupTypes:
    """Type triple handling when entities collapse into one group."""

    def test_group_carries_stable_type_id_and_proposed_type(self):
        """A single member's stable id and proposed type reach the group."""
        extracted = replace(
            ex("e1", "Acme", "Unknown"), ontology_type_id=0, proposed_type="Startup"
        )
        group = build_groups([extracted], ResolutionPolicy(), TypeRelations())[0]
        assert group.ontology_type_id == 0
        assert group.proposed_type == "Startup"

    def test_more_specific_member_replaces_type_triple_together(self):
        """A more specific member replaces the whole type triple."""
        less = replace(ex("e1", "Acme", "Object"), ontology_type_id=3, proposed_type="")
        more = replace(
            ex("e2", "Acme", "Organization"), ontology_type_id=9, proposed_type="Org"
        )
        group = build_groups([less, more], ResolutionPolicy(), TypeRelations())[0]
        assert (group.type_label, group.ontology_type_id, group.proposed_type) == (
            "Organization",
            9,
            "Org",
        )

    def test_less_specific_member_keeps_existing_type_triple(self):
        """A less specific member leaves the group's type triple intact."""
        more = replace(
            ex("e1", "Acme", "Organization"), ontology_type_id=9, proposed_type="Org"
        )
        less = replace(ex("e2", "Acme", "Object"), ontology_type_id=3, proposed_type="")
        group = build_groups([more, less], ResolutionPolicy(), TypeRelations())[0]
        assert (group.type_label, group.ontology_type_id, group.proposed_type) == (
            "Organization",
            9,
            "Org",
        )

    def test_later_unknown_mention_supplies_missing_proposed_type(self):
        """Keep a proposed type when the first Unknown mention supplied none."""
        first = ex("e1", "Acme", "Unknown")
        second = replace(ex("e2", "Acme", "Unknown"), proposed_type="Startup")

        [group] = build_groups([first, second], ResolutionPolicy(), TypeRelations())

        assert (group.type_label, group.ontology_type_id, group.proposed_type) == (
            "Unknown",
            0,
            "Startup",
        )
