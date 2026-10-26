"""Tests for schema part declarations."""

from ontogr_mem.graphstore.indices import (
    FulltextIndex,
    IndexSpec,
    RangeIndex,
    SchemaPart,
    UniqueConstraint,
    merge_parts,
    plan,
)
from ontogr_mem.graphstore.schemas import ALL, ONTOLOGY, TEMPORAL


class TestSchemaPart:
    """Behavior of SchemaPart and merge_parts."""

    def test_fulltext_properties_merge_once_with_dependencies(self):
        """A multi-property fulltext index remains one requirement."""
        index = FulltextIndex("Observation", ("statement", "quote"))
        base = SchemaPart("temporal", labels=("Observation",))
        first = SchemaPart("keyword", fulltexts=(index,), requires=("temporal",))
        second = SchemaPart("other", fulltexts=(index,))

        merged = merge_parts([first, second], {"temporal": base})

        assert merged.fulltexts == (index,)
        assert plan(merged) == (
            IndexSpec("unique", "Observation", "id"),
            IndexSpec("fulltext", "Observation", "", props=("statement", "quote")),
        )

    def test_every_label_gets_an_id_constraint(self):
        """Declaring a label implies a uniqueness constraint on its id."""
        part = SchemaPart("a", labels=("X",), ranges=(RangeIndex("X", "name"),))

        assert plan(part) == (
            IndexSpec("range", "X", "name"),
            IndexSpec("unique", "X", "id"),
        )

    def test_a_unique_constraint_needs_no_range_spec(self):
        """A unique constraint implies an index on the same property."""
        part = SchemaPart("a", labels=("X",), uniques=(UniqueConstraint("X", "email"),))

        assert plan(part) == (
            IndexSpec("unique", "X", "id"),
            IndexSpec("unique", "X", "email"),
        )

    def test_merge_drops_duplicates_and_joins_edge_keys(self):
        """Parts that share a label produce one declaration."""
        first = SchemaPart("a", labels=("X", "Y"), edge_keys={"R": ("k",)})
        second = SchemaPart("b", labels=("Y", "Z"), edge_keys={"S": ()})

        merged = merge_parts([first, second])

        assert merged.labels == ("X", "Y", "Z")
        assert merged.edge_keys == {"R": ("k",), "S": ()}

    def test_component_schema_registry_resolves_declared_dependencies(self):
        """The ontology declaration includes the episode schema it needs."""
        registry = {part.name: part for part in ALL}

        merged = merge_parts([ONTOLOGY], registry)

        assert merged.name == "episodes+ontology"
        assert merged.labels == (
            "Episode",
            "Source",
            "Actor",
            "EpisodeMeta",
            "OntologyHead",
            "OntologySnapshot",
            "OntologyChangeSet",
            "OntologyUnmatched",
            "OntologyProposal",
        )

    def test_temporal_schema_includes_its_episode_and_entity_dependencies(self):
        """Opening temporal storage alone provisions its referenced node types."""
        registry = {part.name: part for part in ALL}

        merged = merge_parts([TEMPORAL], registry)

        assert merged.name == "episodes+resolution+temporal"
        assert {"Episode", "Entity", "Slot"} <= set(merged.labels)
