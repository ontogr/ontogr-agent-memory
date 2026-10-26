"""Tests for the public ontology API."""

from ontogr_mem.ontology import TableMapping, UnknownEdges


class TestOntologyPublicAPI:
    """Caller-facing ontology types are available from the package."""

    def test_exports_mapping_and_unknown_edge_types(self):
        """The schema model's public types can be imported from ontology."""
        mapping = TableMapping(source="hr.people", entity_id=1, columns={})

        assert mapping.source == "hr.people"
        assert UnknownEdges.KEEP.value == "keep"
