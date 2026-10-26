"""Tests for model reply parsing."""

import pytest

from ontogr_mem.entity_extraction.parsing import (
    SchemaError,
    parse_entities,
)


class TestParseEntities:
    """Behavior of parse_entities."""

    def test_reads_type_label_and_optional_fields(self):
        """A complete item is read."""
        (entity,) = parse_entities(
            '{"entities": [{"name": "Zorbit", "type_id": -1, '
            '"type_label": "Gadget", "confidence": 0.9}]}'
        )

        assert (entity.name, entity.type_id, entity.type_label, entity.confidence) == (
            "Zorbit",
            -1,
            "Gadget",
            0.9,
        )

    def test_reports_every_problem_at_once(self):
        """Errors accumulate so one repair round can fix them all."""
        reply = (
            '{"entities": [{"name": 5, "type_id": true},'
            ' {"name": "x", "type_id": 1, "confidence": 2}]}'
        )

        with pytest.raises(SchemaError) as excinfo:
            parse_entities(reply)

        assert excinfo.value.errors == [
            "entities[0].type_id must be an integer",
            "entities[0].name must be a string",
            "entities[1].confidence must be a number in 0..1 or null",
        ]

    def test_missing_entities_key_is_an_error(self):
        """The reply must have the entities array."""
        with pytest.raises(SchemaError, match="entities"):
            parse_entities("{}")
