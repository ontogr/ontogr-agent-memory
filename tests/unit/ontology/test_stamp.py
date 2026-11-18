"""Tests for stamping extracted entities with ontology types."""

import pytest

from ontogr_mem.data_models import ExtractedEntity, Method
from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.errors import DenseIdError
from ontogr_mem.ontology.models import DeprecateType, Kind, Ontology
from ontogr_mem.ontology.projection import project
from ontogr_mem.ontology.stamp import stamp_types
from tests.support.ontology import base, cs


class TestStampTypes:
    """Resolve dense ids against the catalog that issued them."""

    def test_projects_the_stable_id_across_a_deprecated_type(self):
        """A deprecated source type still stamps the surviving stable id."""
        catalog = base()
        catalog = apply_changeset(
            catalog, cs(catalog, DeprecateType(kind=Kind.ENTITY, target="Employee"))
        )
        entity = ExtractedEntity(
            "e0", "Acme", 1, 0.9, Method.LLM, "episode", description="An employer"
        )

        [stamped] = stamp_types([entity], project(catalog))

        assert stamped == ExtractedEntity(
            "e0",
            "Acme",
            1,
            0.9,
            Method.LLM,
            "episode",
            type_label="Organization",
            description="An employer",
            ontology_type_id=3,
        )

    def test_unknown_keeps_the_proposed_type(self):
        """An unmapped dense id keeps the proposed type and reports Unknown."""
        catalog = Ontology.genesis()
        entity = ExtractedEntity(
            "e0", "Atlas", -1, 0.9, Method.LLM, "episode", proposed_type="Robot"
        )

        [stamped] = stamp_types([entity], project(catalog))

        assert stamped.ontology_type_id == 0
        assert stamped.type_label == "Unknown"
        assert stamped.proposed_type == "Robot"

    def test_rejects_an_id_outside_the_projection(self):
        """A dense id absent from the projection is rejected."""
        entity = ExtractedEntity("e0", "Acme", 5, 0.9, Method.LLM, "episode")

        with pytest.raises(DenseIdError):
            stamp_types([entity], project(base()))
