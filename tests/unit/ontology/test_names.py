"""Tests for ontology name normalization."""

from ontogr_mem.data_models.text import normalize_label
from ontogr_mem.ontology.names import normalize


def test_ontology_label_normalization_uses_the_shared_label_key():
    """Ontology names use the shared case, whitespace and underscore rules."""
    assert normalize("  Data_Model ") == normalize_label("Data Model") == "datamodel"
