"""Tests for the type catalog."""

import pytest

from ontogr_mem.data_models import (
    UNKNOWN_TYPE_ID,
    EntityType,
    RelationType,
    TypeCatalog,
    TypeMode,
    TypeView,
)


def make_catalog() -> TypeCatalog:
    """Make catalog."""
    return TypeCatalog(
        [EntityType(0, "Person"), EntityType(1, "Org")],
        [RelationType(0, "WORKS_AT", source_types=frozenset({0}))],
    )


class TestTypeCatalog:
    """Behavior of TypeCatalog and TypeView."""

    def test_rejects_ids_that_are_not_dense(self):
        """Rejects ids that are not dense."""
        with pytest.raises(ValueError, match="0..n-1"):
            TypeCatalog([EntityType(1, "Person")], [])

    def test_unknown_id_is_minus_one(self):
        """Unknown id is minus one."""
        catalog = make_catalog()

        assert UNKNOWN_TYPE_ID == -1
        assert catalog.has_entity_id(-1)
        assert not catalog.has_entity_id(2)
        assert catalog.entity_label(-1) == "Unknown"

    def test_version_is_a_stable_sixteen_character_hash(self):
        """Version is a stable sixteen character hash."""
        assert make_catalog().version == "b0afa31bcb938538"

    def test_direction_is_checked_unless_a_side_is_unknown(self):
        """Direction is checked unless a side is unknown."""
        catalog = make_catalog()
        relation = catalog.relation_types[0]

        assert catalog.direction_ok(relation, 0, 1)
        assert not catalog.direction_ok(relation, 1, 0)
        assert catalog.direction_ok(relation, UNKNOWN_TYPE_ID, 0)

    def test_view_version_joins_catalog_version_and_mode(self):
        """View version joins catalog version and mode."""
        catalog = make_catalog()

        view = TypeView(catalog, TypeMode.STRICT)

        assert view.version == "b0afa31bcb938538:strict"

    def test_entity_and_relation_lookups_ignore_label_separators(self):
        """Lookups use canonical label normalization for both type kinds."""
        catalog = TypeCatalog(
            [EntityType(0, "Data Model")],
            [RelationType(0, "USES_TOOL")],
        )

        assert catalog.entity_by_label("data_model") == EntityType(0, "Data Model")
        assert catalog.relation_by_label(" uses tool ") == RelationType(0, "USES_TOOL")
