"""Stamp extracted entities with their ontology type."""

from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.data_models import ExtractedEntity
from ontogr_mem.ontology.projection import Projection


def stamp_types(
    entities: Sequence[ExtractedEntity], projection: Projection
) -> list[ExtractedEntity]:
    """Return the entities with their stable type id and catalog label set.

    The dense id remains available for fact extraction.

    Args:
        entities: Entities with dense ids from the projection.
        projection: The projection that issued the dense ids.

    Raises:
        DenseIdError: If an entity has a dense id outside the projection.
    """
    catalog = projection.view.catalog
    return [
        replace(
            entity,
            ontology_type_id=projection.stable_entity_id(entity.type_id),
            type_label=catalog.entity_label(entity.type_id),
        )
        for entity in entities
    ]
