"""Tests that stored ontology type fields survive resolution round trips."""

from dataclasses import replace

import pytest

from ontogr_mem.data_models import ExtractedEntity
from tests.integration.resolution.resolver_helpers import count_active, ex, make


def typed(
    entity: ExtractedEntity, label: str, type_id: int, proposed: str = ""
) -> ExtractedEntity:
    """Return an entity with an explicit stored type triple."""
    return replace(
        entity,
        type_label=label,
        ontology_type_id=type_id,
        proposed_type=proposed,
    )


class TestStoredEntityTypes:
    """Stored entity type fields across resolution, merge and unmerge."""

    @pytest.mark.asyncio
    async def test_matching_entity_keeps_stored_type_fields(self, graph_backend):
        """A matching mention preserves the stored type triple."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        original = typed(ex("e1", "Acme"), "Organization", 17, "company")
        await resolver.resolve([original])

        result = await resolver.resolve(
            [typed(ex("e2", "Acme", episode_id="ep2"), "Organization", 17, "business")]
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            17,
            "Organization",
            "company",
        )

    @pytest.mark.asyncio
    async def test_specific_mention_types_stored_object(self, graph_backend):
        """A specific mention overwrites a stored generic type."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([typed(ex("e1", "Acme"), "Object", 1, "thing")])

        result = await resolver.resolve(
            [typed(ex("e2", "Acme", episode_id="ep2"), "Organization", 17, "company")]
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            17,
            "Organization",
            "company",
        )

    @pytest.mark.asyncio
    async def test_unknown_mention_keeps_stored_specific_type(self, graph_backend):
        """An Unknown mention does not overwrite a stored specific type."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([typed(ex("e1", "Acme"), "Organization", 17, "company")])

        result = await resolver.resolve(
            [typed(ex("e2", "Acme", episode_id="ep2"), "Unknown", 0, "business")]
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            17,
            "Organization",
            "company",
        )

    @pytest.mark.asyncio
    async def test_same_name_with_distinct_types_stays_separate(self, graph_backend):
        """Same-name mentions with distinct types remain separate entities."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)

        await resolver.resolve(
            [
                typed(ex("e1", "Acme"), "Organization", 17),
                typed(ex("e2", "Acme"), "Preference", 29),
            ]
        )

        assert await count_active(resolver_store) == 2

    @pytest.mark.asyncio
    async def test_unmerge_restores_survivor_type_fields(self, graph_backend):
        """Unmerging restores the survivor's original type fields."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        org = typed(ex("e1", "Acme"), "Organization", 17, "company")
        unknown = typed(
            ex("e2", "Acme Holdings", episode_id="ep2"), "Unknown", 0, "business"
        )
        await resolver.resolve([org])
        await resolver.resolve([unknown])
        entities = await resolver_store.active_entities()
        organization = next(
            item for item in entities if item.type_label == "Organization"
        )
        unknown_entity = next(item for item in entities if item.type_label == "Unknown")
        merge = await resolver.merge(
            organization.entity_id, unknown_entity.entity_id, actor="user"
        )
        survivor = (await resolver_store.get_entities([merge.survivor_id]))[0]
        assert survivor.type_label == "Organization"
        assert survivor.ontology_type_id == 17
        assert survivor.proposed_type == "company"

        await resolver.unmerge(merge.merge_id)
        restored = (await resolver_store.get_entities([merge.survivor_id]))[0]
        assert (
            restored.ontology_type_id,
            restored.type_label,
            restored.proposed_type,
        ) == (
            17,
            "Organization",
            "company",
        )

    @pytest.mark.asyncio
    async def test_default_policy_merges_person_without_model_call(self, graph_backend):
        """Default policy merges matching Person mentions without a model call."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        person = typed(ex("e1", "Ada"), "Person", 3)
        await resolver.resolve([person])

        result = await resolver.resolve(
            [typed(ex("e2", "Ada", episode_id="ep2"), "Person", 3)]
        )

        assert len(result.entities) == 1
        assert await count_active(resolver_store) == 1
