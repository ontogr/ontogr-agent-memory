"""Client integration tests for entity listings and ontology reports."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.data_models.episode import Kind
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    RelationDeclaration,
)
from tests.support.falkordb import falkordb_test_settings
from tests.support.tenants import new_tenant
from tests.support.text_writes import (
    ScriptedEntity,
    ScriptedFact,
    open_memory,
    text_models,
)


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point Memory settings at the live FalkorDB test service."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


class TestClientReadCatalogs:
    """Exercise paged client reads backed by FalkorDB."""

    async def test_list_entities_filters_type_and_pages_in_id_order(
        self, falkordb_store: FalkorDBGraphStore
    ) -> None:
        """List only the requested active type and resume after its cursor."""
        tenant = new_tenant()
        ontology = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Person"),
                EntityDeclaration(label="Place"),
            ],
            relations=[RelationDeclaration(label="KNOWS")],
        )
        script = {
            "Ana knows Bo in Oslo.": (
                [
                    ScriptedEntity("Ana", "Person"),
                    ScriptedEntity("Bo", "Person"),
                    ScriptedEntity("Oslo", "Place"),
                ],
                [
                    ScriptedFact("Ana knows Bo", 0, 1, "KNOWS", "Ana knows Bo"),
                ],
            )
        }
        try:
            async with open_memory(
                tenant, text_models(script), ontology=ontology
            ) as memory:
                receipt = await memory.write_text(
                    "Ana knows Bo in Oslo.", wait_seconds=10
                )
                await memory.drain()
                first = await memory.list_entities(entity_type="Person", limit=1)
                assert len(first.entities) == 1
                assert first.next_cursor is not None
                second = await memory.list_entities(
                    entity_type="Person", cursor=first.next_cursor, limit=1
                )
                assert len(second.entities) == 1
                assert second.next_cursor is None
                assert first.entities[0].entity_id < second.entities[0].entity_id
                assert {
                    first.entities[0].type_label,
                    second.entities[0].type_label,
                } == {"Person"}
                assert receipt.job_id
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_get_ontology_returns_filter_vocabulary_and_optional_deprecated_types(
        self, falkordb_store: FalkorDBGraphStore
    ) -> None:
        """Expose active vocabulary by default and deprecated labels on request."""
        tenant = new_tenant()
        ontology = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Person", description="A person"),
                EntityDeclaration(label="LegacyPerson", deprecated=True),
            ],
            relations=[
                RelationDeclaration(label="KNOWS", aliases=["is acquainted with"])
            ],
        )
        try:
            async with open_memory(
                tenant, text_models({}), ontology=ontology
            ) as memory:
                active = await memory.get_ontology()
                complete = await memory.get_ontology(include_deprecated=True)
                assert active.tenant == tenant
                assert {item.label for item in active.entity_types} == {"Person"}
                assert {item.label for item in active.relation_types} == {"KNOWS"}
                assert "is acquainted with" in next(
                    item.aliases
                    for item in complete.relation_types
                    if item.label == "KNOWS"
                )
                assert {item.label for item in complete.entity_types} == {
                    "Person",
                    "LegacyPerson",
                }
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_stale_episodes_reports_old_extraction_after_reopening(
        self, falkordb_store: FalkorDBGraphStore
    ) -> None:
        """Report a stored text episode after a later ontology version is opened."""
        tenant = new_tenant()
        original = OntologyDeclaration(entities=[EntityDeclaration(label="Person")])
        updated = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Person"),
                EntityDeclaration(label="Place"),
            ]
        )
        script = {
            "Ana lives in Oslo.": (
                [ScriptedEntity("Ana", "Person"), ScriptedEntity("Oslo", "Person")],
                [],
            )
        }
        try:
            async with open_memory(
                tenant, text_models(script), ontology=original
            ) as memory:
                receipt = await memory.write_text("Ana lives in Oslo.", wait_seconds=10)
                await memory.drain()
                old_version = (await memory.get_ontology()).version
                assert receipt.job_id
            async with open_memory(tenant, text_models({}), ontology=updated) as memory:
                page = await memory.stale_episodes()
                assert page.current_version > old_version
                assert len(page.episodes) == 1
                assert page.episodes[0].kind == Kind.DOCUMENT
                assert page.episodes[0].ontology_version == old_version
                assert page.episodes[0].episode_id
                assert page.episodes[0].ingested_at
        finally:
            await falkordb_store.erase_tenant(tenant)
