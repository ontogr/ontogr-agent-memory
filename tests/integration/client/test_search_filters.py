"""End-to-end checks for type and modality search filters."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    RelationDeclaration,
)
from ontogr_mem.service.errors import InvalidArgumentError
from tests.support.falkordb import falkordb_test_settings
from tests.support.tenants import new_tenant
from tests.support.text_writes import (
    ScriptedEntity,
    ScriptedFact,
    open_memory,
    text_models,
)


@pytest.fixture(params=["two_pass", "combined"], autouse=True)
def extraction_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Run each case with both text extraction modes."""
    monkeypatch.setenv("MEMORY_EXTRACTION_MODE", request.param)
    return str(request.param)


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point Memory settings at the live FalkorDB test service."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


class TestSearchFilters:
    """Exercise search filters through scripted writes and FalkorDB."""

    async def test_entity_and_relation_filters_select_facts_and_keep_unplaced(
        self, falkordb_store: FalkorDBGraphStore
    ) -> None:
        """Select endpoint and relation types while retaining unplaced facts."""
        tenant = new_tenant()
        script = {
            "Ana works at Acme.": (
                [
                    ScriptedEntity("Ana", "Person"),
                    ScriptedEntity("Acme", "Organization"),
                ],
                [
                    ScriptedFact(
                        "Ana works at Acme", 0, 1, "WORKS_AT", "Ana works at Acme"
                    )
                ],
            ),
            "Ben lives in Berlin.": (
                [ScriptedEntity("Ben", "Person"), ScriptedEntity("Berlin", "Location")],
                [
                    ScriptedFact(
                        "Ben lives in Berlin", 0, 1, "LIVES_IN", "Ben lives in Berlin"
                    )
                ],
            ),
            "Ana mentors Ben.": (
                [ScriptedEntity("Ana", "Person"), ScriptedEntity("Ben", "Person")],
                [ScriptedFact("Ana mentors Ben", 0, 1, "", "Ana mentors Ben")],
            ),
        }
        ontology = OntologyDeclaration(
            entities=[
                EntityDeclaration(label=name)
                for name in ("Person", "Organization", "Location")
            ],
            relations=[
                RelationDeclaration(label=name) for name in ("WORKS_AT", "LIVES_IN")
            ],
        )
        try:
            async with open_memory(
                tenant, text_models(script), ontology=ontology
            ) as memory:
                for text in script:
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                by_entity = await memory.search(
                    "Ana Ben Berlin", entity_types=["Location"]
                )
                assert {fact.text for fact in by_entity.facts} == {
                    "Ben lives in Berlin"
                }
                included = await memory.search("Ana Ben", relation_types=["LIVES_IN"])
                assert {fact.text for fact in included.facts} == {"Ben lives in Berlin"}
                excluded = await memory.search(
                    "Ana Ben", exclude_relation_types=["LIVES_IN"]
                )
                assert "Ana mentors Ben" in {fact.text for fact in excluded.facts}
                assert "Ben lives in Berlin" not in {
                    fact.text for fact in excluded.facts
                }
                no_org = await memory.search(
                    "Ana Acme", exclude_entity_types=["Organization"]
                )
                assert "Ana works at Acme" not in {fact.text for fact in no_org.facts}
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_modalities_are_opt_in_and_invalid_labels_are_refused(
        self, falkordb_store: FalkorDBGraphStore
    ) -> None:
        """Return planned facts only on request and reject unknown type labels."""
        tenant = new_tenant()
        text = "Ana will join Acme next year."
        script = {
            text: (
                [
                    ScriptedEntity("Ana", "Person"),
                    ScriptedEntity("Acme", "Organization"),
                ],
                [
                    ScriptedFact(
                        "Ana joins Acme", 0, 1, "WORKS_AT", text, modality="planned"
                    )
                ],
            )
        }
        ontology = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Person"),
                EntityDeclaration(label="Organization"),
            ],
            relations=[RelationDeclaration(label="WORKS_AT")],
        )
        try:
            async with open_memory(
                tenant, text_models(script), ontology=ontology
            ) as memory:
                await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                assert not (await memory.search("Ana Acme")).facts
                planned = await memory.search("Ana Acme", modalities=["planned"])
                assert {fact.text for fact in planned.facts} == {"Ana joins Acme"}
                with pytest.raises(InvalidArgumentError, match="MissingType"):
                    await memory.search("Ana", entity_types=["MissingType"])
        finally:
            await falkordb_store.erase_tenant(tenant)
