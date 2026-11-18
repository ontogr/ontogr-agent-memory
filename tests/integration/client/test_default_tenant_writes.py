"""End-to-end checks of text writes on a fresh tenant with the default ontology."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest

from ontogr_mem.client import Memory
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.service.contracts import EntityOut, HostFact
from tests.support.falkordb import falkordb_test_settings
from tests.support.tenants import new_tenant
from tests.support.text_writes import (
    ScriptedEntity,
    ScriptedFact,
    counted_text_models,
    open_memory,
    text_models,
)


FIRST = "Ana works at Acme."
SECOND = "Ana works at Beta."
UNKNOWN_RELATION = "Ana mentors Ben."
MODEL_LABEL = "Ana works at Zeta."
THIRD = "Ana still works at Acme."
HYDRA_OBJECT = "Ana owns Hydra."
HYDRA_ORG = "Ana works at Hydra."
ACME_PREFERENCE = "Ana likes Acme."
STUDIED = "Ana studied at MIT."
ANA_ONE = "Ana (ana@one.org) works at Acme."
ANA_TWO = "Ana (ana@two.org) works at Beta."
ANA_NONE = "Ana works at Cyd."
ANA_ONE_AGAIN = "Ana (Ana@One.org) works at Dune."
ANA_UNSTATED = "Ana works at Elm."
ANA_NONE_AGAIN = "Ana works at Fig."
ZETA_ORG = "Ana works at Zeta."
EARLIER = datetime(2025, 1, 1, tzinfo=UTC)
LATER = datetime(2025, 6, 1, tzinfo=UTC)
SCRIPT = {
    FIRST: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Acme", "Organization")],
        [ScriptedFact("Ana works at Acme", 0, 1, "WORKS_AT", "Ana works at Acme")],
    ),
    SECOND: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Beta", "Organization")],
        [ScriptedFact("Ana works at Beta", 0, 1, "WORKS_AT", "Ana works at Beta")],
    ),
    UNKNOWN_RELATION: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Ben", "Person")],
        [ScriptedFact("Ana mentors Ben", 0, 1, "mentors", "Ana mentors Ben")],
    ),
    MODEL_LABEL: (
        [
            ScriptedEntity("Ana", "Person"),
            ScriptedEntity("Zeta", None, proposed_label="Startup"),
        ],
        [ScriptedFact("Ana works at Zeta", 0, 1, "WORKS_AT", "Ana works at Zeta")],
    ),
    THIRD: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Acme", "Organization")],
        [
            ScriptedFact(
                "Ana works at Acme", 0, 1, "WORKS_AT", "Ana still works at Acme"
            )
        ],
    ),
    HYDRA_OBJECT: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Hydra", "Object")],
        [ScriptedFact("Ana owns Hydra", 0, 1, "OWNS", "Ana owns Hydra")],
    ),
    HYDRA_ORG: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Hydra", "Organization")],
        [ScriptedFact("Ana works at Hydra", 0, 1, "WORKS_AT", "Ana works at Hydra")],
    ),
    ACME_PREFERENCE: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Acme", "Preference")],
        [ScriptedFact("Ana prefers Acme", 0, 1, "PREFERS", "Ana likes Acme")],
    ),
    STUDIED: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("MIT", "Organization")],
        [ScriptedFact("Ana studied at MIT", 0, 1, "STUDENT_OF", "Ana studied at MIT")],
    ),
    ANA_ONE: (
        [
            ScriptedEntity("Ana", "Person", attributes={"email": "ana@one.org"}),
            ScriptedEntity("Acme", "Organization"),
        ],
        [
            ScriptedFact(
                "Ana works at Acme", 0, 1, "WORKS_AT", "Ana (ana@one.org) works at Acme"
            )
        ],
    ),
    ANA_TWO: (
        [
            ScriptedEntity("Ana", "Person", attributes={"email": "ana@two.org"}),
            ScriptedEntity("Beta", "Organization"),
        ],
        [
            ScriptedFact(
                "Ana works at Beta", 0, 1, "WORKS_AT", "Ana (ana@two.org) works at Beta"
            )
        ],
    ),
    ANA_NONE: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Cyd", "Organization")],
        [ScriptedFact("Ana works at Cyd", 0, 1, "WORKS_AT", "Ana works at Cyd")],
    ),
    ANA_ONE_AGAIN: (
        [
            ScriptedEntity("Ana", "Person", attributes={"email": "Ana@One.org"}),
            ScriptedEntity("Dune", "Organization"),
        ],
        [
            ScriptedFact(
                "Ana works at Dune", 0, 1, "WORKS_AT", "Ana (Ana@One.org) works at Dune"
            )
        ],
    ),
    ANA_UNSTATED: (
        [
            ScriptedEntity("Ana", "Person", attributes={"email": "ana@three.org"}),
            ScriptedEntity("Elm", "Organization"),
        ],
        [ScriptedFact("Ana works at Elm", 0, 1, "WORKS_AT", "Ana works at Elm")],
    ),
    ANA_NONE_AGAIN: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Fig", "Organization")],
        [ScriptedFact("Ana works at Fig", 0, 1, "WORKS_AT", "Ana works at Fig")],
    ),
    ZETA_ORG: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Zeta", "Organization")],
        [ScriptedFact("Ana works at Zeta", 0, 1, "WORKS_AT", "Ana works at Zeta")],
    ),
}


async def _entities_named(memory: Memory, name: str) -> list[EntityOut]:
    """Return the entities named ``name`` that a search returns, sorted by type."""
    context = await memory.search(name)
    return sorted(
        (entity for entity in context.entities if entity.name == name),
        key=lambda entity: entity.type_label,
    )


@pytest.fixture(params=["two_pass", "combined"], autouse=True)
def extraction_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Select the extraction structure for each test through the environment."""
    monkeypatch.setenv("MEMORY_EXTRACTION_MODE", request.param)
    return str(request.param)


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point the service settings at the compose FalkorDB for each test."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


class TestDefaultTenantWrites:
    """Text writes on a fresh tenant behave as the default ontology defines."""

    async def test_a_text_write_makes_the_expected_number_of_model_requests(
        self, falkordb_store: FalkorDBGraphStore, extraction_mode: str
    ):
        """Each extraction mode makes its expected number of model requests."""
        tenant = new_tenant()
        try:
            models, scripted = counted_text_models(SCRIPT)
            async with open_memory(tenant, models) as memory:
                await memory.write_text(FIRST, wait_seconds=10)
                await memory.drain()
            assert len(scripted.calls) == (1 if extraction_mode == "combined" else 2)
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_two_incompatible_types_with_one_name_stay_two_entities(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Incompatible catalog types keep separate entities with the same name."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, wait_seconds=10)
                await memory.write_text(ACME_PREFERENCE, wait_seconds=10)
                await memory.drain()
                entities = await _entities_named(memory, "Acme")
                assert [entity.type_label for entity in entities] == [
                    "Organization",
                    "Preference",
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_person_seen_in_two_episodes_is_one_entity(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A person mentioned in two episodes resolves to one entity."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, wait_seconds=10)
                await memory.write_text(THIRD, wait_seconds=10)
                await memory.drain()
                entities = await _entities_named(memory, "Ana")
                assert [entity.type_label for entity in entities] == ["Person"]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_an_object_entity_and_an_organization_with_one_name_stay_two_entities(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Object and Organization remain separate types with the same name."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(HYDRA_OBJECT, wait_seconds=10)
                await memory.write_text(ZETA_ORG, wait_seconds=10)
                await memory.drain()
                entities = await _entities_named(memory, "Hydra")
                assert [entity.type_label for entity in entities] == [
                    "Object",
                    "Organization",
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_two_people_with_one_name_and_different_emails_stay_two_entities(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Different grounded emails keep namesakes separate."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                for text in (ANA_ONE, ANA_TWO):
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                assert len(await _entities_named(memory, "Ana")) == 2
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_a_mention_with_a_known_email_joins_that_person(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A matching email resolves a mention to the stored person."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                for text in (ANA_ONE, ANA_TWO, ANA_ONE_AGAIN):
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                assert len(await _entities_named(memory, "Ana")) == 2
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_an_email_the_text_does_not_state_is_ignored(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An ungrounded extracted email does not split a person."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                for text in (ANA_ONE, ANA_UNSTATED):
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                assert len(await _entities_named(memory, "Ana")) == 1
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_a_person_without_an_email_joins_the_first_person_with_one(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A mention without an email joins the first matching person."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                for text in (ANA_NONE, ANA_ONE, ANA_TWO):
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                assert len(await _entities_named(memory, "Ana")) == 2
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_namesakes_do_not_multiply_when_a_mention_has_no_email(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Later ambiguous mentions do not create more namesakes."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                for text in (ANA_ONE, ANA_TWO, ANA_NONE, ANA_NONE_AGAIN):
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                assert len(await _entities_named(memory, "Ana")) == 2
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_an_unknown_entity_and_a_typed_entity_with_one_name_become_one_entity(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A typed mention promotes a prior Unknown entity."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(MODEL_LABEL, wait_seconds=10)
                await memory.write_text(HYDRA_ORG, wait_seconds=10)
                await memory.drain()
                entities = await _entities_named(memory, "Zeta")
                assert len(entities) == 1
                assert entities[0].type_label == "Organization"
                assert entities[0].proposed_type == ""
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_structured_and_text_writes_share_one_entity(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A typed structured relation endpoint merges with its text mention."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_fact("Ana", "WORKS_AT", "Acme", wait_seconds=10)
                await memory.write_text(FIRST, wait_seconds=10)
                await memory.drain()
                entities = await _entities_named(memory, "Acme")
                assert [entity.type_label for entity in entities] == ["Organization"]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_multi_pattern_relation_endpoint_merges_with_typed_mention(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An initially ambiguous endpoint gains its type from a later mention."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_fact("Ana", "STUDENT_OF", "MIT", wait_seconds=10)
                await memory.write_text(STUDIED, wait_seconds=10)
                await memory.drain()
                mit = await _entities_named(memory, "MIT")
                ana = await _entities_named(memory, "Ana")
                assert [entity.type_label for entity in mit] == ["Organization"]
                assert [entity.type_label for entity in ana] == ["Person"]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_host_unknown_type_is_kept_as_text(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Host supplied types outside the ontology remain visible as text."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_facts(
                    "Voyager uses Acme.",
                    [
                        HostFact(
                            subject="Voyager",
                            subject_type="Spaceship",
                            relation="USES",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Voyager uses Acme",
                        )
                    ],
                    wait_seconds=10,
                )
                await memory.drain()
                voyager = await _entities_named(memory, "Voyager")
                acme = await _entities_named(memory, "Acme")
                assert [
                    (entity.type_label, entity.proposed_type) for entity in voyager
                ] == [("Unknown", "Spaceship")]
                assert [entity.type_label for entity in acme] == ["Organization"]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_unknown_mentions_do_not_multiply_same_name_entities(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Repeated unknown mentions and structured writes keep one entity per name."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, wait_seconds=10)
                await memory.write_text(ACME_PREFERENCE, wait_seconds=10)
                await memory.write_fact("Ana", "OWNS", "Acme", wait_seconds=10)
                await memory.write_fact("Ana", "USES", "Acme", wait_seconds=10)
                await memory.drain()
                entities = await _entities_named(memory, "Acme")
                assert [entity.type_label for entity in entities] == [
                    "Organization",
                    "Preference",
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_default_relation_is_stored_and_recalled(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A fact on a default relation is stored and found by search."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                receipt = await memory.write_text(
                    FIRST, reference_time=EARLIER, wait_seconds=10
                )
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                context = await memory.search("Where does Ana work?")
                assert [fact.statement for fact in context.facts] == [
                    "Ana works at Acme"
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_single_valued_relation_keeps_only_the_latest_object(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A newer single-valued fact replaces the older object."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, reference_time=EARLIER, wait_seconds=10)
                await memory.write_text(SECOND, reference_time=LATER, wait_seconds=10)
                await memory.drain()
                view = await memory.get_entity("Ana")
                assert [fact.statement for fact in view.facts] == ["Ana works at Beta"]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_unknown_relation_is_refused_and_counted(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An unknown relation is refused and counted in the job result."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                receipt = await memory.write_text(
                    UNKNOWN_RELATION, reference_time=EARLIER, wait_seconds=10
                )
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["ontology_refused"] == 1
                assert job.result["fact_ids"] == []
                context = await memory.search("Who does Ana mentor?")
                assert context.facts == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_catalog_typed_entity_reads_back_with_its_type_label(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An entity the model typed from the catalog keeps that type's label."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, reference_time=EARLIER, wait_seconds=10)
                await memory.drain()
                acme = await memory.get_entity("Acme")
                ana = await memory.get_entity("Ana")
                assert acme.entity.type_label == "Organization"
                assert ana.entity.type_label == "Person"
                assert acme.entity.proposed_type == ""
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_model_proposed_entity_type_is_stored_as_text_on_an_unknown_entity(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A type name the catalog lacks is kept as text and the entity is Unknown."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(
                    MODEL_LABEL, reference_time=EARLIER, wait_seconds=10
                )
                await memory.drain()
                view = await memory.get_entity("Zeta")
                assert view.entity.type_label == "Unknown"
                assert view.entity.proposed_type == "Startup"
        finally:
            await falkordb_store.erase_tenant(tenant)
