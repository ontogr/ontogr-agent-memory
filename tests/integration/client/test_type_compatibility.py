"""Client writes resolve entity types through the configured ontology."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from tests.support.falkordb import falkordb_test_settings
from tests.support.tenants import new_tenant
from tests.support.text_writes import (
    ScriptedEntity,
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


def ontology_file(tmp_path: Path, entities: str, name: str = "ontology.yaml") -> Path:
    """Write ontology entities to a temporary YAML file."""
    path = tmp_path / name
    path.write_text(f"entities:\n{entities}\n", encoding="utf-8")
    return path


def script(label_by_text: dict[str, str]):
    """Return text-write answers for the named entity labels."""
    return {
        text: (
            [ScriptedEntity("Hydra deal" if "Hydra" in text else "Acme", label)],
            [],
        )
        for text, label in label_by_text.items()
    }


class TestTypeCompatibility:
    """Custom ontology relationships determine whether client writes merge."""

    @pytest.mark.parametrize("child_first", [False, True])
    async def test_parent_and_child_merge_in_either_order(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path, child_first: bool
    ):
        """A parent and child type resolve to the more specific type."""
        tenant = new_tenant()
        ontology = ontology_file(
            tmp_path,
            "  - label: Contract\n  - label: PurchaseOrder\n    parent: Contract\n"
            "  - label: Person\n    identity_attributes: [email]\n",
        )
        answers = script(
            {
                "Hydra deal as a contract.": "Contract",
                "Hydra deal as a purchase order.": "PurchaseOrder",
            }
        )
        texts = list(answers)
        if child_first:
            texts.reverse()
        try:
            async with open_memory(tenant, text_models(answers), ontology) as memory:
                for text in texts:
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                entities = (await memory.search("Hydra deal")).entities
                assert [(entity.name, entity.type_label) for entity in entities] == [
                    ("Hydra deal", "PurchaseOrder")
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_type_rename_matches_entities_written_before_reopen(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """A renamed type matches entities stored under its former label."""
        tenant = new_tenant()
        old = ontology_file(tmp_path, "  - label: Employer\n", name="old.yaml")
        new = ontology_file(
            tmp_path,
            "  - label: Company\n    renamed_from: [Employer]\n",
            name="new.yaml",
        )
        answers = script(
            {"Acme is an employer.": "Employer", "Acme is a company.": "Company"}
        )
        try:
            async with open_memory(tenant, text_models(answers), old) as memory:
                await memory.write_text("Acme is an employer.", wait_seconds=10)
                await memory.drain()
            async with open_memory(tenant, text_models(answers), new) as memory:
                await memory.write_text("Acme is a company.", wait_seconds=10)
                await memory.drain()
                entities = (await memory.search("Acme")).entities
                assert [(entity.name, entity.type_label) for entity in entities] == [
                    ("Acme", "Company")
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_merged_type_matches_entities_written_before_reopen(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """A merged type matches entities stored under its former id."""
        tenant = new_tenant()
        old = ontology_file(
            tmp_path, "  - label: Firm\n  - label: Company\n", name="old.yaml"
        )
        new = ontology_file(
            tmp_path,
            "  - label: Firm\n    merged_into: Company\n  - label: Company\n",
            name="new.yaml",
        )
        answers = script({"Acme is a firm.": "Firm", "Acme is a company.": "Company"})
        try:
            async with open_memory(tenant, text_models(answers), old) as memory:
                await memory.write_text("Acme is a firm.", wait_seconds=10)
                await memory.drain()
            async with open_memory(tenant, text_models(answers), new) as memory:
                await memory.write_text("Acme is a company.", wait_seconds=10)
                await memory.drain()
                entities = (await memory.search("Acme")).entities
                assert [(entity.name, entity.type_label) for entity in entities] == [
                    ("Acme", "Company")
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_parent_catch_all_merges_with_specific_type(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """A parent catch-all type merges with its specific child."""
        tenant = new_tenant()
        ontology = ontology_file(
            tmp_path, "  - label: Thing\n  - label: Company\n    parent: Thing\n"
        )
        answers = {
            "Hydra is a thing.": ([ScriptedEntity("Hydra", "Thing")], []),
            "Hydra is a company.": ([ScriptedEntity("Hydra", "Company")], []),
        }
        try:
            async with open_memory(tenant, text_models(answers), ontology) as memory:
                for text in answers:
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                entities = (await memory.search("Hydra")).entities
                assert [(entity.name, entity.type_label) for entity in entities] == [
                    ("Hydra", "Company")
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_identity_names_are_inherited_from_parent_type(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """A child type uses its parent's identity attributes."""
        tenant = new_tenant()
        ontology = ontology_file(
            tmp_path,
            "  - label: Person\n    identity_attributes: [email]\n"
            "  - label: Employee\n    parent: Person\n",
        )
        answers = {
            "Ana (ana@one.org) is an employee.": (
                [
                    ScriptedEntity(
                        "Ana", "Employee", attributes={"email": "ana@one.org"}
                    )
                ],
                [],
            ),
            "Ana (ana@two.org) is an employee.": (
                [
                    ScriptedEntity(
                        "Ana", "Employee", attributes={"email": "ana@two.org"}
                    )
                ],
                [],
            ),
        }
        try:
            async with open_memory(tenant, text_models(answers), ontology) as memory:
                for text in answers:
                    await memory.write_text(text, wait_seconds=10)
                await memory.drain()
                entities = (await memory.search("Ana")).entities
                assert len([entity for entity in entities if entity.name == "Ana"]) == 2
        finally:
            await falkordb_store.erase_tenant(tenant)
