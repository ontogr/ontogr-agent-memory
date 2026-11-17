"""Tests for loading ontology declarations when opening Memory."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.ontology.errors import InvalidChange
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


class TestOntologyAtOpen:
    """Memory opens with the configured ontology declaration."""

    async def test_a_yaml_file_defines_the_types_a_text_write_can_use(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """Store, search, and reuse a custom relation after reopening."""
        tenant = new_tenant()
        ontology = tmp_path / "ontology.yaml"
        ontology.write_text(
            "entities:\n  - label: Person\n  - label: Employer\n"
            "relations:\n  - label: EMPLOYS\n    patterns:\n"
            "      - source: Employer\n        target: Person\n",
            encoding="utf-8",
        )
        script = {
            "Acme employs Ana.": (
                [ScriptedEntity("Acme", "Employer"), ScriptedEntity("Ana", "Person")],
                [ScriptedFact("Acme employs Ana", 0, 1, "EMPLOYS", "Acme employs Ana")],
            )
        }
        script["Beta employs Ben."] = (
            [ScriptedEntity("Beta", "Employer"), ScriptedEntity("Ben", "Person")],
            [ScriptedFact("Beta employs Ben", 0, 1, "EMPLOYS", "Beta employs Ben")],
        )
        try:
            async with open_memory(
                tenant, text_models(script), ontology=ontology
            ) as memory:
                receipt = await memory.write_text("Acme employs Ana.", wait_seconds=10)
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                result = await memory.search("Who does Acme employ?")
                assert [fact.statement for fact in result.facts] == ["Acme employs Ana"]
            async with open_memory(tenant, text_models(script)) as memory:
                receipt = await memory.write_text("Beta employs Ben.", wait_seconds=10)
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_a_bad_file_fails_when_memory_opens(self, tmp_path: Path):
        """A missing ontology file is rejected while opening Memory."""
        with pytest.raises(InvalidChange):
            async with open_memory(
                new_tenant(), text_models({}), tmp_path / "missing.yaml"
            ):
                pass
