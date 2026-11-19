"""Exercise Memory.connect against an auto-started server process."""

import contextlib
import json
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from ontogr_mem.client import Memory
from ontogr_mem.service.contracts import HostFact
from ontogr_mem.service.errors import InvalidArgumentError, ObjectNotFoundError
from tests.support.falkordb import falkordb_test_url
from tests.support.tenants import new_tenant


@pytest.fixture
def connected_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Configure an isolated server directory and stop its process afterward."""
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMORY_GRAPH_URL", falkordb_test_url())
    monkeypatch.setenv("MEMORY_STATE_URL", f"sqlite:///{tmp_path / 'state.sqlite3'}")
    monkeypatch.setenv("MEMORY_MCP_API_KEY", "connect-test-key")
    monkeypatch.delenv("MEMORY_MCP_SERVER_URL", raising=False)
    yield
    info_path = tmp_path / "data" / "server.json"
    if info_path.exists():
        server_info = json.loads(info_path.read_text())
        process_id = server_info.get("pid")
        if isinstance(process_id, int):
            with contextlib.suppress(ProcessLookupError):
                os.kill(process_id, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    os.kill(process_id, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                os.kill(process_id, signal.SIGKILL)


class TestMemoryConnect:
    """Verify the connected client writes, reads and isolates tenants."""

    async def test_connect_writes_and_reads_fact(self, connected_server: None) -> None:
        """Write sourced facts through the server and search them back."""
        tenant = new_tenant()
        source = "Aster likes tea."
        async with Memory.connect(tenant=tenant) as memory:
            receipt = await memory.write_facts(
                source,
                [
                    HostFact(
                        subject="Aster",
                        subject_type="Person",
                        relation="likes",
                        object="tea",
                        object_type="Object",
                        evidence_quote=source,
                    )
                ],
                wait_seconds=10,
            )
            job = await memory.get_job(receipt.job_id)
            assert job.error is None
            result = await memory.search("Aster tea")
            assert result.facts
            assert any(fact.subject.name == "Aster" for fact in result.facts)

    async def test_connected_tenants_are_isolated(self, connected_server: None) -> None:
        """Keep facts written for one tenant out of another tenant's search."""
        tenant_a = new_tenant()
        tenant_b = new_tenant()
        async with Memory.connect(tenant=tenant_a) as memory_a:
            await memory_a.write_facts(
                "Aster likes tea.",
                [
                    HostFact(
                        subject="Aster",
                        subject_type="Person",
                        relation="likes",
                        object="tea",
                        object_type="Object",
                        evidence_quote="Aster likes tea.",
                    )
                ],
                wait_seconds=10,
            )
        async with Memory.connect(tenant=tenant_b) as memory_b:
            result = await memory_b.search("Aster tea")
            assert not result.facts

    async def test_connected_client_reads_catalogs_and_drains(
        self, connected_server: None
    ) -> None:
        """Read entity and catalog views and drain writes through the connection."""
        tenant = new_tenant()
        source = "Aster likes tea."
        async with Memory.connect(tenant=tenant) as memory:
            receipt = await memory.write_facts(
                source,
                [
                    HostFact(
                        subject="Aster",
                        subject_type="Person",
                        relation="likes",
                        object="tea",
                        object_type="Object",
                        evidence_quote=source,
                    )
                ],
                wait_seconds=10,
            )
            assert (await memory.get_job(receipt.job_id)).error is None

            entity = await memory.get_entity("Aster")
            assert entity.entity.name == "Aster"
            assert any(fact.subject.name == "Aster" for fact in entity.facts)

            entities = await memory.list_entities()
            assert any(item.name == "Aster" for item in entities.entities)

            ontology = await memory.get_ontology()
            assert ontology.tenant == tenant
            assert ontology.entity_types

            stale = await memory.stale_episodes()
            assert stale.episodes == []

            await memory.drain()
            assert (await memory.get_job(receipt.job_id)).status.value == "done"

    async def test_connected_client_maps_remote_errors_and_rejects_forget(
        self, connected_server: None
    ) -> None:
        """Translate remote errors and reject direct forget over a connection."""
        tenant = new_tenant()
        async with Memory.connect(tenant=tenant) as memory:
            with pytest.raises(ObjectNotFoundError, match="entity not found"):
                await memory.get_entity("missing-entity")

            source = "Aster likes tea."
            await memory.write_facts(
                source,
                [
                    HostFact(
                        subject="Aster",
                        subject_type="Person",
                        relation="likes",
                        object="tea",
                        object_type="Object",
                        evidence_quote=source,
                    )
                ],
                wait_seconds=10,
            )
            fact_id = (await memory.search("Aster tea")).facts[0].fact_id

            receipt = await memory.invalidate(fact_id, reason="no longer true")
            assert receipt.job_id
            assert (await memory.get_job(receipt.job_id)).error is None

            with pytest.raises(InvalidArgumentError, match="does not support forget"):
                await memory.forget(fact_id=fact_id)
