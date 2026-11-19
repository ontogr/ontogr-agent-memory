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
