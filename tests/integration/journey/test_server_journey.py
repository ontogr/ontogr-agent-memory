"""Exercise the installed package's shared-server journey."""

import asyncio
import contextlib
import json
import os
import pty
import secrets
import select
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from tests.support.falkordb import falkordb_test_url


CONNECT_SCRIPT = """
import asyncio
import os
from ontogr_mem import Memory

async def main():
    tenant = os.environ["MEMORY_MCP_TENANT"]
    async with Memory.connect(tenant=tenant) as memory:
        await memory.search("independent client")

asyncio.run(main())
"""

OPEN_SCRIPT = """
import asyncio
from ontogr_mem import Memory

asyncio.run(Memory.open().__aenter__())
"""


def _run_client(
    installed_cli: Path, script: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a script with the interpreter from the installed environment."""
    return subprocess.run(
        [str(installed_cli.parent / "python"), "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _run_data_command_journey(installed_cli: Path, environment: dict[str, str]) -> None:
    """Check doctor, tenant listing and the clear confirmation flow."""
    tenant = environment["MEMORY_MCP_TENANT"]
    doctor = subprocess.run(
        [str(installed_cli), "doctor"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert doctor.returncode == 0, doctor.stdout + doctor.stderr
    tenants = subprocess.run(
        [str(installed_cli), "tenants"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert tenants.returncode == 0, tenants.stdout + tenants.stderr
    assert tenant in tenants.stdout

    stopped = subprocess.run(
        [str(installed_cli), "server", "stop"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    guarded = subprocess.run(
        [str(installed_cli), "clear", "--tenant", tenant],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert guarded.returncode == 1
    assert "interactive terminal" in guarded.stdout + guarded.stderr
    assert "Stop the server first" not in guarded.stdout + guarded.stderr

    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [str(installed_cli), "clear", "--tenant", tenant],
        env=environment,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    output = bytearray()
    deadline = time.monotonic() + 30
    confirmed = False
    while process.poll() is None and time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.2)
        if ready:
            try:
                output.extend(os.read(master_fd, 4096))
            except OSError:
                break
            if not confirmed and tenant.encode() in output:
                os.write(master_fd, (tenant + "\n").encode())
                confirmed = True
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    os.close(master_fd)
    assert confirmed, output.decode(errors="replace")
    assert process.returncode == 0, output.decode(errors="replace")
    tenants_after = subprocess.run(
        [str(installed_cli), "tenants"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert tenants_after.returncode == 0, tenants_after.stdout + tenants_after.stderr
    assert tenant not in tenants_after.stdout


@pytest.fixture(scope="module")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and install the local wheel with its server dependencies."""
    directory = tmp_path_factory.mktemp("server-wheel")
    wheelhouse = directory / "wheelhouse"
    wheelhouse.mkdir()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheelhouse)], check=True)
    wheel = next(wheelhouse.glob("ontogr_mem-*.whl"))
    environment = directory / "venv"
    subprocess.run(["uv", "venv", str(environment)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment / "bin/python"),
            f"{wheel}[local]",
        ],
        check=True,
    )
    return environment / "bin/ontogr-mem"


@pytest.fixture
def journey_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    """Isolate the server's files and tenant for one journey test."""
    values = {
        "MEMORY_GRAPH_URL": falkordb_test_url(),
        "MEMORY_STATE_URL": f"sqlite:///{tmp_path / 'state.sqlite'}",
        "MEMORY_DATA_DIR": str(tmp_path / "data"),
        "FASTEMBED_CACHE_PATH": str(tmp_path / "fastembed"),
        "MEMORY_MCP_TENANT": f"journey_{secrets.token_hex(8)}",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return {**os.environ, **values}


@pytest.fixture(autouse=True)
def stop_server(
    installed_cli: Path, journey_env: dict[str, str]
) -> Generator[None, None, None]:
    """Stop any server started by a test, including after assertion failures."""
    yield
    info_path = Path(journey_env["MEMORY_DATA_DIR"]) / "server.json"
    try:
        subprocess.run(
            [str(installed_cli), "server", "stop"],
            env=journey_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        if not info_path.exists():
            return
        try:
            process_id = json.loads(info_path.read_text())["pid"]
            os.kill(process_id, 15)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        with contextlib.suppress(ProcessLookupError):
            os.kill(process_id, 9)


class TestServerJourney:
    """Verify clients share and recover the installed server process."""

    def test_second_process_attaches_to_running_server(
        self, installed_cli: Path, journey_env: dict[str, str]
    ) -> None:
        """Independent client processes can connect to one server instance."""
        first = _run_client(installed_cli, CONNECT_SCRIPT, journey_env)
        assert first.returncode == 0, first.stdout + first.stderr
        info_path = Path(journey_env["MEMORY_DATA_DIR"]) / "server.json"
        before = json.loads(info_path.read_text())["pid"]
        second = _run_client(installed_cli, CONNECT_SCRIPT, journey_env)
        assert second.returncode == 0, second.stdout + second.stderr
        after = json.loads(info_path.read_text())["pid"]
        assert after == before
        assert before > 0

    async def test_stdio_front_writes_waits_and_recalls(
        self, installed_cli: Path, journey_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The installed MCP front stores sourced facts and returns them in search."""
        transport = StdioTransport(
            command=str(installed_cli), args=["mcp"], env=journey_env
        )
        started = time.monotonic()
        async with Client(transport) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools} == {
                "search_memory",
                "get_entity",
                "list_entities",
                "get_ontology",
                "stale_episodes",
                "get_job",
                "write_facts",
                "forget_memory",
                "get_skill",
            }
            forget_tool = next(tool for tool in tools if tool.name == "forget_memory")
            assert "mode" not in forget_tool.inputSchema.get("properties", {})
            source = "Rina works at the Atlas research team."
            write = await client.call_tool(
                "write_facts",
                {
                    "source_text": source,
                    "facts": [
                        {
                            "subject": "Rina",
                            "subject_type": "Person",
                            "relation": "works_at",
                            "object": "Atlas research team",
                            "object_type": "Organization",
                            "evidence_quote": source,
                        }
                    ],
                },
            )
            receipt = write.structured_content
            job_id = receipt["job_id"]
            deadline = time.monotonic() + 60
            while True:
                job_result = await client.call_tool("get_job", {"job_id": job_id})
                job = job_result.structured_content
                if job["status"] in {"done", "failed", "cancelled"}:
                    break
                if time.monotonic() >= deadline:
                    pytest.fail(f"Job {job_id} did not finish within 60 seconds")
                await asyncio.sleep(0.25)
            assert job["status"] == "done", job
            fact_ids = job["result"]["fact_ids"]
            assert len(fact_ids) == 1, job
            search = await client.call_tool(
                "search_memory", {"query": "Atlas research team"}
            )
            assert fact_ids[0] in str(search.structured_content)
        artifact = {
            "steps": ["list_tools", "write_facts", "wait_for_job", "search_memory"],
            "job_id": job_id,
            "fact_id": fact_ids[0],
            "elapsed_seconds": time.monotonic() - started,
        }
        artifact_path = Path(os.environ.get("JOURNEY_ARTIFACT", "journey-result.json"))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
        _run_data_command_journey(installed_cli, journey_env)

    def test_server_recovers_after_crash(
        self, installed_cli: Path, journey_env: dict[str, str]
    ) -> None:
        """A subsequent client restarts a server killed without cleanup."""
        directory = Path(journey_env["MEMORY_DATA_DIR"])
        first = _run_client(installed_cli, CONNECT_SCRIPT, journey_env)
        assert first.returncode == 0, first.stdout + first.stderr
        before = json.loads((directory / "server.json").read_text())["pid"]
        os.kill(before, 9)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                os.kill(before, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"Server process {before} did not exit after SIGKILL")
        recovered = _run_client(installed_cli, CONNECT_SCRIPT, journey_env)
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        after = json.loads((directory / "server.json").read_text())["pid"]
        assert after != before

    def test_memory_open_refuses_while_server_runs(
        self, installed_cli: Path, journey_env: dict[str, str]
    ) -> None:
        """Memory.open rejects SQLite state owned by the managed server."""
        connect = _run_client(installed_cli, CONNECT_SCRIPT, journey_env)
        assert connect.returncode == 0, connect.stdout + connect.stderr
        opened = _run_client(installed_cli, OPEN_SCRIPT, journey_env)
        assert opened.returncode != 0
        assert "Memory.connect" in opened.stderr
