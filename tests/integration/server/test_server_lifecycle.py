"""Integration tests for the server process lifecycle."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from tests.support.falkordb import falkordb_test_url


@pytest.fixture
def server_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[subprocess.Popen[str], Path]]:
    """Start a server with isolated state and stop it after the test."""
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    state_database = tmp_path / "state.sqlite3"
    server_log = tmp_path / "server.log"
    environment = os.environ.copy()
    environment.update(
        {
            "MEMORY_DATA_DIR": str(data_directory),
            "MEMORY_GRAPH_URL": falkordb_test_url(),
            "MEMORY_STATE_URL": f"sqlite:///{state_database}",
            "MEMORY_MCP_API_KEY": "lifecycle-test-key",
            "MEMORY_SERVER_IDLE_MINUTES": "30",
        }
    )
    with server_log.open("w") as log_file:
        process = subprocess.Popen(
        [sys.executable, "-m", "ontogr_mem", "server", "run"],
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    monkeypatch.setenv("MEMORY_DATA_DIR", str(data_directory))
    try:
        server_info = data_directory / "server.json"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"server exited during startup with {process.returncode}:\n"
                    f"{server_log.read_text()}"
                )
            if server_info.exists():
                info = json.loads(server_info.read_text())
                try:
                    with urlopen(f"http://127.0.0.1:{info['port']}/health", timeout=1):
                        break
                except (OSError, URLError):
                    pass
            time.sleep(0.1)
        else:
            pytest.fail(
                f"server did not become healthy within 30 seconds:\n"
                f"{server_log.read_text()}"
            )
        yield process, data_directory
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


class TestServerLifecycle:
    def test_health_auth_and_shutdown(self, server_process: tuple[subprocess.Popen[str], Path]) -> None:
        process, data_directory = server_process
        info = json.loads((data_directory / "server.json").read_text())
        base_url = f"http://127.0.0.1:{info['port']}"

        with urlopen(f"{base_url}/health", timeout=2) as response:
            health = json.load(response)
        assert health["version"]
        assert health["store"]

        with pytest.raises(HTTPError) as unauthorized:
            urlopen(f"{base_url}/mcp", timeout=2)
        assert unauthorized.value.code == 401

        request = Request(
            f"{base_url}/mcp",
            headers={"Authorization": "Bearer lifecycle-test-key"},
        )
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as error:
            assert error.code != 401
        else:
            response.close()

        shutdown = Request(
            f"{base_url}/shutdown",
            data=b"{}",
            headers={"Authorization": "Bearer lifecycle-test-key"},
            method="POST",
        )
        with urlopen(shutdown, timeout=2) as response:
            assert response.status == 202
        assert process.wait(timeout=10) == 0
        assert not (data_directory / "server.json").exists()
