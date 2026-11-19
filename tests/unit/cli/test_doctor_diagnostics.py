"""Behavior tests for the public doctor command."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from ontogr_mem.cli import doctor


class TestRunDoctor:
    """The doctor reports actionable diagnostics without leaking secrets."""

    def _quiet_unrelated_checks(self, monkeypatch):
        monkeypatch.setattr(doctor, "_check_preflight", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_embedding", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_models", lambda *_args: None)
        monkeypatch.setattr(doctor, "_print_locations", lambda *_args: None)

    def test_invalid_embedding_settings_are_reported_without_values_or_tracebacks(
        self, monkeypatch
    ):
        """Invalid embedding configuration fails safely at both doctor checks."""
        monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", "5000")
        results = []

        doctor._check_preflight(results, object(), object())
        doctor._check_embedding(results)

        assert (
            results.count(
                ("fail", "embedding settings", "invalid fields: dimensions", None)
            )
            == 2
        )

    def test_state_url_uses_configured_settings_default(self, monkeypatch):
        """Doctor parses the same state URL used by service settings."""
        monkeypatch.delenv("MEMORY_STATE_URL", raising=False)

        _, state = doctor._check_urls([])

        assert (
            state.path
            == doctor.parse_state_url(
                doctor.StateSettings().url.get_secret_value()
            ).path
        )

    async def test_reports_missing_extra_and_redacts_credentials(
        self, monkeypatch, capsys
    ):
        """Missing installation extras fail while credential values stay hidden."""
        monkeypatch.setattr(doctor, "_check_embedding", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_models", lambda *_args: None)
        monkeypatch.setattr(doctor, "_print_locations", lambda *_args: None)
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://:top-secret@localhost:6379")
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///memory.db")
        monkeypatch.setattr(
            doctor,
            "check_requirements",
            lambda **_kwargs: SimpleNamespace(missing=["graph"], required=[]),
        )

        @asynccontextmanager
        async def unavailable_graph():
            raise ConnectionError("top-secret")
            yield

        monkeypatch.setattr(doctor, "open_graph_store", unavailable_graph)
        monkeypatch.setattr(doctor, "_check_server_env", lambda _results: None)
        monkeypatch.setattr(doctor, "_check_server", lambda _results: None)

        status = await doctor.run_doctor()

        output = capsys.readouterr().out
        assert status == 1
        assert "missing extras: graph" in output
        assert "top-secret" not in output

    async def test_invalid_urls_are_reported_without_echoing_input(
        self, monkeypatch, capsys
    ):
        """Invalid store URLs fail without repeating user supplied values."""
        self._quiet_unrelated_checks(monkeypatch)
        monkeypatch.setenv("MEMORY_GRAPH_URL", "https://user:graph-secret@bad")
        monkeypatch.setenv("MEMORY_STATE_URL", "state-secret")
        monkeypatch.setattr(doctor, "_check_server_env", lambda _results: None)
        monkeypatch.setattr(doctor, "_check_server", lambda _results: None)

        status = await doctor.run_doctor()

        output = capsys.readouterr().out
        assert status == 1
        assert "[fail] graph URL: invalid URL" in output
        assert "[fail] state: invalid URL" in output
        assert "graph-secret" not in output
        assert "state-secret" not in output

    async def test_remote_graph_requires_tenant_and_warns_about_sqlite(
        self, monkeypatch, capsys
    ):
        """Remote graphs require an explicit tenant and flag local SQLite state."""
        monkeypatch.setattr(doctor, "_check_preflight", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_embedding", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_models", lambda *_args: None)
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://graph.example:6379")
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///memory.db")
        monkeypatch.delenv("MEMORY_MCP_TENANT", raising=False)
        monkeypatch.setattr(doctor, "_check_server_env", lambda _results: None)
        monkeypatch.setattr(doctor, "_check_server", lambda _results: None)

        status = await doctor.run_doctor()

        output = capsys.readouterr().out
        assert status == 1
        assert "remote graph requires an explicit tenant" in output
        assert "SQLite state is local to this machine" in output

    async def test_server_env_reports_keys_redacts_values_and_flags_invalid_settings(
        self, monkeypatch, tmp_path, capsys
    ):
        """Server environment keys are listed safely and forbidden keys fail."""
        self._quiet_unrelated_checks(monkeypatch)
        monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
        (tmp_path / "server.env").write_text(
            "OPENAI_API_KEY=env-secret\nMEMORY_GRAPH_URL=redis://bad\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MEMORY_GRAPH_URL", "not-a-url")
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///memory.db")
        monkeypatch.setattr(doctor, "_check_server", lambda _results: None)

        status = await doctor.run_doctor()

        output = capsys.readouterr().out
        assert status == 1
        assert "keys: MEMORY_GRAPH_URL, OPENAI_API_KEY" in output
        assert "env-secret" not in output
        assert "invalid URL" in output
        assert "settings file" in output and "[fail]" in output

    async def test_healthy_graph_reports_tenant_count(self, monkeypatch, capsys):
        """A healthy graph returns success and prints the discovered tenant count."""
        monkeypatch.setattr(doctor, "_check_preflight", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_embedding", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_models", lambda *_args: None)
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://localhost:6379")
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///memory.db")
        monkeypatch.setattr(doctor, "_check_server_env", lambda _results: None)
        monkeypatch.setattr(doctor, "_check_server", lambda _results: None)

        @asynccontextmanager
        async def healthy_graph():
            class Store:
                async def health(self):
                    return None

                async def list_tenants(self):
                    return ["alpha", "beta"]

            yield Store()

        monkeypatch.setattr(doctor, "open_graph_store", healthy_graph)

        status = await doctor.run_doctor()

        output = capsys.readouterr().out
        assert status == 0
        assert "[pass] graph reachable: redis://localhost:6379" in output
        assert "tenants: 2" in output

    async def test_server_fingerprint_mismatch_is_reported(self, monkeypatch, capsys):
        """A running server with different extraction roles emits a warning."""
        self._quiet_unrelated_checks(monkeypatch)
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://localhost:6379")
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///memory.db")
        monkeypatch.setattr(doctor, "_check_server_env", lambda _results: None)
        monkeypatch.setattr(doctor, "read_server_info", lambda _path: {"port": 8123})
        monkeypatch.setattr(doctor, "read_server_token", lambda _path: None)

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __init__(self):
                self.health = {"roles": {}}

            def read(self):
                return json.dumps(self.health).encode()

        monkeypatch.setattr(
            doctor.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
        )

        @asynccontextmanager
        async def healthy_graph():
            class Store:
                async def health(self):
                    return None

                async def list_tenants(self):
                    return []

            yield Store()

        monkeypatch.setattr(doctor, "open_graph_store", healthy_graph)

        status = await doctor.run_doctor()

        output = capsys.readouterr().out
        assert status == 0
        assert "extraction roles differ from these settings" in output
