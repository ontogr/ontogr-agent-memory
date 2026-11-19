"""Doctor and model cache commands report setup state safely."""

from contextlib import asynccontextmanager

from ontogr_mem.cli import doctor, warm
from ontogr_mem.relational_store import probe
from ontogr_mem.store_urls import parse_graph_url, parse_state_url


class TestDoctor:
    """Diagnostics report failed connections."""

    async def test_reports_unreachable_graph_and_returns_failure(
        self, monkeypatch, capsys
    ):
        """A failed graph health check sets a nonzero status."""
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://127.0.0.1:6379")
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///missing-state.db")

        @asynccontextmanager
        async def unavailable_graph():
            raise ConnectionError("unavailable")
            yield

        monkeypatch.setattr(doctor, "open_graph_store", unavailable_graph)
        monkeypatch.setattr(doctor, "_check_preflight", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_tenant_rules", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_embedding", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_models", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_server_env", lambda *_args: None)
        monkeypatch.setattr(doctor, "_check_server", lambda *_args: None)
        monkeypatch.setattr(doctor, "_print_locations", lambda *_args: None)

        result = await doctor.run_doctor()

        assert result == 1
        assert "[fail] graph reachable" in capsys.readouterr().out

    async def test_checks_sqlite_state_and_lists_tenants(self, monkeypatch, tmp_path):
        """A reachable graph reports discovered tenants and first use state."""

        @asynccontextmanager
        async def graph_store():
            class Store:
                async def health(self):
                    return None

                async def list_tenants(self):
                    return ["alpha", "beta"]

            yield Store()

        graph = parse_graph_url("redis://127.0.0.1:6379")
        state = parse_state_url(f"sqlite:///{tmp_path}/new.db")
        monkeypatch.setattr(doctor, "open_graph_store", graph_store)
        results = []
        await doctor._check_state(results, state)
        tenants = await doctor._check_graph(results, graph)
        assert tenants == ["alpha", "beta"]
        assert results == [
            ("pass", "state", f"{tmp_path}/new.db created on first use", None),
            ("pass", "graph reachable", graph.identity(), None),
        ]

    async def test_postgres_probe_failure_is_reported(self, monkeypatch):
        """Probe exceptions become actionable state failures."""

        async def unavailable(_state):
            raise OSError("down")

        monkeypatch.setattr(probe, "probe_state", unavailable)
        results = []
        await doctor._check_state(results, parse_state_url("postgresql://localhost/db"))
        assert results == [
            (
                "fail",
                "state",
                "PostgreSQL probe failed (OSError)",
                "check MEMORY_STATE_URL",
            )
        ]

    def test_cache_probe_handles_cache_miss(self, monkeypatch):
        """A missing local model suggests the warm command."""
        monkeypatch.setattr(doctor, "_has_extra", lambda _name: True)
        monkeypatch.setattr(doctor, "model_cached", lambda _settings: False)
        results = []
        doctor._check_embedding(results)
        assert results[0][:3] == ("warn", "embedder", "model not cached")
        assert results[0][3] == "ontogr-mem warm"

    def test_cache_probe_reports_unexpected_error_class_only(self, monkeypatch):
        """Probe failures are visible without exposing exception details."""
        monkeypatch.setattr(doctor, "_has_extra", lambda _name: True)

        def failed_probe(_settings):
            raise OSError("private path /home/user/secret")

        monkeypatch.setattr(doctor, "model_cached", failed_probe)
        results = []
        doctor._check_embedding(results)
        assert results == [("warn", "embedder", "cache probe failed (OSError)", None)]


class TestWarm:
    """Warm handles endpoint embeddings without local downloads."""

    async def test_endpoint_configuration_skips_local_model_open(
        self, monkeypatch, capsys
    ):
        """Endpoint settings avoid opening a local embedder."""

        async def unexpected_open(_settings):
            raise AssertionError("endpoint embeddings do not need a local model")

        monkeypatch.setattr(warm.LocalEmbedder, "open", unexpected_open)

        result = await warm.warm_model(
            warm.EmbeddingSettings(provider="endpoint", base_url="https://example.test")
        )

        assert result == 0
        assert "there is no local model to download" in capsys.readouterr().out
