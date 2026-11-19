"""Doctor and model cache commands report setup state safely."""

from contextlib import asynccontextmanager

from ontogr_mem.cli import doctor, warm


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
