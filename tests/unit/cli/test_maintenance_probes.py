"""Behavior tests for local maintenance probes."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ontogr_mem.cli import warm
from ontogr_mem.relational_store import probe
from ontogr_mem.vectorstore.builtin import local
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


def _settings(cache: Path, model: str = "Example/Small") -> EmbeddingSettings:
    return EmbeddingSettings(
        fastembed_kwargs={"model_name": model, "cache_dir": str(cache)}
    )


class TestModelCached:
    """Check cache status using FastEmbed's supported model identities."""

    @pytest.mark.parametrize("cached", [True, False])
    def test_reports_whether_supported_model_directory_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
    ) -> None:
        """Return the status of the canonical model directory."""
        cache = tmp_path / "cache"
        model_dir = cache / "models--example--small"
        if cached:
            model_dir.mkdir(parents=True)
        fake_fastembed = SimpleNamespace(
            TextEmbedding=SimpleNamespace(
                list_supported_models=lambda: [
                    {"model": "Example/Small", "hf": "example/small"}
                ]
            )
        )
        monkeypatch.setattr(local, "import_extra", lambda *_args: fake_fastembed)

        assert local.model_cached(_settings(cache)) is cached

    def test_returns_unknown_for_unsupported_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return unknown when FastEmbed does not support the model."""
        monkeypatch.setattr(
            local,
            "import_extra",
            lambda *_args: SimpleNamespace(
                TextEmbedding=SimpleNamespace(list_supported_models=lambda: [])
            ),
        )

        assert local.model_cached(_settings(tmp_path, "unknown/model")) is None


class TestWarmModel:
    """Check local model warming lifecycle without downloading a model."""

    async def test_loads_and_closes_local_embedder(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Close a successfully loaded local embedder after reporting its size."""
        events: list[str] = []

        class FakeEmbedder:
            dim = 384

            async def aclose(self) -> None:
                events.append("closed")

        async def open_embedder(settings: EmbeddingSettings) -> FakeEmbedder:
            assert settings.provider == "auto"
            events.append("opened")
            return FakeEmbedder()

        monkeypatch.setattr(warm, "model_cache_directory", lambda: Path("/fake/cache"))
        monkeypatch.setattr(warm, "LocalEmbedder", SimpleNamespace(open=open_embedder))

        result = await warm.warm_model(EmbeddingSettings())

        assert result == 0
        assert events == ["opened", "closed"]
        assert "Model ready: 384 dimensions." in capsys.readouterr().out


class TestProbeState:
    """Check PostgreSQL connectivity through a short-lived pool checkout."""

    async def test_executes_read_only_query_and_releases_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run the read-only query and release its acquired connection."""
        events: list[str] = []

        class FakeConnection:
            async def fetchval(self, query: str) -> int:
                events.append(query)
                return 1

        class FakePool:
            @asynccontextmanager
            async def acquire(self):
                events.append("acquired")
                try:
                    yield FakeConnection()
                finally:
                    events.append("released")

        async def _pool() -> FakePool:
            return FakePool()

        monkeypatch.setattr(probe, "pg_pool", lambda _dsn: _pool())

        settings = SimpleNamespace(
            url=SimpleNamespace(get_secret_value=lambda: "postgresql://user@host/db")
        )

        assert await probe.probe_state(settings) is True
        assert events == ["acquired", "SELECT 1", "released"]
