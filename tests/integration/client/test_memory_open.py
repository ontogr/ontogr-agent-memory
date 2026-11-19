"""Integration tests for opening a Memory client."""

from pathlib import Path

import pytest

from ontogr_mem.client import Memory
from ontogr_mem.errors import StoreInUseError
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings
from tests.support.falkordb import falkordb_test_url
from tests.support.scripted_model import scripted_role_models


class TestMemoryOpen:
    """Verify opening and locking memory clients."""

    async def test_locks_sqlite_store_until_context_exits(self, tmp_path: Path) -> None:
        """Hold the SQLite lock until the client context exits."""
        state = StateSettings(
            url=f"sqlite:///{tmp_path / 'state.sqlite3'}", _env_file=None
        )
        options = {
            "service_settings": ServiceSettings(
                graph_url=falkordb_test_url(), _env_file=None
            ),
            "state_settings": state,
            "models": scripted_role_models(),
        }
        options["embedder"] = None
        async with Memory.open(**options):
            with pytest.raises(StoreInUseError, match="Memory.connect"):
                async with Memory.open(**options):
                    pass
        async with Memory.open(**options):
            pass

    async def test_defers_incomplete_endpoint_embedder_error(self, monkeypatch) -> None:
        """Defer endpoint configuration errors until embedding is needed."""
        monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "endpoint")
        monkeypatch.delenv("MEMORY_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMORY_EMBEDDING_BASE_URL", raising=False)
        monkeypatch.delenv("MEMORY_EMBEDDING_DIMENSIONS", raising=False)
        state = StateSettings(_env_file=None)
        async with Memory.open(
            service_settings=ServiceSettings(
                graph_url=falkordb_test_url(), _env_file=None
            ),
            state_settings=state,
            models=scripted_role_models(),
        ):
            pass
