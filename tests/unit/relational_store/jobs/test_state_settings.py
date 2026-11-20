"""State configuration reads its dedicated environment prefix."""

import math

import pytest

from ontogr_mem.paths import data_directory
from ontogr_mem.relational_store.config import StateSettings


class TestStateSettings:
    """The state store is configured by URL."""

    def test_reads_the_state_environment_prefix(self, monkeypatch):
        """State variables override the durable defaults."""
        monkeypatch.setenv("MEMORY_STATE_URL", "sqlite:///tmp/state.sqlite3")

        settings = StateSettings()

        assert settings.url.get_secret_value() == "sqlite:///tmp/state.sqlite3"

    def test_default_url_is_under_the_data_directory(self, monkeypatch):
        """The default SQLite URL stores state under the user data directory."""
        monkeypatch.delenv("MEMORY_STATE_URL", raising=False)
        monkeypatch.setenv("MEMORY_DATA_DIR", "/tmp/agent-memory-data")

        settings = StateSettings(_env_file=None)

        assert settings.url.get_secret_value() == (
            f"sqlite:///{data_directory() / 'agent_memory_state.sqlite3'}"
        )

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_lease_settings(self, value):
        """Lease durations must be finite positive values."""
        with pytest.raises(ValueError):
            StateSettings(lease_seconds=value, _env_file=None)

        with pytest.raises(ValueError):
            StateSettings(heartbeat_seconds=value, _env_file=None)
