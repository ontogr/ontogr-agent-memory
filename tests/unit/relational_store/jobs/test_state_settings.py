"""State configuration reads its dedicated environment prefix."""

from ontogr_mem.relational_store.config import StateSettings


class TestStateSettings:
    """Backend choice and ephemeral opt-in can come from the environment."""

    def test_reads_the_state_environment_prefix(self, monkeypatch):
        """State variables override the durable defaults."""
        monkeypatch.setenv("MEMORY_STATE_BACKEND", "memory")
        monkeypatch.setenv("MEMORY_STATE_ALLOW_EPHEMERAL", "true")

        settings = StateSettings()

        assert settings.backend == "memory"
        assert settings.allow_ephemeral is True
