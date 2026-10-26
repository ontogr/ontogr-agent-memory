"""Service settings select the graph backend and ephemeral-storage policy."""

from ontogr_mem.service.bootstrap import ServiceSettings


class TestServiceSettings:
    """Backend and ephemeral settings use the service environment prefix."""

    def test_defaults_to_falkordb_without_ephemeral_permission(self, monkeypatch):
        """Durable graph storage is the default configuration."""
        monkeypatch.delenv("MEMORY_SERVICE_BACKEND", raising=False)
        monkeypatch.delenv("MEMORY_SERVICE_ALLOW_EPHEMERAL", raising=False)

        settings = ServiceSettings(_env_file=None)

        assert settings.backend == "falkordb"
        assert settings.allow_ephemeral is False

    def test_reads_backend_and_ephemeral_environment_values(self, monkeypatch):
        """The MEMORY_SERVICE_ variables override the durable defaults."""
        monkeypatch.setenv("MEMORY_SERVICE_BACKEND", "memory")
        monkeypatch.setenv("MEMORY_SERVICE_ALLOW_EPHEMERAL", "true")

        settings = ServiceSettings(_env_file=None)

        assert settings.backend == "memory"
        assert settings.allow_ephemeral is True
