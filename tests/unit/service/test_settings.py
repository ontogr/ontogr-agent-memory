"""Service settings select the graph backend."""

import pytest
from pydantic import ValidationError

from ontogr_mem.service.bootstrap import ServiceSettings
from ontogr_mem.store_urls import DEFAULT_GRAPH_URL


class TestServiceSettings:
    """The backend setting uses the service environment prefix."""

    def test_defaults_to_local_graph_url(self, monkeypatch):
        """The default graph URL points at the local graph store."""
        monkeypatch.delenv("MEMORY_GRAPH_URL", raising=False)

        settings = ServiceSettings(_env_file=None)

        assert settings.graph_url.get_secret_value() == DEFAULT_GRAPH_URL

    def test_reads_graph_url_environment_value(self, monkeypatch):
        """The graph URL comes from MEMORY_GRAPH_URL."""
        monkeypatch.setenv("MEMORY_GRAPH_URL", "neo4j://db.internal")

        settings = ServiceSettings(_env_file=None)

        assert settings.graph_url.get_secret_value() == "neo4j://db.internal"

    def test_rejects_a_bad_graph_url(self, monkeypatch):
        """Malformed graph URLs are rejected during settings validation."""
        monkeypatch.setenv("MEMORY_GRAPH_URL", "memory://")

        with pytest.raises(ValidationError):
            ServiceSettings(_env_file=None)
