"""Service settings select the graph backend."""

import pytest
from pydantic import ValidationError

from ontogr_mem.service.bootstrap import ServiceSettings


class TestServiceSettings:
    """The backend setting uses the service environment prefix."""

    def test_defaults_to_falkordb(self, monkeypatch):
        """FalkorDB is the default graph backend."""
        monkeypatch.delenv("MEMORY_SERVICE_BACKEND", raising=False)

        settings = ServiceSettings(_env_file=None)

        assert settings.backend == "falkordb"

    def test_reads_backend_environment_value(self, monkeypatch):
        """The MEMORY_SERVICE_BACKEND variable selects Neo4j."""
        monkeypatch.setenv("MEMORY_SERVICE_BACKEND", "neo4j")

        settings = ServiceSettings(_env_file=None)

        assert settings.backend == "neo4j"

    def test_rejects_a_backend_outside_the_graph_stores(self, monkeypatch):
        """Only the graph stores are valid backends."""
        monkeypatch.setenv("MEMORY_SERVICE_BACKEND", "memory")

        with pytest.raises(ValidationError):
            ServiceSettings(_env_file=None)
