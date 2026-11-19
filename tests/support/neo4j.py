"""Neo4j settings for integration tests."""

import os
from urllib.parse import quote

from ontogr_mem.graphstore import ResilienceSettings
from ontogr_mem.graphstore.neo4j import Neo4jSettings


def neo4j_test_url(*, uri: str | None = None) -> str:
    """Return the integration Neo4j URL, allowing a direct override."""
    if uri is not None:
        return uri
    override = os.environ.get("INTEGRATION_NEO4J_URL")
    if override is not None:
        return override
    password = quote(os.environ.get("MEMORY_NEO4J_PASSWORD", "memorytest123"), safe="")
    return f"bolt://neo4j:{password}@localhost:17687?database=neo4j"


def neo4j_test_settings(**resilience: float) -> Neo4jSettings:
    """Return settings for the compose Neo4j service.

    Args:
        **resilience: Overrides for `ResilienceSettings` fields.
    """
    values = {"retry_delay": 0.05, **resilience}
    return Neo4jSettings(resilience=ResilienceSettings(**values))
