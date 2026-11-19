"""Neo4j settings for integration tests."""

import os

from ontogr_mem.graphstore import ResilienceSettings
from ontogr_mem.graphstore.neo4j import Neo4jSettings


def neo4j_test_url(*, uri: str | None = None) -> str:
    """Return the integration Neo4j URL, allowing a direct override."""
    if uri is not None:
        return uri
    return os.environ.get(
        "INTEGRATION_NEO4J_URL",
        "bolt://neo4j:memorytest123@localhost:17687?database=neo4j",
    )


def neo4j_test_settings(**resilience: float) -> Neo4jSettings:
    """Return settings for the compose Neo4j service.

    Args:
        **resilience: Overrides for `ResilienceSettings` fields.
    """
    values = {"retry_delay": 0.05, **resilience}
    return Neo4jSettings(resilience=ResilienceSettings(**values))
