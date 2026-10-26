"""Neo4j settings for integration tests."""

import os

from pydantic import SecretStr

from ontogr_mem.graphstore import ResilienceSettings
from ontogr_mem.graphstore.neo4j import Neo4jSettings


def neo4j_test_settings(
    *, uri: str | None = None, **resilience: float
) -> Neo4jSettings:
    """Return settings for the compose Neo4j service.

    Args:
        uri: Override the URI, for example to reach a closed port.
        **resilience: Overrides for `ResilienceSettings` fields.
    """
    values = {"retry_delay": 0.05, **resilience}
    return Neo4jSettings(
        uri=uri or os.environ.get("MEMORY_NEO4J_URI", "bolt://localhost:17687"),
        username=os.environ.get("MEMORY_NEO4J_USERNAME", "neo4j"),
        password=SecretStr(os.environ.get("MEMORY_NEO4J_PASSWORD", "memorytest123")),
        resilience=ResilienceSettings(**values),
    )
