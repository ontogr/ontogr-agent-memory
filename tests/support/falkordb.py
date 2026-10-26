"""FalkorDB settings for integration tests."""

import os

from redis.asyncio import Redis

from ontogr_mem.graphstore import ResilienceSettings
from ontogr_mem.graphstore.falkordb import FalkorDBSettings


def falkordb_test_settings(
    *, port: int | None = None, **resilience: float
) -> FalkorDBSettings:
    """Return settings for the compose FalkorDB service.

    Args:
        port: Override the port, for example to reach a closed one.
        **resilience: Overrides for `ResilienceSettings` fields.
    """
    values = {"retry_delay": 0.05, **resilience}
    return FalkorDBSettings(
        host=os.environ.get("MEMORY_FALKORDB_HOST", "localhost"),
        port=port or int(os.environ.get("MEMORY_FALKORDB_PORT", "16379")),
        resilience=ResilienceSettings(**values),
    )


async def drop_unique_constraint(graph: str, label: str, prop: str) -> None:
    """Drop a unique constraint with a raw command, behind the store's back."""
    settings = falkordb_test_settings()
    client = Redis(host=settings.host, port=settings.port, decode_responses=True)
    try:
        await client.execute_command(
            "GRAPH.CONSTRAINT",
            "DROP",
            graph,
            "UNIQUE",
            "NODE",
            label,
            "PROPERTIES",
            1,
            prop,
        )
    finally:
        await client.aclose()
