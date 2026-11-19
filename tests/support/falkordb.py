"""FalkorDB settings for integration tests."""

import os
from urllib.parse import urlsplit, urlunsplit

from redis.asyncio import Redis

from ontogr_mem.graphstore import ResilienceSettings
from ontogr_mem.graphstore.falkordb import FalkorDBSettings
from ontogr_mem.store_urls import parse_graph_url


def falkordb_test_url(*, port: int | None = None) -> str:
    """Return the integration graph URL, optionally using another port."""
    value = os.environ.get("INTEGRATION_FALKORDB_URL", "redis://localhost:16379")
    if port is None:
        return value
    parts = urlsplit(value)
    host = parts.hostname or "localhost"
    if ":" in host:
        host = f"[{host}]"
    userinfo = f"{parts.username}:{parts.password}@" if parts.username else ""
    netloc = f"{userinfo}{host}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def falkordb_test_settings(**resilience: float) -> FalkorDBSettings:
    """Return settings for the compose FalkorDB service.

    Args:
        **resilience: Overrides for `ResilienceSettings` fields.
    """
    values = {"retry_delay": 0.05, **resilience}
    return FalkorDBSettings(
        resilience=ResilienceSettings(**values),
    )


async def drop_unique_constraint(graph: str, label: str, prop: str) -> None:
    """Drop a unique constraint with a raw command, behind the store's back."""
    target = parse_graph_url(falkordb_test_url())
    client = Redis(host=target.host, port=target.port, decode_responses=True)
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
