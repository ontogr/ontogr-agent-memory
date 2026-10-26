"""Driver and pool for one Neo4j server."""

from neo4j import AsyncGraphDatabase

from ontogr_mem.graphstore.neo4j.runner import Neo4jRunner
from ontogr_mem.graphstore.neo4j.settings import Neo4jSettings


class Neo4jClient:
    """One driver and one runner shared by every tenant."""

    def __init__(self, settings: Neo4jSettings) -> None:
        """Create the driver; it connects on first use."""
        self.settings = settings
        auth = (
            (settings.username, settings.password.get_secret_value())
            if settings.password
            else None
        )
        self._driver = AsyncGraphDatabase.driver(
            settings.uri,
            auth=auth,
            max_connection_pool_size=settings.max_connections,
            connection_acquisition_timeout=settings.pool_timeout,
            connection_timeout=settings.connect_timeout,
            max_transaction_retry_time=settings.transaction_retry_s,
        )
        self.runner = Neo4jRunner(
            lambda: self._driver, settings.resilience, database=settings.database
        )

    async def ping(self) -> None:
        """Check connectivity through the runner."""
        await self.runner.execute(
            self._driver.verify_connectivity, idempotent=True, tag="ping"
        )

    async def close(self) -> None:
        """Close the driver."""
        await self._driver.close()
