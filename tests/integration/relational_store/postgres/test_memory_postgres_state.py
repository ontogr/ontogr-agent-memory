"""Exercise the public Memory client with Postgres state."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.client import Memory
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.relational_store import engine
from ontogr_mem.service.bootstrap import ServiceSettings
from tests.support.embedder import MockEmbedder
from tests.support.tenants import new_tenant
from tests.support.falkordb import falkordb_test_url
from tests.support.scripted_model import scripted_role_models


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point Memory at the compose FalkorDB service."""
    monkeypatch.setenv("MEMORY_GRAPH_URL", falkordb_test_url())
    monkeypatch.setattr(engine, "_pg_pools", {})
    monkeypatch.setattr(engine, "_pg_pool_locks", {})
    yield


class TestMemoryOnPostgresState:
    """Verify the public client can use Postgres for durable state."""

    async def test_write_fact_finishes_with_postgres_state(self) -> None:
        """Persist a completed write and expose its terminal job state."""
        tenant = new_tenant()
        async with Memory.open(
            tenant=tenant,
            service_settings=ServiceSettings(
                graph_url=falkordb_test_url(), _env_file=None
            ),
            models=scripted_role_models(),
            embedder=MockEmbedder(),
        ) as memory:
            receipt = await memory.write_fact(
                "sky", "is", "blue", wait_seconds=10
            )
            job = await memory.get_job(receipt.job_id)

        assert receipt.status == JobState.DONE
        assert job.status == JobState.DONE
