"""Run job runner behavior against transient and durable state stores."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.relational_store.memory.jobs import MemoryJobStore
from ontogr_mem.relational_store.sqlite.jobs import SqliteJobStore


@pytest.fixture(params=["memory", "sqlite"])
async def job_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[StateStore]:
    """Yield an isolated store and close it after the runner test."""
    store = (
        MemoryJobStore()
        if request.param == "memory"
        else SqliteJobStore(str(tmp_path / "state.sqlite3"))
    )
    try:
        yield store
    finally:
        await store.close()
