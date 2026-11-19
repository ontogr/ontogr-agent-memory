"""Verify state store selection from configured URLs."""

from uuid import uuid4

from ontogr_mem.relational_store import StateSettings, open_state_store
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from ontogr_mem.relational_store.sqlite import SqliteJobStore
from ontogr_mem.store_urls import sqlite_url


class TestStateUrl:
    """Select the relational adapter from a typed state URL."""

    async def test_opens_postgres_store(self) -> None:
        """A Postgres URL opens the Postgres adapter."""
        settings = StateSettings(
            url=(
                "postgresql://memory:memorytest123@127.0.0.1:15433/memory"
                f"?application_name=state-url-{uuid4().hex}"
            ),
            _env_file=None,
        )

        store = await open_state_store(settings, owner_id=uuid4().hex)

        assert isinstance(store, PostgresJobStore)

    async def test_opens_sqlite_store(self, tmp_path) -> None:
        """A SQLite URL opens the SQLite adapter."""
        store = await open_state_store(
            StateSettings(url=sqlite_url(tmp_path / "state.sqlite3"), _env_file=None),
            owner_id=uuid4().hex,
        )

        assert isinstance(store, SqliteJobStore)
        await store.close()
