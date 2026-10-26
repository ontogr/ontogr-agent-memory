"""Ephemeral state requires an explicit opt-in at the service boundary."""

import pytest

from ontogr_mem.relational_store import (
    EphemeralStateError,
    StateSettings,
    open_state_store,
)
from ontogr_mem.relational_store.memory import MemoryJobStore


class TestStateFactory:
    """The factory enforces the durability setting."""

    @pytest.mark.parametrize("allow_ephemeral", [False, True])
    async def test_refuses_the_memory_backend_unless_ephemeral_is_allowed(
        self, allow_ephemeral
    ):
        """Only an opted-in caller receives the memory adapter."""
        settings = StateSettings(backend="memory", allow_ephemeral=allow_ephemeral)

        if not allow_ephemeral:
            with pytest.raises(
                EphemeralStateError, match="MEMORY_STATE_ALLOW_EPHEMERAL"
            ):
                await open_state_store(settings)
            return

        store = await open_state_store(settings)
        assert isinstance(store, MemoryJobStore)
        await store.close()
