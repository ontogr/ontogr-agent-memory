"""The outbox contract on the in-memory reference implementation."""

import pytest

from ontogr_mem.vectorstore.pgvector.outbox import InMemoryOutbox, Outbox
from tests.contracts.outbox import OutboxContract


class TestInMemoryOutbox(OutboxContract):
    """`InMemoryOutbox` satisfies the outbox contract."""

    @pytest.fixture
    def make_outbox(self):
        """Return an async factory for a fresh in-memory outbox."""

        async def make() -> Outbox:
            return InMemoryOutbox()

        return make
