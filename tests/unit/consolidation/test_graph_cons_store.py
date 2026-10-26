"""Lease behavior of the graph adapter when the server refuses every guarded write.

The graph store is replaced at its port by a double that answers reads with no
lease row and refuses each guarded write. The adapter must then report that it
holds no lease and that a rerun request was not stored.
"""

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from ontogr_mem.consolidation.errors import StoreError
from ontogr_mem.graphstore.cons_store import GraphConsStore
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle, Guard


NOW = datetime(2030, 1, 1, tzinfo=UTC)


class RefusingGraph:
    """A graph store whose guarded writes never hold."""

    def runner(self, handle: GraphHandle) -> "RefusingGraph":
        """Return the double itself as the scoped runner."""
        return self

    async def get_nodes(self, handle: GraphHandle, label: str, ids: Any) -> list:
        """Answer with no lease row."""
        return []

    async def guarded_write(
        self, handle: GraphHandle, guard: Guard, nodes: Any = (), edges: Any = ()
    ) -> bool:
        """Refuse every guarded write."""
        return False


def _store() -> GraphConsStore:
    graph = cast(GraphStore, RefusingGraph())
    handle = GraphHandle(tenant="t1", graph="g1")
    return GraphConsStore(graph, handle, tenant_keyed=False, lease_attempts=3)


async def test_acquire_gives_up_when_the_lease_cannot_be_written() -> None:
    """Acquire gives up when the lease cannot be written."""
    store = _store()

    assert await store.acquire("holder", NOW, 60) is False


async def test_a_rerun_request_that_cannot_be_stored_is_an_error() -> None:
    """A rerun request that cannot be stored raises StoreError."""
    store = _store()

    with pytest.raises(StoreError):
        await store.request_rerun()
