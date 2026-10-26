"""Tests for the FalkorDB vector backend."""

from typing import cast

from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.vectorstore.graph.falkordb import FalkorDBVectorBackend
from ontogr_mem.vectorstore.models import SearchFilters, VectorKind
from ontogr_mem.vectorstore.settings import VectorStoreSettings


class RecordingRunner:
    """A runner that answers with no rows and remembers what it was asked."""

    def __init__(self) -> None:
        """Create an empty request log."""
        self.asked: list[int] = []

    async def run(self, spec: QuerySpec) -> Rows:
        """Record the requested neighbour count and answer with no rows."""
        self.asked.append(int(spec.params["k"]))
        return Rows(rows=[], stats={})


class RunnerStore:
    """A graph store that lends one recording runner."""

    def __init__(self, runner: RecordingRunner) -> None:
        """Bind the store to its runner."""
        self._runner = runner

    def runner(self, handle: GraphHandle) -> RecordingRunner:
        """Return the recording runner."""
        return self._runner


class TestFalkorDBVectorBackend:
    """Behavior of `FalkorDBVectorBackend`."""

    async def test_the_server_is_asked_for_the_planned_number_of_neighbours(
        self,
    ) -> None:
        """The over-fetch factor is applied by the planner, once."""
        runner = RecordingRunner()
        backend = FalkorDBVectorBackend(
            cast(GraphStore, RunnerStore(runner)),
            GraphHandle(tenant="t1", graph="g1"),
            VectorStoreSettings(overfetch=4.0),
        )

        await backend.search(
            VectorKind.NODE, (1.0, 0.0, 0.0), 10, SearchFilters(), None
        )

        assert runner.asked == [10]
