"""FalkorDB runner result conversion and schema-command timeouts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ontogr_mem.graphstore.config import ResilienceSettings
from ontogr_mem.graphstore.errors import (
    QueryTimeoutError,
    StoreQueryError,
    ValidationError,
)
from ontogr_mem.graphstore.falkordb.runner import QueryRunner, to_rows
from ontogr_mem.graphstore.falkordb.schema import SchemaManager
from ontogr_mem.graphstore.indices import SchemaPart


class TestFalkorDBRows:
    """Behavior of FalkorDB result conversion."""

    def test_rejects_rows_with_a_different_column_count(self):
        """Malformed server results fail instead of losing trailing values."""
        result = SimpleNamespace(header=["left", "right"], result_set=[["value"]])

        with pytest.raises(StoreQueryError, match="expected 2"):
            to_rows(result)

    def test_sorts_node_labels_in_returned_rows(self):
        """Server label order does not affect the normalized row."""
        node = SimpleNamespace(labels={"Zebra", "Alpha"}, properties={"id": "n1"})
        result = SimpleNamespace(header=["node"], result_set=[[node]], nodes_created=0)

        assert to_rows(result).rows == [
            {"node": {"labels": ["Alpha", "Zebra"], "properties": {"id": "n1"}}}
        ]


class TestFalkorDBSchemaCommands:
    """Timeout behavior of runner-managed schema commands."""

    async def test_unique_constraint_creation_uses_the_query_timeout(self):
        """A stalled SDK command becomes a typed query timeout."""

        class HangingGraph:
            async def create_node_unique_constraint(self, label, prop):
                await asyncio.Event().wait()

        async def graph_factory(name):
            return HangingGraph()

        runner = QueryRunner(
            graph_factory,
            ResilienceSettings(max_attempts=1, query_timeout_ms=5),
        )

        with pytest.raises(QueryTimeoutError):
            await runner.create_unique_constraint("graph", "Entity", "id")

    async def test_rejects_unsafe_schema_names_before_contacting_the_server(self):
        """Invalid schema identifiers are rejected before graph creation."""
        runner = Mock(run=AsyncMock())
        manager = SchemaManager(runner)

        with pytest.raises(ValidationError):
            await manager.provision("graph", SchemaPart("unsafe", labels=("Bad`",)))

        runner.run.assert_not_awaited()
