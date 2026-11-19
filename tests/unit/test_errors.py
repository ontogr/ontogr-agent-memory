"""Tests for the package error hierarchy."""

import pytest

from ontogr_mem.errors import AgentMemoryError, MissingExtraError, StoreError
from ontogr_mem.graphstore.errors import (
    AlreadyExistsError,
    CircuitOpenError,
    ConstraintViolationError,
    DatabaseUnavailableError,
    DeleteRefusedError,
    GraphStoreError,
    NotFoundError,
    QueryTimeoutError,
    SchemaDriftError,
    StoreQueryError,
    ValidationError,
)


GRAPH_STORE_ERRORS = (
    ValidationError,
    DatabaseUnavailableError,
    CircuitOpenError,
    StoreQueryError,
    QueryTimeoutError,
    ConstraintViolationError,
    AlreadyExistsError,
    NotFoundError,
    DeleteRefusedError,
    SchemaDriftError,
)


class TestMissingExtraError:
    """Behavior of the missing-extra error."""

    def test_is_an_ontogr_mem_error(self):
        """Callers can catch it through the package root error."""
        assert isinstance(MissingExtraError(("falkordb",)), AgentMemoryError)

    def test_message_names_the_install_command(self):
        """The message tells the user how to install the extra."""
        error = MissingExtraError(("falkordb",))

        assert 'uv add "ontogr-mem[falkordb]"' in str(error)
        assert error.extras == ("falkordb",)
        assert "pip" not in str(error)

    def test_message_explains_features_and_uvx_command(self):
        """The error connects missing extras to features and a runnable command."""
        error = MissingExtraError(
            ("mcp",),
            features=("MCP server",),
            needed=("falkordb", "mcp"),
            subcommand="mcp",
        )

        assert str(error).splitlines() == [
            "missing extras: mcp",
            "needed for: MCP server",
            'install with: uv add "ontogr-mem[mcp]"',
            'or run with: uvx --from "ontogr-mem[falkordb,mcp]" ontogr-mem mcp',
        ]


class TestStoreError:
    """Behavior of the shared store error base."""

    def test_is_an_ontogr_mem_error(self):
        """Callers can catch it through the package root error."""
        assert issubclass(StoreError, AgentMemoryError)

    def test_graph_store_error_is_a_store_error(self):
        """The graph store family carries the shared base."""
        assert issubclass(GraphStoreError, StoreError)

    @pytest.mark.parametrize("error_class", GRAPH_STORE_ERRORS)
    def test_graph_store_error_family_is_a_store_error(self, error_class):
        """Every graph store error carries the shared base."""
        assert issubclass(error_class, StoreError)

    @pytest.mark.parametrize("error_class", GRAPH_STORE_ERRORS)
    def test_graph_store_error_family_is_an_ontogr_mem_error(self, error_class):
        """Every graph store error stays under the package root error."""
        assert issubclass(error_class, AgentMemoryError)

    def test_validation_error_is_a_value_error(self):
        """A rejected input reads as a value error as well."""
        assert issubclass(ValidationError, ValueError)
