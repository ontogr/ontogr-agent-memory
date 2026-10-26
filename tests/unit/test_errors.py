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
        assert isinstance(MissingExtraError("falkordb"), AgentMemoryError)

    def test_message_names_the_install_command(self):
        """The message tells the user how to install the extra."""
        error = MissingExtraError("falkordb")

        assert 'pip install "agent-memory[falkordb]"' in str(error)
        assert error.extra == "falkordb"


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
