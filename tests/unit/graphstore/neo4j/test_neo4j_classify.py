"""Tests for Neo4j error classification."""

import pytest
from neo4j.exceptions import (
    ClientError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)

from ontogr_mem.graphstore.executor import ErrorKind
from ontogr_mem.graphstore.neo4j.runner import classify


def server_error(cls, code: str, message: str = "message"):
    """Build a driver error that carries a server code."""
    return type(cls.__name__, (cls,), {"code": code})(message)


class TestClassify:
    """Behavior of classify."""

    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (ConnectionRefusedError(), ErrorKind.PRE_EXECUTION),
            (
                ServiceUnavailable("Couldn't connect to localhost:1"),
                ErrorKind.PRE_EXECUTION,
            ),
            (SessionExpired("Connection closed"), ErrorKind.OUTCOME_UNKNOWN),
            (
                server_error(
                    ClientError, "Neo.ClientError.Schema.ConstraintValidationFailed"
                ),
                ErrorKind.CONFLICT,
            ),
            (
                server_error(
                    ClientError,
                    "Neo.ClientError.Schema.ConstraintValidationFailed",
                    "because it still has relationships",
                ),
                ErrorKind.CONFLICT,
            ),
            (
                server_error(
                    ClientError, "Neo.ClientError.Transaction.TransactionTimedOut"
                ),
                ErrorKind.TIMEOUT,
            ),
            (
                server_error(
                    TransientError, "Neo.TransientError.Transaction.DeadlockDetected"
                ),
                ErrorKind.SERVER_TRANSIENT,
            ),
            (
                server_error(ClientError, "Neo.ClientError.Statement.SyntaxError"),
                ErrorKind.REJECTED,
            ),
            (ValueError("not the server"), None),
        ],
    )
    def test_maps_exception_to_kind(self, error, kind):
        """The exception class and server code decide the retry policy."""
        assert classify(error) == kind
