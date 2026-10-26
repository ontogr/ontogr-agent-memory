"""Tests for FalkorDB error classification."""

import pytest
from redis.exceptions import BusyLoadingError, ResponseError, TryAgainError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from ontogr_mem.graphstore.executor import ErrorKind
from ontogr_mem.graphstore.falkordb.runner import classify


class TestClassify:
    """Behavior of classify."""

    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (ConnectionRefusedError(), ErrorKind.PRE_EXECUTION),
            (RedisConnectionError("No connection available."), ErrorKind.PRE_EXECUTION),
            (
                RedisConnectionError(
                    "Error 111 connecting to h:1. Connect call failed"
                ),
                ErrorKind.PRE_EXECUTION,
            ),
            (
                RedisConnectionError("Connection reset by peer"),
                ErrorKind.OUTCOME_UNKNOWN,
            ),
            (
                RedisTimeoutError("Timeout reading from socket"),
                ErrorKind.OUTCOME_UNKNOWN,
            ),
            (OSError("broken pipe"), ErrorKind.OUTCOME_UNKNOWN),
            (BusyLoadingError("loading"), ErrorKind.SERVER_TRANSIENT),
            (TryAgainError("try again"), ErrorKind.SERVER_TRANSIENT),
        ],
    )
    def test_uses_the_exception_class(self, error, kind):
        """Connection and transient errors are told apart by class first."""
        assert classify(error) is kind

    @pytest.mark.parametrize(
        ("message", "kind"),
        [
            ("LOADING Redis is loading the dataset", ErrorKind.SERVER_TRANSIENT),
            ("BUSY Redis is busy running a script", ErrorKind.SERVER_TRANSIENT),
            ("TRYAGAIN try again later", ErrorKind.SERVER_TRANSIENT),
            ("Max pending queries exceeded", ErrorKind.SERVER_TRANSIENT),
            ("Query timed out", ErrorKind.TIMEOUT),
            ("unique constraint violation on Entity.id", ErrorKind.CONFLICT),
            ("Cannot merge node using null property value", ErrorKind.REJECTED),
            ("", ErrorKind.REJECTED),
        ],
    )
    def test_reads_the_message_only_for_bare_response_errors(self, message, kind):
        """Each documented message fragment maps to one kind."""
        assert classify(ResponseError(message)) is kind

    def test_ignores_errors_that_are_not_the_servers(self):
        """A programming error is not classified, so it propagates."""
        assert classify(KeyError("x")) is None
