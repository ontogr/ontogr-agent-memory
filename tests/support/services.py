"""Helpers for tests that need external services."""

import pytest


def service_unreachable(name: str, address: str) -> None:
    """Fail the running test because a required service is not reachable.

    Args:
        name: Name of the service the test needs.
        address: Where the test tried to reach it.
    """
    pytest.fail(f"{name} is unreachable at {address}; start it with: make services-up")
