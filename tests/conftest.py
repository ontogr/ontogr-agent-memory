"""Adds test markers by directory."""

from collections.abc import Iterator

import pytest
from pydantic_ai import models


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Add markers to tests under tests/integration/.

    Each collected test under tests/integration/ gets the integration,
    enable_socket, and flaky markers. Test files must not add these markers
    directly.

    Args:
        config: The pytest run configuration.
        items: The collected test items.
    """
    for item in items:
        path = str(item.path).replace("\\", "/")
        if "tests/integration" in path:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.enable_socket)
            item.add_marker(
                pytest.mark.flaky(
                    reruns=3, reruns_delay=30, rerun_except=[AssertionError]
                )
            )


@pytest.fixture(autouse=True)
def prohibit_model_network() -> Iterator[None]:
    """Require tests to opt in explicitly before contacting a model provider."""
    with models.override_allow_model_requests(False):
        yield
