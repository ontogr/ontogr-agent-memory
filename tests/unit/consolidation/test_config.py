"""Consolidation settings: defaults and the environment names they read."""

import pytest

from ontogr_mem.consolidation.config import ConsolidationSettings


def test_defaults_are_the_documented_starting_values() -> None:
    """The defaults match the documented starting values."""
    settings = ConsolidationSettings(_env_file=None)

    assert settings.runner.lease_ttl_s == 600.0
    assert settings.runner.max_reruns == 3
    assert settings.feedback.alpha == 0.1
    assert settings.prune.dry_run is True


def test_a_nested_field_is_read_from_its_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested setting is read from its double-underscore variable."""
    monkeypatch.setenv("MEMORY_CONSOLIDATION_RUNNER__LEASE_TTL_S", "30")

    settings = ConsolidationSettings(_env_file=None)

    assert settings.runner.lease_ttl_s == 30.0
