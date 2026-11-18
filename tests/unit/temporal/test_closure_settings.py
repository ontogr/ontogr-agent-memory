"""Check the environment-backed settings for temporal closure calls."""

import pytest
from pydantic import ValidationError

from ontogr_mem.temporal.settings import ClosureSettings


class TestClosureSettings:
    """Closure settings use stable defaults and the MEMORY_CLOSURE_ prefix."""

    def test_defaults(self, monkeypatch):
        """Use enabled closure and the documented candidate and prompt defaults."""
        monkeypatch.delenv("MEMORY_CLOSURE_ENABLED", raising=False)
        monkeypatch.delenv("MEMORY_CLOSURE_MAX_CANDIDATES", raising=False)
        monkeypatch.delenv("MEMORY_CLOSURE_PROMPT_VERSION", raising=False)
        settings = ClosureSettings(_env_file=None)

        assert settings.enabled is True
        assert settings.max_candidates == 20
        assert settings.prompt_version == "closure-1"

    def test_reads_environment_overrides(self, monkeypatch):
        """Load values from variables with the closure-specific prefix."""
        monkeypatch.setenv("MEMORY_CLOSURE_ENABLED", "false")
        monkeypatch.setenv("MEMORY_CLOSURE_MAX_CANDIDATES", "7")
        monkeypatch.setenv("MEMORY_CLOSURE_PROMPT_VERSION", "closure-2")

        settings = ClosureSettings(_env_file=None)

        assert settings.enabled is False
        assert settings.max_candidates == 7
        assert settings.prompt_version == "closure-2"

    def test_rejects_a_candidate_limit_below_one(self):
        """Require at least one candidate per model call."""
        with pytest.raises(ValidationError):
            ClosureSettings(max_candidates=0, _env_file=None)
