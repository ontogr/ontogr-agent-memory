"""Environment-backed settings for local access principals."""

import pytest
from pydantic import ValidationError

from ontogr_mem.service.access.settings import StdioSettings
from ontogr_mem.service.principal import Grant, Principal


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove access settings a developer environment may have set."""
    for name in (
        "MEMORY_PRINCIPAL",
        "MEMORY_TENANT",
        "MEMORY_GRANTS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestStdioSettings:
    """Stdio settings create a principal from the shared environment prefix."""

    def test_defaults_create_the_local_stdio_principal(self):
        """Defaults grant read, write, and forget on the default tenant."""
        settings = StdioSettings(_env_file=None)

        assert settings.to_principal() == Principal(
            id="local",
            grants={"default": frozenset({Grant.READ, Grant.WRITE, Grant.FORGET})},
            default_tenant="default",
        )

    def test_environment_values_build_a_stdio_principal(self, monkeypatch):
        """The principal, tenant, and grant list use MEMORY_ variables."""
        monkeypatch.setenv("MEMORY_PRINCIPAL", "bot")
        monkeypatch.setenv("MEMORY_TENANT", "acme")
        monkeypatch.setenv("MEMORY_GRANTS", "read,admin")

        principal = StdioSettings(_env_file=None).to_principal()

        assert principal == Principal(
            id="bot",
            grants={"acme": frozenset({Grant.READ, Grant.ADMIN})},
            default_tenant="acme",
        )

    @pytest.mark.parametrize(
        ("name", "value"),
        [("MEMORY_GRANTS", "read,fly"), ("MEMORY_TENANT", "bad id")],
    )
    def test_rejects_invalid_tenant_or_grant(self, monkeypatch, name, value):
        """Unknown grants and invalid tenant ids fail settings validation."""
        monkeypatch.setenv(name, value)

        with pytest.raises(ValidationError):
            StdioSettings(_env_file=None)
