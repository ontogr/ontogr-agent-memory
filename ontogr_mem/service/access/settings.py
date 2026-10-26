"""Environment settings for the local stdio principal."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.service.principal import Grant, Principal


class StdioSettings(BaseSettings):
    """Environment settings for the local stdio principal.

    Environment variables use the ``MEMORY_`` prefix.

    Attributes:
        principal: Principal id. Env: ``MEMORY_PRINCIPAL``.
        tenant: Default tenant id. Env: ``MEMORY_TENANT``.
        grants: Comma-separated grant names. Env: ``MEMORY_GRANTS``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    principal: str = "local"
    tenant: str = "default"
    grants: str = "read,write,forget"

    @field_validator("tenant")
    @classmethod
    def validate_tenant(cls, value: str) -> str:
        """Reject tenant ids outside the service's tenant alphabet."""
        return validate_tenant_id(value)

    @field_validator("grants")
    @classmethod
    def validate_grants(cls, value: str) -> str:
        """Validate and normalize the comma-separated grant names."""
        names = [name.strip() for name in value.split(",")]
        if not names or any(not name for name in names):
            raise ValueError("grants must contain one or more grant names")
        try:
            return ",".join(Grant(name).value for name in names)
        except ValueError as error:
            raise ValueError("grants must be valid Grant values") from error

    def to_principal(self) -> Principal:
        """Build the stdio principal represented by these settings."""
        return Principal(
            id=self.principal,
            grants={
                self.tenant: frozenset(Grant(name) for name in self.grants.split(","))
            },
            default_tenant=self.tenant,
        )
