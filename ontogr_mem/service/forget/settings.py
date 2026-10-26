"""Environment settings for forget confirmation tokens."""

from typing import Any

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.data_models.clock import Clock, SystemClock
from ontogr_mem.service.access.errors import InvalidKeySecretError
from ontogr_mem.service.forget.confirm import ConfirmTokens


class ForgetSettings(BaseSettings):
    """HMAC secret used to sign forget confirmation tokens.

    Environment variables use the ``MEMORY_FORGET_`` prefix.

    Attributes:
        confirm_secret: Shared token secret. Env:
            ``MEMORY_FORGET_CONFIRM_SECRET``. When absent, tokens use a random
            process-local secret and do not verify across restarts.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_FORGET_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    confirm_secret: SecretStr | None = None

    def model_post_init(self, context: Any) -> None:
        """Reject configured secrets shorter than 16 UTF-8 bytes."""
        if (
            self.confirm_secret is not None
            and len(self.confirm_secret.get_secret_value().encode("utf-8")) < 16
        ):
            raise InvalidKeySecretError(
                "forget confirmation secret must contain at least 16 bytes"
            )

    def tokens(self, clock: Clock | None = None) -> ConfirmTokens:
        """Build the signer with an injected or system clock."""
        secret = (
            self.confirm_secret.get_secret_value().encode("utf-8")
            if self.confirm_secret is not None
            else None
        )
        return ConfirmTokens(secret, clock if clock is not None else SystemClock())
