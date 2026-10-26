"""Confirmation-token signing and target digests."""

import base64
import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Sequence

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.service.access.errors import InvalidKeySecretError
from ontogr_mem.service.errors import ConfirmationRequiredError


CONFIRM_TTL_SECONDS = 600
_MAC_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")


def target_digest(selector_kind: str, episode_ids: Sequence[str]) -> str:
    """Hash a canonical selector kind and resolved Episode id set.

    Args:
        selector_kind: Name of the selector field.
        episode_ids: Episode ids resolved by the dry run.

    Returns:
        A SHA-256 hex digest that changes when the selector kind or target set
        changes.
    """
    payload = json.dumps(
        {"episodes": sorted(set(episode_ids)), "selector": selector_kind},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ConfirmTokens:
    """Issue and verify short-lived tokens bound to a forget request.

    Args:
        secret: HMAC key bytes, or None to create a process-local random key.
        clock: Time source used for token issue and expiry checks.
    """

    def __init__(self, secret: bytes | None, clock: Clock) -> None:
        """Bind the token signer to a secret and clock."""
        if secret is not None and len(secret) < 16:
            raise InvalidKeySecretError("forget confirmation secret is too short")
        self._secret = secret or secrets.token_bytes(32)
        self._clock = clock

    def issue(self, caller: str, tenant: str, target: str, mode: str) -> str:
        """Sign a request context and return its expiry-prefixed token."""
        expires_at = int(self._clock.now().timestamp()) + CONFIRM_TTL_SECONDS
        mac = self._mac(caller, tenant, target, mode, expires_at)
        return f"{expires_at}.{mac}"

    def verify(
        self,
        token: str,
        caller: str,
        tenant: str,
        target: str,
        mode: str,
    ) -> None:
        """Reject a malformed, expired, or context-mismatched token.

        Args:
            token: Token returned by `issue`.
            caller: Current principal id.
            tenant: Tenant id for the operation.
            target: Digest of the currently resolved target.
            mode: Forget mode for the operation.

        Raises:
            ConfirmationRequiredError: If the token is not valid for the
                current context and time.
        """
        try:
            expiry_text, mac = token.split(".", 1)
            if not expiry_text.isascii() or not expiry_text.isdigit():
                raise ValueError
            if not _MAC_PATTERN.fullmatch(mac):
                raise ValueError
            expires_at = int(expiry_text)
            expected = self._mac(caller, tenant, target, mode, expires_at)
            valid_mac = hmac.compare_digest(mac, expected)
        except (AttributeError, TypeError, ValueError):
            valid_mac = False
            expires_at = 0

        if not valid_mac or int(self._clock.now().timestamp()) >= expires_at:
            raise ConfirmationRequiredError(
                "confirmation token is invalid or expired; request a new preview"
            )

    @staticmethod
    def expiry(token: str) -> int:
        """Return the Unix expiry second encoded in a well-formed token.

        Args:
            token: Token returned by `issue`.

        Raises:
            ConfirmationRequiredError: If the token has no integer expiry.
        """
        try:
            expiry_text, mac = token.split(".", 1)
            if not expiry_text.isascii() or not expiry_text.isdigit():
                raise ValueError
            if not _MAC_PATTERN.fullmatch(mac):
                raise ValueError
            return int(expiry_text)
        except (AttributeError, TypeError, ValueError):
            raise ConfirmationRequiredError("confirmation token is malformed") from None

    def _mac(
        self, caller: str, tenant: str, target: str, mode: str, expiry: int
    ) -> str:
        """Return the MAC for a context encoded without delimiter ambiguity."""
        payload = json.dumps(
            [caller, tenant, target, mode, expiry],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
