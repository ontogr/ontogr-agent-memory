"""Bearer token verification for the MCP http transport."""

import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier


class ApiKeyVerifier(TokenVerifier):
    """Accept the configured bearer token using a constant-time comparison."""

    def __init__(self, api_key: str) -> None:
        """Retain the UTF-8 bytes of a non-empty configured key.

        Raises:
            ValueError: If api_key is empty, because an empty bearer token would
                then match it.
        """
        if not api_key:
            raise ValueError("api_key must not be empty")
        super().__init__()
        self._key = api_key.encode("utf-8")

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return the fixed local identity only for the configured token."""
        if not hmac.compare_digest(token.encode("utf-8"), self._key):
            return None
        return AccessToken(token=token, client_id="local", scopes=[])
