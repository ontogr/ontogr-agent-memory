"""Bearer token checks for the MCP server, run in process without sockets."""

import httpx2
import pytest
from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken

from ontogr_mem.mcp_server.auth import ApiKeyVerifier


PING_CALL = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "ping", "arguments": {}},
}

MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


def _build_server(auth: ApiKeyVerifier | None) -> FastMCP:
    """Build a server with one ping tool that replies with pong."""
    server = FastMCP("test", auth=auth)

    @server.tool
    async def ping() -> str:
        """Reply with a fixed word."""
        return "pong"

    return server


async def _call_ping(server: FastMCP, headers: dict[str, str]) -> httpx2.Response:
    """POST a ping tool call to the /mcp endpoint of the server in process."""
    app = server.http_app(path="/mcp", stateless_http=True)
    transport = httpx2.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=transport, base_url="http://test", trust_env=False
        ) as client,
    ):
        return await client.post(
            "/mcp", json=PING_CALL, headers={**MCP_HEADERS, **headers}
        )


class TestApiKeyVerifier:
    """Bearer token verification against the configured API key."""

    def test_rejects_an_empty_api_key(self):
        """An empty key is refused, because it would accept an empty token."""
        with pytest.raises(ValueError, match="api_key must not be empty"):
            ApiKeyVerifier("")

    async def test_returns_none_for_an_empty_token(self):
        """An empty bearer token is never accepted."""
        verifier = ApiKeyVerifier("sentinel-key")

        assert await verifier.verify_token("") is None

    @pytest.mark.parametrize("token", ["nope", "sentinel-kéy"])
    async def test_returns_none_for_a_wrong_token(self, token):
        """Any token other than the configured key is rejected, ASCII or not."""
        verifier = ApiKeyVerifier("sentinel-key")

        assert await verifier.verify_token(token) is None

    async def test_returns_local_identity_for_the_configured_key(self):
        """The configured key maps to the fixed local client identity."""
        verifier = ApiKeyVerifier("sentinel-key")

        access_token = await verifier.verify_token("sentinel-key")

        assert isinstance(access_token, AccessToken)
        assert access_token.token == "sentinel-key"
        assert access_token.client_id == "local"

    async def test_http_request_without_token_is_unauthorized(self):
        """A request with no Authorization header gets 401."""
        server = _build_server(ApiKeyVerifier("sentinel-key"))

        response = await _call_ping(server, {})

        assert response.status_code == 401

    async def test_http_request_with_wrong_token_is_unauthorized(self):
        """A request with a bearer token that does not match gets 401."""
        server = _build_server(ApiKeyVerifier("sentinel-key"))

        response = await _call_ping(server, {"authorization": "Bearer nope"})

        assert response.status_code == 401

    async def test_http_request_with_configured_token_runs_the_tool(self):
        """The configured bearer token lets the tool call through."""
        server = _build_server(ApiKeyVerifier("sentinel-key"))

        response = await _call_ping(server, {"authorization": "Bearer sentinel-key"})

        assert response.status_code == 200
        assert "pong" in response.text

    async def test_server_without_auth_accepts_anonymous_requests(self):
        """A server built without an auth provider needs no token."""
        server = _build_server(None)

        response = await _call_ping(server, {})

        assert response.status_code == 200
        assert "pong" in response.text
