"""Error handling of the MCP tools: typed errors, hidden internals, safe receipts."""

import json

import pytest
from fastmcp import Client, FastMCP

from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.mcp_server.server import build_server
from ontogr_mem.mcp_server.settings import ALL_TOOLS, McpSettings
from ontogr_mem.service.contracts import ErrorInfo, WriteReceipt
from ontogr_mem.service.errors import (
    ErrorCode,
    InvalidArgumentError,
    ObjectNotFoundError,
)
from tests.support.mcp_memory import (
    TENANT,
    FakeMemoryService,
    build_memory,
    job_view,
)


SENTINEL = "sentinel-secret-123"
FAILED_MESSAGE = "The memory operation failed."


def _server(service: FakeMemoryService) -> FastMCP:
    """Build a server over the fake service that skips the skills extension."""
    settings = McpSettings(
        _env_file=None, tools=ALL_TOOLS, api_key=None, skills_extension=False
    )
    return build_server(build_memory(service), settings=settings)


def _text(result) -> str:
    """Join the text blocks of a tool result."""
    return "".join(block.text for block in result.content)


class TestToolErrors:
    """Failures reach the client as a typed error or a masked receipt."""

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (
                InvalidArgumentError("bad query", "Shorten the query."),
                "invalid_argument",
            ),
            (ObjectNotFoundError("no such thing"), "not_found"),
        ],
    )
    async def test_service_error_becomes_a_coded_error_result(self, error, code):
        """A service error reaches the client as code, message and hint."""
        service = FakeMemoryService()
        service.failure = error
        server = _server(service)

        async with Client(server) as client:
            result = await client.call_tool(
                "search_memory", {"query": "x"}, raise_on_error=False
            )

        assert result.is_error
        payload = json.loads(_text(result))
        assert set(payload) == {"code", "message", "hint"}
        assert payload == {
            "code": code,
            "message": error.message,
            "hint": error.hint,
        }

    async def test_unexpected_exception_hides_its_message(self):
        """An unexpected exception gives an error result without its message."""
        service = FakeMemoryService()
        service.failure = RuntimeError(f"connection to {SENTINEL} refused")
        server = _server(service)

        async with Client(server) as client:
            result = await client.call_tool(
                "search_memory", {"query": "x"}, raise_on_error=False
            )

        assert result.is_error
        assert SENTINEL not in repr(result)

    async def test_internal_error_in_a_write_receipt_is_replaced(self):
        """An internal error inside a returned receipt loses its details."""
        service = FakeMemoryService()
        service.receipt = WriteReceipt(
            job_id="job-9",
            status=JobState.FAILED,
            tenant=TENANT,
            error=ErrorInfo(
                code=ErrorCode.INTERNAL,
                message=f"driver failed for {SENTINEL}",
                hint=f"check {SENTINEL}",
            ),
        )
        server = _server(service)

        async with Client(server) as client:
            result = await client.call_tool(
                "forget_memory", {"fact_id": "fact-1"}, raise_on_error=False
            )

        assert not result.is_error
        assert result.structured_content is not None
        assert result.structured_content["job_id"] == "job-9"
        assert result.structured_content["error"]["code"] == "internal"
        assert result.structured_content["error"]["message"] == FAILED_MESSAGE
        assert SENTINEL not in repr(result)

    async def test_internal_error_in_a_job_view_is_replaced(self):
        """An internal error inside a job view loses its details."""
        service = FakeMemoryService()
        service.job = job_view(
            JobState.FAILED,
            ErrorInfo(
                code=ErrorCode.INTERNAL,
                message=f"driver failed for {SENTINEL}",
                hint=f"check {SENTINEL}",
            ),
        )
        server = _server(service)

        async with Client(server) as client:
            result = await client.call_tool(
                "get_job", {"job_id": "job-9"}, raise_on_error=False
            )

        assert not result.is_error
        assert result.structured_content is not None
        assert result.structured_content["error"]["code"] == "internal"
        assert result.structured_content["error"]["message"] == FAILED_MESSAGE
        assert SENTINEL not in repr(result)

    async def test_non_internal_error_in_a_job_view_passes_through(self):
        """An error with a caller-actionable code is returned unchanged."""
        service = FakeMemoryService()
        service.job = job_view(
            JobState.FAILED,
            ErrorInfo(
                code=ErrorCode.LLM_UNAVAILABLE,
                message="No extraction model is configured.",
                hint="Configure a model.",
            ),
        )
        server = _server(service)

        async with Client(server) as client:
            result = await client.call_tool(
                "get_job", {"job_id": "job-9"}, raise_on_error=False
            )

        assert not result.is_error
        assert result.structured_content is not None
        assert result.structured_content["error"] == {
            "code": "llm_unavailable",
            "message": "No extraction model is configured.",
            "hint": "Configure a model.",
        }
