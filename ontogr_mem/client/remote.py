"""MCP-backed implementation of the Memory service boundary."""

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ontogr_mem._extras import import_extra
from ontogr_mem.service.contracts import (
    AddMemory,
    EntityPage,
    EntityView,
    GetEntity,
    GetOntology,
    InvalidateMemory,
    JobView,
    ListEntities,
    ListStaleEpisodes,
    MemoryContext,
    OntologyView,
    SearchMemory,
    StaleEpisodePage,
    WriteReceipt,
)
from ontogr_mem.service.errors import (
    ConfirmationRequiredError,
    ConflictError,
    ErrorCode,
    InternalError,
    InvalidArgumentError,
    LLMUnavailableError,
    ObjectNotFoundError,
    PayloadTooLargeError,
    RateLimitedError,
    ServiceError,
)
from ontogr_mem.service.principal import Principal


_ERROR_TYPES: dict[ErrorCode, type[ServiceError]] = {
    ErrorCode.INVALID_ARGUMENT: InvalidArgumentError,
    ErrorCode.NOT_FOUND: ObjectNotFoundError,
    ErrorCode.CONFLICT: ConflictError,
    ErrorCode.LLM_UNAVAILABLE: LLMUnavailableError,
    ErrorCode.RATE_LIMITED: RateLimitedError,
    ErrorCode.PAYLOAD_TOO_LARGE: PayloadTooLargeError,
    ErrorCode.CONFIRMATION_REQUIRED: ConfirmationRequiredError,
    ErrorCode.INTERNAL: InternalError,
}


def _mcp_url(url: str) -> str:
    """Return the streamable HTTP endpoint for a configured server URL."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if path.endswith("/mcp"):
        return urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
        )
    return urlunsplit(
        (parts.scheme, parts.netloc, f"{path}/mcp", parts.query, parts.fragment)
    )


class RemoteService:
    """Send tenant-scoped memory operations to a remote MCP server."""

    def __init__(self, client: Any) -> None:
        """Bind an open FastMCP client."""
        self._client = client
        self._job_ids: set[str] = set()

    @classmethod
    async def open(cls, *, url: str, token: str | None, tenant: str) -> "RemoteService":
        """Open one authenticated MCP connection for a tenant."""
        import_extra("fastmcp", "mcp")
        from fastmcp import Client  # noqa: PLC0415
        from fastmcp.client.transports import StreamableHttpTransport  # noqa: PLC0415

        transport = StreamableHttpTransport(
            _mcp_url(url),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Memory-Tenant": tenant,
            },
        )
        client = Client(transport)
        await client.__aenter__()
        return cls(client)

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        try:
            result = await self._client.call_tool(name, arguments)
        except ToolError as error:
            try:
                payload = json.loads(str(error))
                code = ErrorCode(payload["code"])
                error_type = _ERROR_TYPES[code]
                raise error_type(payload["message"], payload.get("hint", "")) from error
            except (ValueError, KeyError, TypeError):
                raise InternalError("The memory operation failed.") from error
        return result.structured_content

    async def add_memory(
        self, principal: Principal, request: AddMemory
    ) -> WriteReceipt:
        """Add memory through the remote server."""
        del principal
        arguments = request.model_dump(
            mode="json", exclude={"tenant", "kind", "facts"}, exclude_none=True
        )
        if request.kind.value == "sourced_facts":
            arguments = {
                "source_text": request.content,
                "facts": [
                    fact.model_dump(mode="json", exclude_none=True)
                    for fact in request.facts or ()
                ],
                "role": request.role.value,
                "idempotency_key": request.idempotency_key,
                "wait_seconds": request.wait_seconds,
            }
            tool = "write_facts"
        else:
            tool = "add_memory"
        response = await self._call(tool, arguments)
        receipt = WriteReceipt.model_validate(response)
        self._job_ids.add(receipt.job_id)
        return receipt

    async def search_memory(
        self, principal: Principal, request: SearchMemory
    ) -> MemoryContext:
        """Search remote memory for relevant context."""
        del principal
        return MemoryContext.model_validate(
            await self._call(
                "search_memory",
                {
                    "query": request.query,
                    "as_of": request.valid_at,
                    "include_history": request.include_history,
                    "limit": request.limit,
                    "budget_tokens": request.budget_tokens,
                },
            )
        )

    async def get_entity(self, principal: Principal, request: GetEntity) -> EntityView:
        """Return one entity view from the remote server."""
        del principal
        return EntityView.model_validate(
            await self._call(
                "get_entity",
                {
                    "name": request.name,
                    "include_history": request.include_history,
                },
            )
        )

    async def list_entities(
        self, principal: Principal, request: ListEntities
    ) -> EntityPage:
        """List active entities through the remote server."""
        del principal
        return EntityPage.model_validate(
            await self._call(
                "list_entities",
                {
                    "entity_type": request.entity_type,
                    "cursor": request.cursor,
                    "limit": request.limit,
                },
            )
        )

    async def get_ontology(
        self, principal: Principal, request: GetOntology
    ) -> OntologyView:
        """Return the remote tenant ontology."""
        del principal
        return OntologyView.model_validate(
            await self._call(
                "get_ontology", {"include_deprecated": request.include_deprecated}
            )
        )

    async def stale_episodes(
        self, principal: Principal, request: ListStaleEpisodes
    ) -> StaleEpisodePage:
        """List stale episodes through the remote server."""
        del principal
        return StaleEpisodePage.model_validate(
            await self._call(
                "stale_episodes", {"cursor": request.cursor, "limit": request.limit}
            )
        )

    async def get_job(self, principal: Principal, job_id: str) -> JobView:
        """Return the status of one remote write job."""
        del principal
        return JobView.model_validate(await self._call("get_job", {"job_id": job_id}))

    async def invalidate(
        self, principal: Principal, request: InvalidateMemory
    ) -> WriteReceipt:
        """Invalidate remote memory and return its write receipt."""
        del principal
        receipt = WriteReceipt.model_validate(
            await self._call(
                "forget_memory",
                {
                    "fact_id": request.fact_id,
                    "reason": request.reason,
                },
            )
        )
        self._job_ids.add(receipt.job_id)
        return receipt

    async def forget_memory(self, principal: Principal, request: Any) -> Any:
        """Reject unsupported direct forget requests."""
        del principal, request
        raise InvalidArgumentError(
            "Connected Memory does not support forget.", "Use invalidate."
        )

    async def drain(self) -> None:
        """Wait for remote writes tracked by this connection to finish."""
        pending = set(self._job_ids)
        deadline = asyncio.get_running_loop().time() + 30
        while pending and asyncio.get_running_loop().time() < deadline:
            for job_id in tuple(pending):
                job = JobView.model_validate(
                    await self._call("get_job", {"job_id": job_id})
                )
                if job.status.value not in {"queued", "running"}:
                    pending.remove(job_id)
            if pending:
                await asyncio.sleep(0.5)

    async def aclose(self) -> None:
        """Close the MCP connection."""
        await self._client.__aexit__(None, None, None)
