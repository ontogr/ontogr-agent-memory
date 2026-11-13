"""Error handling shared by the MCP tools."""

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from fastmcp.exceptions import ToolError

from ontogr_mem.service.contracts import ErrorInfo, JobView, WriteReceipt
from ontogr_mem.service.errors import ErrorCode, ServiceError


P = ParamSpec("P")
T = TypeVar("T")


def _safe_error(error: ErrorInfo) -> ErrorInfo:
    """Replace the details of an internal error and keep any other error."""
    if error.code is ErrorCode.INTERNAL:
        return ErrorInfo(
            code=ErrorCode.INTERNAL,
            message="The memory operation failed.",
            hint="Inspect the server logs using the job id or trace id.",
        )
    return error


def guard_errors(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Wrap a tool function so that clients never see internal error details.

    A ServiceError raised by the function becomes a ToolError whose text is the
    error as JSON with code, message and hint. Other exceptions are left to the
    server, which masks their messages. An internal error carried by a returned
    WriteReceipt or JobView keeps its code but loses its message and hint.

    Args:
        fn: Async tool function to wrap.

    Returns:
        An async function with the signature and docstring of fn.
    """

    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            result = await fn(*args, **kwargs)
        except ServiceError as exc:
            error = _safe_error(ErrorInfo.from_error(exc))
            raise ToolError(error.model_dump_json()) from exc
        if isinstance(result, (WriteReceipt, JobView)) and result.error is not None:
            return result.model_copy(update={"error": _safe_error(result.error)})
        return result

    return wrapped
