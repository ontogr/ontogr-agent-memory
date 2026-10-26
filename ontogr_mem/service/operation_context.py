"""Operation rows, tracing, and request context binding."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from ontogr_mem.observability import SpanName, Tracer, open_span
from ontogr_mem.relational_store import (
    OperationKind,
    OperationOutcome,
    OperationStore,
)
from ontogr_mem.request_context import RequestContext, bind_context


_OPERATION_SPANS = {
    OperationKind.WRITE: SpanName.WRITE,
    OperationKind.RETRIEVE: SpanName.RETRIEVE,
    OperationKind.INVALIDATE: SpanName.FORGET,
    OperationKind.ERASE: SpanName.ERASE,
    OperationKind.EXPORT: SpanName.EXPORT,
    OperationKind.HISTORY: SpanName.HISTORY,
}

log = logging.getLogger(__name__)


class OperationRecorder:
    """Bind request contexts and persist their measured operation outcomes."""

    def __init__(self, operations: OperationStore, tracer: Tracer) -> None:
        """Bind operation persistence and tracing."""
        self._operations = operations
        self._tracer = tracer

    @asynccontextmanager
    async def track(
        self,
        kind: OperationKind,
        principal_id: str,
        tenant: str | None,
        *,
        parent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> AsyncGenerator[RequestContext, None]:
        """Create an operation row, bind its context, and record its outcome."""
        context = RequestContext(
            principal_id=principal_id,
            tenant=tenant,
            session_id=session_id,
            trace_id=trace_id if trace_id is not None else "",
        )
        context.operation_id = await self._operations.start(
            kind=kind,
            principal_id=principal_id,
            tenant=tenant,
            session_id=session_id,
            parent_id=parent_id,
            trace_id=context.trace_id,
        )
        span_name = _OPERATION_SPANS[kind]
        with (
            open_span(
                self._tracer,
                span_name,
                tenant=tenant,
                principal=principal_id,
                trace_id=context.trace_id,
            ) as span,
            bind_context(context),
        ):
            try:
                yield context
            except BaseException as exc:
                span.set("outcome", "failed")
                await self._operations.finish(
                    context.operation_id,
                    outcome=OperationOutcome.FAILED,
                    usage=context.usage.snapshot(),
                    error=exc,
                )
                log.info(
                    "op=%s tenant=%s trace=%s outcome=failed",
                    span_name.value,
                    tenant,
                    context.trace_id,
                )
                raise
            else:
                span.set("outcome", "ok")
                await self._operations.finish(
                    context.operation_id,
                    outcome=OperationOutcome.OK,
                    usage=context.usage.snapshot(),
                )
                log.info(
                    "op=%s tenant=%s trace=%s outcome=ok",
                    span_name.value,
                    tenant,
                    context.trace_id,
                )
