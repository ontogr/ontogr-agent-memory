"""Per-task request data and model usage accounting."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.observability.ids import accept_trace_id, new_trace_id


@dataclass
class Usage:
    """Accumulate token counts and model costs for one operation."""

    tokens_in: int = 0
    tokens_out: int = 0
    priced_cost_usd: float = 0.0
    unpriced_calls: int = 0

    def add(self, tokens_in: int, tokens_out: int, cost_usd: float | None) -> None:
        """Add one model call to the request total.

        Raises:
            ValueError: If either token count is negative.
        """
        if tokens_in < 0 or tokens_out < 0:
            raise ValueError("usage token counts must not be negative")
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.priced_cost_usd += cost_usd

    def snapshot(self) -> OperationUsage:
        """Return a detached operation usage value."""
        return OperationUsage(
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=None if self.unpriced_calls else self.priced_cost_usd,
        )


@dataclass
class RequestContext:
    """Values and mutable counters shared by one request's async tasks."""

    principal_id: str
    tenant: str | None
    session_id: str | None = None
    trace_id: str = field(default_factory=new_trace_id)
    operation_id: str = ""
    latency_budget_s: float | None = None
    started_monotonic: float = field(default_factory=time.monotonic)
    usage: Usage = field(default_factory=Usage)
    cache_tags: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate the caller and any Tenant before binding the context."""
        if not self.principal_id:
            raise ValueError("principal id must not be empty")
        if self.tenant is not None:
            validate_tenant_id(self.tenant)
        self.trace_id = accept_trace_id(self.trace_id)

    def remaining_seconds(self) -> float | None:
        """Return the remaining latency budget, clamped to zero."""
        if self.latency_budget_s is None:
            return None
        elapsed = time.monotonic() - self.started_monotonic
        return max(0.0, self.latency_budget_s - elapsed)


_CURRENT: ContextVar[RequestContext | None] = ContextVar(
    "request_context", default=None
)


def current_context() -> RequestContext | None:
    """Return the context bound to the current async task, if any."""
    return _CURRENT.get()


@contextmanager
def bind_context(context: RequestContext) -> Generator[RequestContext]:
    """Bind a request context and restore the previous value on exit."""
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def record_usage(tokens_in: int, tokens_out: int, cost_usd: float | None) -> None:
    """Add usage to the current request, or do nothing when unbound."""
    context = current_context()
    if context is not None:
        context.usage.add(tokens_in, tokens_out, cost_usd)
