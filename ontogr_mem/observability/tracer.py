"""Span protocols and tracer implementations for in-process tracing."""

import time
from collections import deque
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from itertools import count
from threading import Lock
from types import MappingProxyType
from typing import Protocol

from ontogr_mem.observability.redact import summarize_exception


class SpanName(StrEnum):
    """Names reserved for memory, authorization and model operations."""

    WRITE = "memory.write"
    RETRIEVE = "memory.retrieve"
    ERASE = "memory.erase"
    EXPORT = "memory.export"
    FORGET = "memory.forget"
    HISTORY = "memory.history"
    LLM_COMPLETE = "llm.complete"
    LLM_EMBED = "llm.embed"
    AUTH_CHECK = "auth.check"


class Span(Protocol):
    """A timed operation that accepts safe attributes and an error summary."""

    def set(self, name: str, value: object) -> None:
        """Set or replace an attribute on the span."""
        ...

    def error(self, error: BaseException) -> None:
        """Record only a redacted, capped summary of a failed operation."""
        ...


class Tracer(Protocol):
    """Open named spans around operations."""

    def span(
        self, name: SpanName | str, **attributes: object
    ) -> AbstractContextManager[Span]:
        """Return a context manager that records one span."""
        ...


@dataclass(frozen=True, slots=True)
class FinishedSpan:
    """A completed in-memory span and its safe attributes.

    Attributes:
        name: Span name as a string.
        span_id: Unique id within its tracer instance.
        parent_id: Active parent id, if one existed.
        started: Monotonic start time in seconds.
        duration_ms: Elapsed time in milliseconds.
        attributes: Read-only snapshot of span attributes.
        error: Redacted one-line exception summary, if the span failed.
    """

    name: str
    span_id: int
    parent_id: int | None
    started: float
    duration_ms: float
    attributes: Mapping[str, object]
    error: str | None


class _NoOpSpan:
    """Accept span updates without retaining them."""

    def set(self, name: str, value: object) -> None:
        """Discard an attribute update."""

    def error(self, error: BaseException) -> None:
        """Discard an error summary."""


NOOP_SPAN = _NoOpSpan()
_NOOP_CONTEXT_MANAGER = nullcontext(NOOP_SPAN)


class NoOpTracer:
    """Provide a no-op implementation without per-span allocations."""

    def span(
        self, name: SpanName | str, **attributes: object
    ) -> AbstractContextManager[Span]:
        """Return the shared context manager for every requested span."""
        return _NOOP_CONTEXT_MANAGER


class _InMemorySpan:
    """Mutable state for one open in-memory span."""

    def __init__(
        self,
        span_id: int,
        attributes: dict[str, object],
    ) -> None:
        self.span_id = span_id
        self.attributes = attributes
        self.error_summary: str | None = None

    def set(self, name: str, value: object) -> None:
        """Store or replace one attribute."""
        self.attributes[name] = value

    def error(self, error: BaseException) -> None:
        """Store a redacted one-line summary of an exception."""
        self.error_summary = summarize_exception(error)


class InMemoryTracer:
    """Keep a bounded, thread-safe buffer of completed spans."""

    def __init__(self, keep: int = 1000) -> None:
        """Create a tracer that retains at most ``keep`` completed spans."""
        self._spans: deque[FinishedSpan] = deque(maxlen=keep)
        self._lock = Lock()
        self._span_ids = count(1)
        self._parent_id: ContextVar[int | None] = ContextVar(
            f"in_memory_tracer_parent_{id(self)}", default=None
        )

    def span(
        self, name: SpanName | str, **attributes: object
    ) -> AbstractContextManager[Span]:
        """Return a context manager that records one completed span."""
        return self._record_span(name, attributes)

    def finished(self) -> tuple[FinishedSpan, ...]:
        """Return a stable snapshot of the retained completed spans."""
        with self._lock:
            return tuple(self._spans)

    @contextmanager
    def _record_span(
        self, name: SpanName | str, attributes: dict[str, object]
    ) -> Generator[Span, None, None]:
        """Record span state and restore its parent context on every exit."""
        with self._lock:
            span_id = next(self._span_ids)
        parent_id = self._parent_id.get()
        started = time.monotonic()
        span = _InMemorySpan(span_id, dict(attributes))
        parent_token: Token[int | None] = self._parent_id.set(span_id)
        try:
            yield span
        except BaseException as exc:
            span.error(exc)
            raise
        finally:
            duration_ms = (time.monotonic() - started) * 1000
            self._parent_id.reset(parent_token)
            finished = FinishedSpan(
                name=str(name),
                span_id=span_id,
                parent_id=parent_id,
                started=started,
                duration_ms=duration_ms,
                attributes=MappingProxyType(dict(span.attributes)),
                error=span.error_summary,
            )
            with self._lock:
                self._spans.append(finished)
