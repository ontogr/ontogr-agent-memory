"""Adapter between the tracing protocol and a caller-configured OpenTelemetry SDK."""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, suppress
from typing import Any, Protocol

from ontogr_mem._extras import import_extra
from ontogr_mem.observability.redact import summarize_exception
from ontogr_mem.observability.tracer import NOOP_SPAN, Span, SpanName


class _OpenTelemetryTracerLike(Protocol):
    """Expose the OpenTelemetry tracer method used by this adapter."""

    def start_as_current_span(self, name: str) -> Any:
        """Return a context manager that activates a named span."""
        ...


class _OpenTelemetrySpan:
    """Expose the package span methods over an OpenTelemetry span object."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set(self, name: str, value: object) -> None:
        """Set a supported value, dropping ``None`` and containing SDK errors."""
        if value is None:
            return
        with suppress(Exception):
            self._span.set_attribute(name, value)

    def error(self, error: BaseException) -> None:
        """Set the redacted one-line exception summary as a span attribute."""
        self.set("error.summary", summarize_exception(error))


class OpenTelemetryTracer:
    """Adapt an OpenTelemetry tracer supplied by the application."""

    def __init__(self, otel_tracer: _OpenTelemetryTracerLike) -> None:
        """Wrap an object that implements ``start_as_current_span``."""
        self._tracer = otel_tracer

    def span(
        self, name: SpanName | str, **attributes: object
    ) -> AbstractContextManager[Span]:
        """Return an OpenTelemetry span context without propagating SDK errors."""
        return self._span(name, attributes)

    @contextmanager
    def _span(
        self, name: SpanName | str, attributes: dict[str, object]
    ) -> Generator[Span, None, None]:
        """Start and finish an SDK span while isolating SDK failures."""
        try:
            start_span = self._tracer.start_as_current_span
            context = start_span(name.value if isinstance(name, SpanName) else name)
            otel_span = context.__enter__()
        except Exception:
            yield NOOP_SPAN
            return

        span = _OpenTelemetrySpan(otel_span)
        for key, value in attributes.items():
            span.set(key, value)
        try:
            yield span
        except BaseException as body_error:
            span.error(body_error)
            with suppress(Exception):
                context.__exit__(type(body_error), body_error, body_error.__traceback__)
            raise
        else:
            with suppress(Exception):
                context.__exit__(None, None, None)


def opentelemetry_tracer(*, name: str = "ontogr_mem") -> OpenTelemetryTracer:
    """Build a tracer from the installed OpenTelemetry API.

    Raises:
        MissingExtraError: If the ``otel`` extra is not installed.
    """
    trace = import_extra("opentelemetry.trace", "otel")
    return OpenTelemetryTracer(trace.get_tracer(name))
