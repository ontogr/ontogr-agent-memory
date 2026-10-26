"""Guard span creation so tracing failures do not affect caller behavior."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from ontogr_mem.observability.redact import (
    redact_attributes,
    redact_value,
    summarize_exception,
)
from ontogr_mem.observability.tracer import NOOP_SPAN, Span, SpanName, Tracer


log = logging.getLogger(__name__)


class _GuardedSpan:
    """Contain failures from updates to an underlying span."""

    def __init__(self, span: Span) -> None:
        self._span = span

    def set(self, name: str, value: object) -> None:
        """Redact an attribute and ignore failures from the tracer."""
        try:
            self._span.set(name, redact_value(name, value))
        except Exception as exc:
            _warn(exc)

    def error(self, error: BaseException) -> None:
        """Record an error summary without exposing tracer failures."""
        try:
            self._span.error(error)
        except Exception as exc:
            _warn(exc)


@contextmanager
def open_span(
    tracer: Tracer, name: SpanName | str, **attributes: object
) -> Generator[Span, None, None]:
    """Open a redacted span while containing every ordinary tracer failure.

    Args:
        tracer: Span implementation selected by the caller.
        name: Reserved span name, or a caller-defined name.
        **attributes: Initial span attributes, redacted before forwarding.

    Note:
        Tracer failures are logged and ignored. Exceptions raised by the body
        pass through unchanged.
    """
    try:
        context = tracer.span(name, **redact_attributes(attributes))
        span = context.__enter__()
    except Exception as exc:
        _warn(exc)
        yield NOOP_SPAN
        return

    guarded = _GuardedSpan(span)
    try:
        yield guarded
    except BaseException as body_error:
        guarded.error(body_error)
        try:
            context.__exit__(type(body_error), body_error, body_error.__traceback__)
        except Exception as exc:
            _warn(exc)
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception as exc:
            _warn(exc)


def _warn(error: Exception) -> None:
    """Log a one-line summary of a tracer failure."""
    log.warning("tracing failure: %s", summarize_exception(error))
