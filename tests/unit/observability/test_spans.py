"""Guard tests ensure tracing failures never change application behavior."""

import asyncio
from contextlib import AbstractContextManager
from types import TracebackType

import pytest

from ontogr_mem.observability import Span, SpanName, open_span


class _RecordingSpan:
    """Record span updates and optionally fail at the tracer boundary."""

    def __init__(self, *, fail_set: bool = False, fail_error: bool = False) -> None:
        self.attributes: dict[str, object] = {}
        self.fail_set = fail_set
        self.fail_error = fail_error

    def set(self, name: str, value: object) -> None:
        if self.fail_set:
            raise RuntimeError("set failed")
        self.attributes[name] = value

    def error(self, error: BaseException) -> None:
        if self.fail_error:
            raise RuntimeError("error failed")
        self.attributes["error.summary"] = type(error).__name__


class _SpanContext(AbstractContextManager[Span]):
    """Context manager with configurable enter and exit failures."""

    def __init__(
        self,
        span: _RecordingSpan,
        *,
        fail_enter: bool = False,
        fail_exit: bool = False,
        suppress: bool = False,
    ) -> None:
        self.span = span
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit
        self.suppress = suppress

    def __enter__(self) -> Span:
        if self.fail_enter:
            raise RuntimeError("enter failed")
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self.fail_exit:
            raise RuntimeError("exit failed")
        return self.suppress


class _Tracer:
    """Return one configured span context and retain received attributes."""

    def __init__(
        self,
        *,
        fail_span: bool = False,
        fail_enter: bool = False,
        fail_exit: bool = False,
        fail_set: bool = False,
        fail_error: bool = False,
        suppress: bool = False,
    ) -> None:
        self.fail_span = fail_span
        self.attributes: dict[str, object] = {}
        self.context = _SpanContext(
            _RecordingSpan(fail_set=fail_set, fail_error=fail_error),
            fail_enter=fail_enter,
            fail_exit=fail_exit,
            suppress=suppress,
        )

    def span(
        self, name: SpanName | str, **attributes: object
    ) -> AbstractContextManager[Span]:
        if self.fail_span:
            raise RuntimeError("span failed")
        self.attributes = attributes
        return self.context


class TestOpenSpan:
    """The single guard redacts attributes and contains tracer exceptions."""

    @pytest.mark.parametrize("failure", ["span", "enter"])
    def test_open_failure_runs_the_body_and_logs_a_warning(self, failure, caplog):
        """A tracer that cannot open a span leaves body behavior unchanged."""
        tracer = _Tracer(fail_span=failure == "span", fail_enter=failure == "enter")

        with open_span(tracer, SpanName.WRITE) as span:
            assert span is not None
            result = "body result"

        assert result == "body result"
        assert "WARNING" in caplog.text

    def test_set_failure_does_not_escape(self, caplog):
        """A broken attribute update is reported but does not stop the body."""
        tracer = _Tracer(fail_set=True)

        with open_span(tracer, SpanName.WRITE) as span:
            span.set("count", 1)
            result = "completed"

        assert result == "completed"
        assert "WARNING" in caplog.text

    def test_error_failure_does_not_replace_the_body_exception(self, caplog):
        """A failing error reporter cannot replace the original exception."""
        tracer = _Tracer(fail_error=True)
        original = ValueError("body failed")

        with pytest.raises(ValueError) as excinfo, open_span(tracer, SpanName.WRITE):
            raise original

        assert excinfo.value is original
        assert "WARNING" in caplog.text

    def test_exit_failure_does_not_change_a_normal_body_result(self, caplog):
        """A context manager failure after a normal body is contained."""
        tracer = _Tracer(fail_exit=True)

        with open_span(tracer, SpanName.WRITE):
            result = "completed"

        assert result == "completed"
        assert "WARNING" in caplog.text

    def test_exit_failure_preserves_the_original_body_exception(self):
        """A tracer exit failure cannot replace the body's exception object."""
        tracer = _Tracer(fail_exit=True)
        original = ValueError("body failed")

        with pytest.raises(ValueError) as excinfo, open_span(tracer, SpanName.WRITE):
            raise original

        assert excinfo.value is original

    def test_tracer_suppression_cannot_swallow_cancellation(self):
        """Cancellation raised by the body always propagates to the caller."""
        tracer = _Tracer(suppress=True)

        with pytest.raises(asyncio.CancelledError), open_span(tracer, SpanName.WRITE):
            raise asyncio.CancelledError

    def test_attributes_are_redacted_and_capped_before_the_tracer_sees_them(self):
        """The tracer receives safe initial attributes and safe later updates."""
        tracer = _Tracer()

        with open_span(
            tracer,
            SpanName.WRITE,
            api_key="private",
            detail="x" * 9000,
        ) as span:
            span.set("authorization", "Bearer secret")

        assert tracer.attributes["api_key"] == "***"
        assert len(tracer.attributes["detail"]) == 8000
        assert tracer.context.span.attributes["authorization"] == "***"
