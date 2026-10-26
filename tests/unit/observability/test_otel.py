"""OpenTelemetry adapter tests use fake SDK boundaries and no network access."""

import sys
from contextlib import contextmanager

import pytest

from ontogr_mem.errors import MissingExtraError
from ontogr_mem.observability.otel import OpenTelemetryTracer, opentelemetry_tracer
from ontogr_mem.observability.spans import open_span
from ontogr_mem.observability.tracer import SpanName


class _OtelSpan:
    """Collect OpenTelemetry attribute writes for assertions."""

    def __init__(self, *, fail_set: bool = False) -> None:
        self.attributes: dict[str, object] = {}
        self.fail_set = fail_set

    def set_attribute(self, name: str, value: object) -> None:
        if self.fail_set:
            raise RuntimeError("attribute failed")
        self.attributes[name] = value


class _OtelTracer:
    """Expose the OpenTelemetry context-manager interface used by the adapter."""

    def __init__(self, *, fail_start: bool = False, fail_set: bool = False) -> None:
        self.names: list[str] = []
        self.span = _OtelSpan(fail_set=fail_set)
        self.fail_start = fail_start

    @contextmanager
    def start_as_current_span(self, name: str):
        self.names.append(name)
        if self.fail_start:
            raise RuntimeError("start failed")
        yield self.span


class TestOpenTelemetryTracer:
    """The adapter maps safe spans to OpenTelemetry without requiring an exporter."""

    def test_factory_reports_the_missing_extra(self, monkeypatch):
        """A missing OpenTelemetry installation gives the documented install hint."""
        monkeypatch.delitem(sys.modules, "opentelemetry.trace", raising=False)
        monkeypatch.setitem(sys.modules, "opentelemetry", None)

        with pytest.raises(MissingExtraError, match=r"agent-memory\[otel\]"):
            opentelemetry_tracer()

    def test_span_names_and_attributes_are_forwarded_safely(self):
        """Names remain unprefixed and secret or None attributes are omitted."""
        otel = _OtelTracer()
        tracer = OpenTelemetryTracer(otel)

        with open_span(
            tracer,
            SpanName.WRITE,
            api_key="private",
            authorization="Bearer secret",
            missing=None,
        ) as span:
            span.error(ValueError("failed"))

        assert otel.names == [SpanName.WRITE.value]
        assert otel.span.attributes == {
            "api_key": "***",
            "authorization": "***",
            "error.summary": "ValueError: failed",
        }

    @pytest.mark.parametrize("failure", ["start", "set"])
    def test_adapter_failures_do_not_interrupt_the_body(self, failure):
        """Failed span creation or attribute updates leave body work intact."""
        otel = _OtelTracer(
            fail_start=failure == "start",
            fail_set=failure == "set",
        )
        tracer = OpenTelemetryTracer(otel)

        with open_span(tracer, SpanName.WRITE, count=1):
            result = "completed"

        assert result == "completed"
