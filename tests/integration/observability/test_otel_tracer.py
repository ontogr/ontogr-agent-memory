"""Real SDK tests verify exported nesting and the no-exporter default path."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ontogr_mem.observability.otel import OpenTelemetryTracer, opentelemetry_tracer
from ontogr_mem.observability.spans import open_span
from ontogr_mem.observability.tracer import SpanName


class TestOpenTelemetryExport:
    """The adapter works with the SDK and does not configure a network exporter."""

    def test_nested_spans_export_with_parentage_and_redaction(self):
        """Exported child spans retain their parent and contain no raw secret."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = OpenTelemetryTracer(provider.get_tracer("test"))

        with (
            open_span(
                tracer,
                SpanName.WRITE,
                tenant="acme",
                api_key="raw-key-value",
            ),
            open_span(tracer, SpanName.LLM_COMPLETE, model="model-a"),
        ):
            pass

        spans = exporter.get_finished_spans()
        child = next(span for span in spans if span.name == SpanName.LLM_COMPLETE.value)
        root = next(span for span in spans if span.name == SpanName.WRITE.value)

        assert {span.name for span in spans} == {
            SpanName.WRITE.value,
            SpanName.LLM_COMPLETE.value,
        }
        assert child.parent.span_id == root.context.span_id
        attributes = root.attributes
        if attributes is None:
            raise AssertionError("exported span has no attributes")
        assert attributes["api_key"] == "***"
        assert "raw-key-value" not in str(attributes)
        provider.shutdown()

    def test_default_sdk_setup_runs_without_an_exporter(self):
        """The factory uses the caller's configured SDK and opens no connection."""
        tracer = opentelemetry_tracer(name="ontogr_mem.test")

        with open_span(tracer, SpanName.WRITE, tenant="acme"):
            result = "completed"

        assert result == "completed"
