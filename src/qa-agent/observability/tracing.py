"""
OpenTelemetry tracing across the agent loop: `setup_tracing()` configures
a `TracerProvider` once at process startup (`api/app.py`'s lifespan), and
`traced_span()` wraps each pipeline stage
(ingestion/planning/execution/verification/triage/reporting) in a span so
a single run's full timeline — including LLM call latency and browser
action latency as child spans — is reconstructable in any OTLP-compatible
backend (Jaeger, Tempo, Honeycomb, etc.).

Exporter choice follows `settings.environment`: console exporter for
local dev (spans printed to stdout, no collector required), OTLP for
everything else (dev/staging/prod all point at a real collector).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span, Status, StatusCode

from qa_agent.config.settings import Environment, Settings, get_settings

_TRACER_NAME = "qa_agent"
_configured = False


def setup_tracing(settings: Settings | None = None, otlp_endpoint: str | None = None) -> None:
    """
    Configures the global `TracerProvider`. Idempotent — safe to call
    more than once (e.g. once from `api/app.py`'s lifespan and once from
    a test fixture) without producing duplicate exporters.
    """
    global _configured
    if _configured:
        return

    settings = settings or get_settings()
    resource = Resource.create(
        {SERVICE_NAME: settings.app_name, "environment": settings.environment.value}
    )
    provider = TracerProvider(resource=resource)

    if settings.environment == Environment.LOCAL:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise RuntimeError(
                "Non-local environments require the "
                "'opentelemetry-exporter-otlp-proto-grpc' package for tracing export."
            ) from exc
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint) if otlp_endpoint else OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def traced_span(name: str, **attributes: str | int | float | bool) -> Iterator[Span]:
    """
    Wraps a block of code in a span, recording an exception (if raised)
    as a span event and marking the span's status as ERROR before
    re-raising — so a failed pipeline stage is visible as a red span in
    the trace UI without every call site needing its own try/except.

    Example:
        with traced_span("planning.generate_plan", story_id=story.id):
            plan = await test_planner.plan(intent)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def reset_tracing_for_tests() -> None:
    """Allows a test suite to call `setup_tracing()` again with different settings between test cases."""
    global _configured
    _configured = False