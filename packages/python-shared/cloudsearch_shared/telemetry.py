"""
OpenTelemetry bootstrap utility.

Call ``setup_telemetry(service_name)`` once at application startup
(before any HTTP handlers are registered) to configure tracing,
metrics, and logging exporters.
"""
from __future__ import annotations

import logging
import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)


def setup_telemetry(
    service_name: str,
    *,
    otlp_endpoint: str | None = None,
    enable_console_exporter: bool = False,
) -> tuple[TracerProvider, MeterProvider]:
    """
    Bootstrap OpenTelemetry SDK for the given service.

    Args:
        service_name:            Service name tag applied to all telemetry.
        otlp_endpoint:           OTLP gRPC endpoint. Defaults to env var
                                 ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
                                 ``http://localhost:4317``.
        enable_console_exporter: If True, also export spans to stdout
                                 (useful for local debugging).

    Returns:
        Tuple of (TracerProvider, MeterProvider) for optional manual use.
    """
    endpoint = (
        otlp_endpoint
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )

    resource = Resource.create({SERVICE_NAME: service_name})

    # ─── Tracing ──────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)

    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    if enable_console_exporter:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    trace.set_tracer_provider(tracer_provider)

    # ─── Metrics ──────────────────────────────────────────────────────
    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    logger.info(
        "OpenTelemetry configured",
        extra={"service": service_name, "otlp_endpoint": endpoint},
    )

    return tracer_provider, meter_provider


def get_tracer(name: str) -> trace.Tracer:
    """Convenience wrapper for ``trace.get_tracer``."""
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Convenience wrapper for ``metrics.get_meter``."""
    return metrics.get_meter(name)
