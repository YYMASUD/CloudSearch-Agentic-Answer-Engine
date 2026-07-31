"""
cloudsearch_shared package init.
Exports the public API surface.
"""
from .document import NormalizedDocument, SourceType
from .events import (
    EventType,
    KafkaEvent,
    QueryReceivedPayload,
    QueryRewrittenPayload,
    ResultsReadyPayload,
    AnswerChunkPayload,
    AnswerDonePayload,
    ErrorPayload,
)
try:
    from .telemetry import setup_telemetry, get_tracer, get_meter
    _telemetry_available = True
except ImportError:
    # opentelemetry not installed (e.g. in test environments)
    _telemetry_available = False

    def setup_telemetry(service_name: str = "unknown") -> None:  # type: ignore[misc]
        pass

    def get_tracer(name: str = "unknown"):  # type: ignore[misc]
        return None

    def get_meter(name: str = "unknown"):  # type: ignore[misc]
        return None

__all__ = [
    "NormalizedDocument",
    "SourceType",
    "EventType",
    "KafkaEvent",
    "QueryReceivedPayload",
    "QueryRewrittenPayload",
    "ResultsReadyPayload",
    "AnswerChunkPayload",
    "AnswerDonePayload",
    "ErrorPayload",
    "setup_telemetry",
    "get_tracer",
    "get_meter",
]
