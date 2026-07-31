"""
Kafka event envelope schemas.

Every message flowing through Kafka MUST be wrapped in a KafkaEvent.
The payload field is typed by the event_type discriminator.
All schemas use Pydantic v2 for validation and serialization.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ─── Event types ──────────────────────────────────────────────────────────────

class EventType(str, Enum):
    QUERY_RECEIVED = "QUERY_RECEIVED"
    QUERY_REWRITTEN = "QUERY_REWRITTEN"
    SOURCES_SELECTED = "SOURCES_SELECTED"
    RESULTS_READY = "RESULTS_READY"
    RERANKED = "RERANKED"
    ANSWER_CHUNK = "ANSWER_CHUNK"
    ANSWER_DONE = "ANSWER_DONE"
    ERROR = "ERROR"


# ─── Payload models ───────────────────────────────────────────────────────────

class QueryReceivedPayload(BaseModel):
    query: str
    session_id: str
    tenant_id: str = "default"
    mode: str = "web"  # web | code | github | local | private
    options: dict[str, Any] = Field(default_factory=dict)


class QueryRewrittenPayload(BaseModel):
    original_query: str
    rewritten_query: str
    intent: str
    session_id: str


class SourcesSelectedPayload(BaseModel):
    session_id: str
    sources: list[str]  # SourceType values


class ResultsReadyPayload(BaseModel):
    session_id: str
    documents: list[dict[str, Any]]  # List[NormalizedDocument.to_dict()]
    source_type: str
    count: int


class RerankedPayload(BaseModel):
    session_id: str
    documents: list[dict[str, Any]]
    strategy: str  # rrf | cross_encoder | llm


class AnswerChunkPayload(BaseModel):
    session_id: str
    chunk: str
    chunk_index: int


class AnswerDonePayload(BaseModel):
    session_id: str
    full_answer: str
    citations: list[dict[str, Any]]
    duration_ms: int


class ErrorPayload(BaseModel):
    session_id: str
    error_type: str
    message: str
    recoverable: bool = True


# ─── Discriminated union ──────────────────────────────────────────────────────

AnyPayload = Union[
    QueryReceivedPayload,
    QueryRewrittenPayload,
    SourcesSelectedPayload,
    ResultsReadyPayload,
    RerankedPayload,
    AnswerChunkPayload,
    AnswerDonePayload,
    ErrorPayload,
]


# ─── Envelope ─────────────────────────────────────────────────────────────────

class KafkaEvent(BaseModel):
    """
    Universal Kafka event envelope.

    Every message published to any CloudSearch topic MUST be wrapped
    in this envelope to ensure traceability and schema uniformity.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    trace_id: str = ""            # OpenTelemetry trace ID (W3C format)
    span_id: str = ""             # OpenTelemetry span ID
    query_id: str = ""            # Alias for session_id where applicable
    payload: dict[str, Any]       # Serialized AnyPayload
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: Literal["1.0"] = "1.0"

    @classmethod
    def make(
        cls,
        event_type: EventType,
        payload: AnyPayload,
        trace_id: str = "",
        span_id: str = "",
        query_id: str = "",
    ) -> "KafkaEvent":
        """Convenience constructor that serializes the payload."""
        return cls(
            event_type=event_type,
            payload=payload.model_dump(),
            trace_id=trace_id,
            span_id=span_id,
            query_id=query_id,
        )

    def to_json_bytes(self) -> bytes:
        """Serialize to UTF-8 JSON bytes for Kafka producer."""
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "KafkaEvent":
        """Deserialize from Kafka consumer bytes."""
        return cls.model_validate_json(data)
