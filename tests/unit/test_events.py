"""
Unit tests for Kafka event schemas.
Run with: pytest tests/unit/test_events.py -v
"""
import json
import pytest
from cloudsearch_shared.events import (
    EventType,
    KafkaEvent,
    QueryReceivedPayload,
    AnswerChunkPayload,
    AnswerDonePayload,
    ErrorPayload,
)


class TestKafkaEvent:
    def _make_query_event(self, session_id: str = "sess-001") -> KafkaEvent:
        payload = QueryReceivedPayload(
            query="What is RAG?",
            session_id=session_id,
            mode="web",
        )
        return KafkaEvent.make(
            event_type=EventType.QUERY_RECEIVED,
            payload=payload,
            query_id=session_id,
        )

    def test_make_sets_event_type(self):
        event = self._make_query_event()
        assert event.event_type == EventType.QUERY_RECEIVED

    def test_make_auto_generates_event_id(self):
        e1 = self._make_query_event()
        e2 = self._make_query_event()
        assert e1.event_id != e2.event_id  # UUIDs are unique

    def test_make_sets_timestamp(self):
        event = self._make_query_event()
        assert event.timestamp  # Non-empty ISO string

    def test_make_serializes_payload(self):
        event = self._make_query_event()
        assert event.payload["query"] == "What is RAG?"
        assert event.payload["session_id"] == "sess-001"

    def test_to_json_bytes_is_valid_json(self):
        event = self._make_query_event()
        raw = event.to_json_bytes()
        assert isinstance(raw, bytes)
        parsed = json.loads(raw)
        assert parsed["event_type"] == "QUERY_RECEIVED"

    def test_round_trip_json_bytes(self):
        event = self._make_query_event("session-abc")
        raw = event.to_json_bytes()
        restored = KafkaEvent.from_json_bytes(raw)
        assert restored.event_id == event.event_id
        assert restored.event_type == event.event_type
        assert restored.payload["query"] == "What is RAG?"
        assert restored.schema_version == "1.0"

    def test_answer_chunk_event(self):
        payload = AnswerChunkPayload(
            session_id="sess-002",
            chunk="The answer is ",
            chunk_index=0,
        )
        event = KafkaEvent.make(
            event_type=EventType.ANSWER_CHUNK,
            payload=payload,
        )
        raw = event.to_json_bytes()
        restored = KafkaEvent.from_json_bytes(raw)
        assert restored.payload["chunk"] == "The answer is "
        assert restored.payload["chunk_index"] == 0

    def test_error_event_has_recoverable_flag(self):
        payload = ErrorPayload(
            session_id="sess-003",
            error_type="ProviderTimeout",
            message="Meilisearch timed out",
            recoverable=True,
        )
        event = KafkaEvent.make(
            event_type=EventType.ERROR,
            payload=payload,
        )
        assert event.payload["recoverable"] is True

    def test_trace_id_propagated(self):
        payload = QueryReceivedPayload(query="test", session_id="s1")
        event = KafkaEvent.make(
            event_type=EventType.QUERY_RECEIVED,
            payload=payload,
            trace_id="abc-trace-123",
            span_id="def-span-456",
        )
        raw = event.to_json_bytes()
        restored = KafkaEvent.from_json_bytes(raw)
        assert restored.trace_id == "abc-trace-123"
        assert restored.span_id == "def-span-456"
