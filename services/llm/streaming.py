"""
SSE streaming bridge — converts async token generators to SSE event format.

CloudSearch uses Server-Sent Events (SSE) to stream answers to clients.

Event types:
    source_card  — emitted once per source before answer generation
    answer_chunk — emitted per LLM token
    answer_done  — emitted when generation is complete (includes citations)
    error        — emitted on fatal errors
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from cloudsearch_shared.document import NormalizedDocument

logger = logging.getLogger(__name__)


def _sse_event(event_type: str, data: Any) -> str:
    """Format a single SSE event string."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event_type}\ndata: {payload}\n\n"


async def stream_search_response(
    query: str,
    sources: list[NormalizedDocument],
    answer_tokens: AsyncIterator[str],
    citations: list[dict] | None = None,
) -> AsyncIterator[str]:
    """
    Full SSE stream for a search response.

    Emits:
      1. source_card events (one per source)
      2. answer_chunk events (one per token)
      3. answer_done event (final citations + full text)

    Args:
        query:         Original user query.
        sources:       Ordered list of source documents.
        answer_tokens: Async generator of LLM tokens.
        citations:     Optional pre-computed citation list from grounder.

    Yields:
        SSE-formatted strings ready to send over HTTP.
    """
    # Phase 1: emit source cards
    for idx, doc in enumerate(sources, start=1):
        card = {
            "index": idx,
            "id": doc.id,
            "title": doc.title,
            "url": doc.url,
            "snippet": doc.snippet,
            "source_type": doc.source_type.value,
            "score": round(doc.score, 4),
            "favicon_url": _favicon_url(doc.url),
            "metadata": doc.metadata,
        }
        yield _sse_event("source_card", card)

    # Phase 2: stream answer tokens
    full_answer = []
    chunk_index = 0
    try:
        async for token in answer_tokens:
            if token:
                full_answer.append(token)
                yield _sse_event("answer_chunk", {
                    "chunk": token,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
    except Exception as exc:
        logger.exception("Token streaming error: %s", exc)
        yield _sse_event("error", {"message": str(exc)})
        return

    # Phase 3: emit done event with full answer + citations
    yield _sse_event("answer_done", {
        "query": query,
        "answer": "".join(full_answer),
        "citations": citations or [],
        "source_count": len(sources),
    })


async def stream_error(message: str, recoverable: bool = False) -> AsyncIterator[str]:
    """Emit a single error SSE event."""
    yield _sse_event("error", {
        "message": message,
        "recoverable": recoverable,
    })


def _favicon_url(url: str) -> str:
    """Extract domain and return Google favicon URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
    except Exception:
        return ""
