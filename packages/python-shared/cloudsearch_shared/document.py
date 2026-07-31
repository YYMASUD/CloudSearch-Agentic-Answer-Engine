"""
cloudsearch_shared — canonical document schema and shared types.

NormalizedDocument is the single unified representation that flows
through every layer of the CloudSearch pipeline. All SearchProvider
adapters MUST return documents conforming to this schema.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """The backend that produced this document."""
    INDEXED = "INDEXED"       # Meilisearch indexed corpus
    WEB = "WEB"               # Live web metasearch
    CODE = "CODE"             # Code / repo search
    PRIVATE = "PRIVATE"       # Private / team knowledge base
    LOCAL = "LOCAL"           # Local / offline corpus
    UNKNOWN = "UNKNOWN"       # Fallback


@dataclass
class NormalizedDocument:
    """
    Canonical document representation used by every pipeline layer.

    All scores are normalized to [0.0, 1.0] before being set here.
    The ``id`` field is a stable, deterministic hash derived from
    ``url`` + ``chunk_idx`` so duplicate detection is URL-keyed.

    Attributes:
        id:          Stable SHA-256 hash (first 16 hex chars) of url+chunk_idx.
        title:       Human-readable title of the source document.
        url:         Canonical URL of the source document.
        snippet:     Short display text (≤ 300 chars) for source cards.
        content:     Full chunk text used by the RAG / grounding layer.
        score:       Relevance score normalized to [0.0, 1.0].
        source_type: Which backend produced this document.
        metadata:    Provider-specific extras (e.g. GitHub stars, language).
        chunk_idx:   Zero-based index of this chunk within the parent document.
        doc_id:      Stable hash of the parent document URL (no chunk suffix).
    """

    id: str
    title: str
    url: str
    snippet: str
    content: str
    score: float
    source_type: SourceType
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_idx: int = 0
    doc_id: str = ""

    def __post_init__(self) -> None:
        # Clamp score to valid range
        self.score = max(0.0, min(1.0, self.score))
        # Auto-derive doc_id from URL if not provided
        if not self.doc_id:
            self.doc_id = _stable_hash(self.url)

    @classmethod
    def create(
        cls,
        *,
        title: str,
        url: str,
        content: str,
        score: float,
        source_type: SourceType,
        snippet: str = "",
        chunk_idx: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "NormalizedDocument":
        """Factory method that auto-generates the ``id`` field."""
        doc_id = _stable_hash(url)
        doc_hash = _stable_hash(f"{url}:{chunk_idx}")
        return cls(
            id=doc_hash,
            title=title,
            url=url,
            snippet=snippet or content[:300],
            content=content,
            score=score,
            source_type=source_type,
            metadata=metadata or {},
            chunk_idx=chunk_idx,
            doc_id=doc_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "score": self.score,
            "source_type": self.source_type.value,
            "metadata": self.metadata,
            "chunk_idx": self.chunk_idx,
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedDocument":
        """Deserialize from a plain dict."""
        return cls(
            id=data["id"],
            title=data["title"],
            url=data["url"],
            snippet=data.get("snippet", ""),
            content=data.get("content", ""),
            score=float(data.get("score", 0.0)),
            source_type=SourceType(data.get("source_type", SourceType.UNKNOWN)),
            metadata=data.get("metadata", {}),
            chunk_idx=int(data.get("chunk_idx", 0)),
            doc_id=data.get("doc_id", ""),
        )


def _stable_hash(value: str) -> str:
    """Return first 16 hex characters of SHA-256 hash."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]
