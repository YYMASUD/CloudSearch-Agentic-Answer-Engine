"""
Semantic + fixed-size chunker for RAG pipelines.

Splits ParsedDocument pages into chunks suitable for embedding and retrieval.
Two strategies are supported and can be combined:

1. SentenceChunker — groups sentences into chunks of ~N tokens, with overlap.
   Best for prose documents where sentence boundaries matter.

2. FixedSizeChunker — splits by character count with configurable overlap.
   Best for code, structured data, or very long uniform text.

The default pipeline uses SentenceChunker with FixedSizeChunker as fallback
for pages whose sentence detection fails.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from cloudsearch_shared.document import NormalizedDocument, SourceType
from .parser import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk ready for embedding."""
    text: str
    chunk_idx: int
    page_num: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class SentenceChunker:
    """
    Groups sentences into overlapping chunks of approximately ``target_tokens`` tokens.

    Uses a simple regex-based sentence splitter (avoids NLTK dependency at runtime,
    though NLTK punkt is used when available for higher quality splits).
    """

    def __init__(
        self,
        target_tokens: int = 256,
        overlap_tokens: int = 32,
        chars_per_token: float = 4.0,
    ) -> None:
        self.target_chars = int(target_tokens * chars_per_token)
        self.overlap_chars = int(overlap_tokens * chars_per_token)
        self._nltk_available = self._check_nltk()

    def _check_nltk(self) -> bool:
        try:
            import nltk
            nltk.data.find("tokenizers/punkt")
            return True
        except Exception:
            return False

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK if available, else regex."""
        if self._nltk_available:
            try:
                import nltk
                return nltk.sent_tokenize(text)
            except Exception:
                pass
        # Regex fallback: split on . ! ? followed by whitespace + capital
        return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    def chunk(self, text: str, page_num: int = 0) -> list[TextChunk]:
        """Split text into overlapping sentence-boundary-aligned chunks."""
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks: list[TextChunk] = []
        current: list[str] = []
        current_len = 0
        start_char = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_len + sentence_len > self.target_chars and current:
                chunk_text = " ".join(current)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_idx=chunk_idx,
                    page_num=page_num,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                ))
                chunk_idx += 1

                # Overlap: keep last N chars worth of sentences
                overlap_text = ""
                for s in reversed(current):
                    if len(overlap_text) + len(s) <= self.overlap_chars:
                        overlap_text = s + " " + overlap_text
                    else:
                        break
                start_char = start_char + len(chunk_text) - len(overlap_text)
                current = [overlap_text.strip()] if overlap_text.strip() else []
                current_len = len(overlap_text)

            current.append(sentence)
            current_len += sentence_len

        # Final chunk
        if current:
            chunk_text = " ".join(current)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_idx=chunk_idx,
                page_num=page_num,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
            ))

        return chunks


class FixedSizeChunker:
    """
    Simple fixed-size character chunker with overlap.
    Used as a fallback for code, structured data, or very long text.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, page_num: int = 0) -> list[TextChunk]:
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_idx=chunk_idx,
                page_num=page_num,
                start_char=start,
                end_char=end,
            ))
            chunk_idx += 1
            start = end - self.overlap
            if start >= len(text):
                break
        return chunks


class DocumentChunker:
    """
    Orchestrates chunking of a ParsedDocument into TextChunks and then
    into NormalizedDocument objects ready for embedding + indexing.

    Strategy selection:
        - Prose pages (HTML, PDF, DOCX) → SentenceChunker
        - Code pages or very short text → FixedSizeChunker fallback
        - Empty pages are skipped
    """

    def __init__(
        self,
        target_tokens: int = 256,
        overlap_tokens: int = 32,
        min_chunk_chars: int = 50,
    ) -> None:
        self._sentence_chunker = SentenceChunker(
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        self._fixed_chunker = FixedSizeChunker(
            chunk_size=int(target_tokens * 4),
            overlap=int(overlap_tokens * 4),
        )
        self.min_chunk_chars = min_chunk_chars

    def chunk_document(
        self,
        doc: ParsedDocument,
        source_type: SourceType = SourceType.INDEXED,
    ) -> list[NormalizedDocument]:
        """
        Chunk a ParsedDocument into NormalizedDocument objects.

        Args:
            doc:         The parsed document.
            source_type: The SourceType to assign to each chunk.

        Returns:
            List of NormalizedDocument objects, one per chunk.
        """
        results: list[NormalizedDocument] = []
        global_chunk_idx = 0

        for page in doc.pages:
            if not page.text or not page.text.strip():
                continue

            chunks = self._chunk_page(page)
            for chunk in chunks:
                if len(chunk.text.strip()) < self.min_chunk_chars:
                    continue  # Skip tiny slivers

                nd = NormalizedDocument.create(
                    title=doc.title,
                    url=doc.url,
                    content=chunk.text,
                    snippet=chunk.text[:300],
                    score=1.0,  # Raw chunks get score=1.0; re-ranking adjusts later
                    source_type=source_type,
                    chunk_idx=global_chunk_idx,
                    metadata={
                        **doc.metadata,
                        "page_num": chunk.page_num,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
                results.append(nd)
                global_chunk_idx += 1

        logger.debug(
            "Chunked %r into %d chunks from %d pages.",
            doc.url,
            len(results),
            len(doc.pages),
        )
        return results

    def _chunk_page(self, page: ParsedPage) -> list[TextChunk]:
        """Choose chunking strategy for a single page."""
        text = page.text.strip()
        if not text:
            return []

        # Use sentence chunker for prose; fixed-size for very short/code text
        try:
            chunks = self._sentence_chunker.chunk(text, page_num=page.page_num)
            if chunks:
                return chunks
        except Exception as exc:
            logger.debug("SentenceChunker failed: %s — falling back to FixedSizeChunker.", exc)

        return self._fixed_chunker.chunk(text, page_num=page.page_num)
