"""
Citation grounder — maps every answer claim to supporting source chunks.

Strategy:
1. Split the generated answer into sentences (claims).
2. For each claim, compute a relevance score against every retrieved chunk
   using a combination of:
   a. BM25 token overlap (fast lexical signal)
   b. Cosine similarity of sentence embeddings (semantic signal)
3. Assign the top-scoring chunk to each claim as its citation.
4. Return a structured list of CitationMapping objects.

The grounder is intentionally lightweight — it runs after answer generation
and must complete in <100ms for a typical answer + 10 source chunks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cloudsearch_shared.document import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """
    A single citation linking an answer sentence to a source chunk.

    Attributes:
        citation_number: 1-based citation index displayed in the answer as [N].
        claim:           The answer sentence that cites this source.
        document:        The supporting NormalizedDocument.
        confidence:      Combined lexical + semantic confidence [0, 1].
        matched_snippet: The exact phrase in the document that supports the claim.
    """
    citation_number: int
    claim: str
    document: NormalizedDocument
    confidence: float
    matched_snippet: str = ""


@dataclass
class GroundingResult:
    """The full grounding output for a generated answer."""
    answer_with_citations: str          # Answer text with [N] markers inserted
    citations: list[Citation]            # Citation objects in number order
    unique_sources: list[NormalizedDocument]  # Deduplicated source list

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_with_citations": self.answer_with_citations,
            "citations": [
                {
                    "number": c.citation_number,
                    "claim": c.claim,
                    "doc_id": c.document.id,
                    "url": c.document.url,
                    "title": c.document.title,
                    "snippet": c.document.snippet,
                    "confidence": c.confidence,
                    "matched_snippet": c.matched_snippet,
                }
                for c in self.citations
            ],
            "unique_sources": [doc.to_dict() for doc in self.unique_sources],
        }


class CitationGrounder:
    """
    Post-generation citation grounding engine.

    Operates purely on text (no GPU required) using BM25 overlap as the
    primary signal and embedding cosine similarity as a secondary signal
    when sentence-transformers is available.
    """

    def __init__(
        self,
        min_confidence: float = 0.1,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
    ) -> None:
        self.min_confidence = min_confidence
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self._embedder: Any = None

    def _try_load_embedder(self) -> Any:
        """Lazy-load the sentence embedder if available."""
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return self._embedder
        except Exception:
            return None

    def ground(
        self,
        answer: str,
        retrieved_docs: list[NormalizedDocument],
    ) -> GroundingResult:
        """
        Ground the answer against retrieved documents.

        Args:
            answer:         The raw LLM-generated answer (no citation markers yet).
            retrieved_docs: The top-K documents returned by the RAG pipeline.

        Returns:
            GroundingResult with citation-annotated answer and citation list.
        """
        if not retrieved_docs:
            return GroundingResult(
                answer_with_citations=answer,
                citations=[],
                unique_sources=[],
            )

        claims = self._split_into_claims(answer)
        doc_tokens = [self._tokenize(doc.content) for doc in retrieved_docs]

        # Lazy import numpy — optional dep, degrade gracefully
        try:
            import numpy as np
            _np = np
        except ImportError:
            _np = None
        # Try to get embeddings for semantic matching
        embedder = self._try_load_embedder()
        claim_embs: list | None = None
        doc_embs: list | None = None

        if embedder is not None and _np is not None:
            try:
                all_texts = claims + [doc.content[:512] for doc in retrieved_docs]
                all_embs = embedder.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
                claim_embs = list(all_embs[:len(claims)])
                doc_embs = list(all_embs[len(claims):])
            except Exception as exc:
                logger.debug("Embedding for grounding failed: %s", exc)

        citations: list[Citation] = []
        seen_doc_ids: set[str] = set()
        unique_sources: list[NormalizedDocument] = []
        citation_counter = 1

        for claim_idx, claim in enumerate(claims):
            if not claim.strip():
                continue

            claim_tokens = self._tokenize(claim)
            best_score = 0.0
            best_doc_idx = 0

            for doc_idx, doc in enumerate(retrieved_docs):
                # Lexical BM25-style overlap
                lexical = self._token_overlap(claim_tokens, doc_tokens[doc_idx])

                # Semantic cosine similarity
                semantic = 0.0
                if claim_embs is not None and doc_embs is not None and _np is not None:
                    semantic = float(_np.dot(claim_embs[claim_idx], doc_embs[doc_idx]))
                    semantic = max(0.0, semantic)  # clamp negatives

                combined = self.lexical_weight * lexical + self.semantic_weight * semantic

                if combined > best_score:
                    best_score = combined
                    best_doc_idx = doc_idx

            if best_score < self.min_confidence:
                continue  # Skip low-confidence claims

            best_doc = retrieved_docs[best_doc_idx]
            matched = self._find_matching_phrase(claim, best_doc.content)

            # Assign citation number
            if best_doc.id not in seen_doc_ids:
                seen_doc_ids.add(best_doc.id)
                unique_sources.append(best_doc)

            citations.append(Citation(
                citation_number=citation_counter,
                claim=claim,
                document=best_doc,
                confidence=best_score,
                matched_snippet=matched,
            ))
            citation_counter += 1

        annotated = self._insert_citation_markers(answer, citations)

        return GroundingResult(
            answer_with_citations=annotated,
            citations=citations,
            unique_sources=unique_sources,
        )

    # ─── Private helpers ──────────────────────────────────────────────

    def _split_into_claims(self, text: str) -> list[str]:
        """Split answer into individual sentences (claims)."""
        # Split on . ! ? followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> set[str]:
        """Simple whitespace+punctuation tokenizer returning a token set."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        # Filter stopwords inline (a minimal set)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                     "on", "at", "to", "for", "and", "or", "but", "it", "this",
                     "that", "be", "been", "with", "by", "as", "from"}
        return {t for t in tokens if t not in stopwords and len(t) > 2}

    def _token_overlap(self, claim_tokens: set[str], doc_tokens: set[str]) -> float:
        """Jaccard-style overlap score between token sets."""
        if not claim_tokens or not doc_tokens:
            return 0.0
        intersection = len(claim_tokens & doc_tokens)
        union = len(claim_tokens | doc_tokens)
        return intersection / union if union > 0 else 0.0

    def _find_matching_phrase(self, claim: str, doc_content: str, window: int = 150) -> str:
        """Find the best matching phrase in the doc for display in citation tooltip."""
        claim_words = set(claim.lower().split())
        doc_words = doc_content.lower().split()
        best_start = 0
        best_overlap = 0

        for i in range(len(doc_words)):
            window_words = set(doc_words[i: i + 20])
            overlap = len(claim_words & window_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_start = i

        # Extract a snippet around the best window
        char_start = len(" ".join(doc_content.split()[:best_start]))
        snippet = doc_content[max(0, char_start): char_start + window].strip()
        return snippet + ("…" if len(snippet) >= window else "")

    def _insert_citation_markers(
        self, answer: str, citations: list[Citation]
    ) -> str:
        """
        Insert [N] citation markers after each cited claim in the answer.

        The LLM may already have inserted [N] markers; if so, this is a no-op.
        Otherwise, we append the marker after each cited sentence.
        """
        if not citations:
            return answer

        # If the answer already contains [1] style markers, trust the LLM
        if re.search(r"\[\d+\]", answer):
            return answer

        # Map claim → citation number
        claim_to_num = {c.claim: c.citation_number for c in citations}

        # Re-split and re-join with markers
        parts = []
        for sentence in self._split_into_claims(answer):
            parts.append(sentence)
            if sentence in claim_to_num:
                parts[-1] = sentence + f" [{claim_to_num[sentence]}]"

        return " ".join(parts)
