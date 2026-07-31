"""
Embedding router — selects local or cloud embedding provider based on config.

Priority: LOCAL (sentence-transformers) → OPENAI (text-embedding-3-small) → stub.
All providers return numpy float32 arrays of the configured dimension.

The EmbedderRouter caches loaded models in memory and is safe to reuse
across concurrent requests (models are thread-safe for inference).
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VECTOR_DIM = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))


class BaseEmbedder(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float32 numpy arrays, one per text.
        """
        ...

    async def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for single-text embedding."""
        results = await self.embed([text])
        return results[0]


class LocalEmbedder(BaseEmbedder):
    """
    sentence-transformers local embedding model.
    Lazy-loads on first call; subsequent calls reuse the cached model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dim: int = VECTOR_DIM

    @property
    def dimension(self) -> int:
        return self._dim

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded local embedding model %r (dim=%d)", self._model_name, self._dim)
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Encode texts using the local model (runs synchronously in-thread)."""
        import asyncio
        import concurrent.futures

        self._load_model()

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            embeddings = await loop.run_in_executor(
                pool,
                lambda: self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            )

        return [emb.astype(np.float32) for emb in embeddings]


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-small cloud embedding provider.
    Falls back to LocalEmbedder if OPENAI_API_KEY is not set.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._dim = 1536 if "3-large" in model else 1536  # text-embedding-3-small = 1536

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY not set — cannot use OpenAIEmbedder.")

        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": texts},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            np.array(item["embedding"], dtype=np.float32)
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]


class StubEmbedder(BaseEmbedder):
    """
    Zero-vector stub — used in tests and when no embedder is configured.
    Always returns a zero vector of the configured dimension.
    """

    @property
    def dimension(self) -> int:
        return VECTOR_DIM

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        logger.warning("StubEmbedder in use — returning zero vectors.")
        return [np.zeros(VECTOR_DIM, dtype=np.float32) for _ in texts]


class EmbedderRouter:
    """
    Selects the appropriate embedding provider based on env config.

    Priority: LOCAL → OPENAI → STUB
    """

    def __init__(self) -> None:
        self._embedder: BaseEmbedder | None = None

    def _build_embedder(self) -> BaseEmbedder:
        source = os.getenv("MEILISEARCH_EMBEDDER_SOURCE", "local").lower()

        if source == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                logger.info("Using OpenAIEmbedder")
                return OpenAIEmbedder(api_key=api_key)
            logger.warning("OPENAI_API_KEY not set; falling back to local embedder.")

        try:
            logger.info("Using LocalEmbedder (sentence-transformers/all-MiniLM-L6-v2)")
            return LocalEmbedder()
        except ImportError:
            logger.warning("sentence-transformers unavailable; using StubEmbedder.")
            return StubEmbedder()

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = self._build_embedder()
        return self._embedder

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts using the selected provider."""
        return await self.embedder.embed(texts)

    async def embed_one(self, text: str) -> np.ndarray:
        return await self.embedder.embed_one(text)


# Module-level singleton — import and use directly
embedder_router = EmbedderRouter()
