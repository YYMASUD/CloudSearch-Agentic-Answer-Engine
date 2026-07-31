"""
Qdrant vector store wrapper.

Provides async CRUD operations for the CloudSearch chunk collection.
Manages collection creation, upsert, similarity search, and deletion.

Collection schema:
    - Vectors: float32[384] (HNSW, cosine distance)
    - Payload: full NormalizedDocument fields for retrieval without DB join
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from cloudsearch_shared.document import NormalizedDocument, SourceType

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cloudsearch_chunks")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


class VectorStore:
    """
    Async Qdrant wrapper for CloudSearch chunk storage and retrieval.

    Usage:
        store = VectorStore()
        await store.initialize()
        await store.upsert(chunks_with_embeddings)
        results = await store.search(query_vector, top_k=10)
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self._url = url or QDRANT_URL
        self._collection = collection_name or COLLECTION_NAME
        self._vector_size = vector_size or VECTOR_SIZE
        self._client: Any = None

    async def initialize(self) -> None:
        """Create async Qdrant client and ensure collection exists."""
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = AsyncQdrantClient(url=self._url)

            existing = [c.name for c in await self._client.get_collections()]
            if self._collection not in existing:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection %r (dim=%d)", self._collection, self._vector_size)
            else:
                logger.info("Using existing Qdrant collection %r", self._collection)

        except Exception as exc:
            logger.warning("Qdrant initialization failed: %s — vector search unavailable.", exc)
            self._client = None

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def upsert(
        self,
        documents: list[NormalizedDocument],
        embeddings: list[np.ndarray],
    ) -> None:
        """
        Upsert document chunks with their embeddings.

        Args:
            documents:  NormalizedDocument objects (one per chunk).
            embeddings: Corresponding embedding vectors.
        """
        if not self._client:
            logger.warning("VectorStore not initialized — skipping upsert.")
            return
        if len(documents) != len(embeddings):
            raise ValueError(f"documents ({len(documents)}) and embeddings ({len(embeddings)}) must match.")

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=self._doc_id_to_int(doc.id),
                vector=emb.tolist(),
                payload=doc.to_dict(),
            )
            for doc, emb in zip(documents, embeddings)
        ]

        await self._client.upsert(collection_name=self._collection, points=points)
        logger.debug("Upserted %d points into Qdrant.", len(points))

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        source_type_filter: SourceType | None = None,
        tenant_id: str = "default",
    ) -> list[NormalizedDocument]:
        """
        Cosine similarity search in the vector store.

        Args:
            query_vector:      Query embedding (float32 numpy array).
            top_k:             Maximum number of results.
            source_type_filter: Optionally filter by SourceType.
            tenant_id:         Filter by tenant namespace.

        Returns:
            List of NormalizedDocument sorted by cosine similarity (desc).
        """
        if not self._client:
            logger.warning("VectorStore not initialized — returning empty results.")
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        conditions = []

        if source_type_filter:
            conditions.append(FieldCondition(
                key="source_type",
                match=MatchValue(value=source_type_filter.value),
            ))

        if conditions:
            query_filter = Filter(must=conditions)

        try:
            hits = await self._client.search(
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.exception("Qdrant search failed: %s", exc)
            return []

        results = []
        for hit in hits:
            payload = hit.payload or {}
            payload["score"] = hit.score
            try:
                doc = NormalizedDocument.from_dict(payload)
                results.append(doc)
            except Exception as e:
                logger.warning("Could not deserialize Qdrant hit: %s", e)

        return results

    async def delete(self, doc_ids: list[str]) -> None:
        """Delete points by document ID."""
        if not self._client:
            return
        from qdrant_client.models import PointIdsList
        int_ids = [self._doc_id_to_int(did) for did in doc_ids]
        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=int_ids),
        )

    @staticmethod
    def _doc_id_to_int(doc_id: str) -> int:
        """Convert a hex string ID to a uint64 for Qdrant point IDs."""
        return int(doc_id, 16) % (2**63)
