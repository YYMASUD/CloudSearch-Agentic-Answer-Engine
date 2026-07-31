"""
Session store — persists search sessions to Postgres for analytics & history.

Uses asyncpg for async writes. Errors are silently logged so a DB failure
never disrupts the search response.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class SessionStore:
    """Async Postgres session store backed by asyncpg connection pool."""

    def __init__(self) -> None:
        self._pool: Any = None
        self._dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://cloudsearch:changeme@localhost:5432/cloudsearch",
        ).replace("postgresql+asyncpg://", "postgresql://")

    async def initialize(self) -> None:
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=5,
            )
            logger.info("SessionStore connected to Postgres.")
        except Exception as exc:
            logger.warning("SessionStore unavailable (%s) — session persistence disabled.", exc)
            self._pool = None

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def available(self) -> bool:
        return self._pool is not None

    async def save_session(
        self,
        *,
        session_id: str,
        query: str,
        rewritten_query: str,
        intent: str,
        sources_used: list[str],
        answer_text: str,
        citations: list[dict],
        duration_ms: int,
        tenant_id: str = "default",
    ) -> None:
        """Persist a completed search session. Fire-and-forget safe."""
        if not self._pool:
            return
        try:
            import json
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO search_sessions (
                        id, query, rewritten_query, query_intent,
                        sources_used, answer_text, citations,
                        duration_ms, tenant_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    session_id,
                    query,
                    rewritten_query,
                    intent,
                    json.dumps(sources_used),
                    answer_text,
                    json.dumps(citations),
                    duration_ms,
                    tenant_id,
                )
        except Exception as exc:
            logger.warning("SessionStore.save_session failed: %s", exc)

    async def get_recent_sessions(
        self, tenant_id: str = "default", limit: int = 20
    ) -> list[dict]:
        """Return recent search sessions for a tenant."""
        if not self._pool:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, query, query_intent, duration_ms, created_at
                    FROM search_sessions
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    tenant_id,
                    limit,
                )
                return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("SessionStore.get_recent_sessions failed: %s", exc)
            return []

    async def health_check(self) -> bool:
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
