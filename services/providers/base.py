"""
SearchProvider — abstract base class for all retrieval backends.

Every source adapter (Meilisearch, Web, Code, Private, Local) MUST
subclass SearchProvider and implement the ``search`` async generator.
This guarantees the Fusion Core can consume any provider uniformly.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator

from cloudsearch_shared.document import NormalizedDocument, SourceType


@dataclass
class SearchOptions:
    """
    Options forwarded to all providers during a fan-out search.

    Providers MAY honour all or a subset of these options.
    Unknown options MUST be silently ignored.
    """
    max_results: int = 10
    semantic_ratio: float = 0.5          # 0.0 = BM25 only, 1.0 = vector only
    language: str = "en"
    tenant_id: str = "default"
    filters: dict = field(default_factory=dict)
    mode: str = "web"                    # web | code | github | local | private
    include_content: bool = True         # include full chunk text in results


class SearchProvider(abc.ABC):
    """
    Abstract base class for all CloudSearch retrieval backends.

    Implementations must be safe to instantiate once and reuse across
    concurrent requests (i.e. connection pools, not per-request clients).

    Lifecycle:
        - ``initialize()`` is called once at service startup.
        - ``search()`` is called for every query fan-out.
        - ``close()`` is called at service shutdown.
    """

    @property
    @abc.abstractmethod
    def source_type(self) -> SourceType:
        """The SourceType enum value that identifies this backend."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name (used in logs + metrics)."""
        ...

    async def initialize(self) -> None:
        """
        Optional async setup (e.g. create index, warm connection pool).
        Called once before the first ``search`` call.
        Default implementation is a no-op.
        """

    async def close(self) -> None:
        """
        Optional async teardown.
        Default implementation is a no-op.
        """

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        opts: SearchOptions,
    ) -> AsyncIterator[NormalizedDocument]:
        """
        Execute a search and yield NormalizedDocument objects.

        Args:
            query: The user's (possibly rewritten) search query.
            opts:  Search options controlling result count, ratio, etc.

        Yields:
            NormalizedDocument objects in relevance order (highest first).

        Notes:
            - MUST be an async generator (``async def`` + ``yield``).
            - MUST NOT raise for transient network errors; yield nothing
              and log the error so the Fusion Core can degrade gracefully.
            - Scores MUST be normalized to [0.0, 1.0].
        """
        ...

    async def health_check(self) -> bool:
        """
        Return True if the backend is reachable and healthy.
        Override for backend-specific probes.
        """
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
