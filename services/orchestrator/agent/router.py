"""
Source router — maps QueryIntent to the set of SearchProvider backends to activate.

The routing table is config-driven (JSON/YAML override supported) so
operators can tune which backends fire for each intent without code changes.

Default routing table:
    FACTUAL   → WEB + INDEXED
    RESEARCH  → WEB + INDEXED + LOCAL
    CODE      → CODE + INDEXED
    GITHUB    → CODE (github submode)
    LOCAL     → LOCAL
    PRIVATE   → PRIVATE
    MATH      → WEB + INDEXED
    UNKNOWN   → WEB + INDEXED

Each route also carries a priority weight per source type that the
Fusion Core uses to bias RRF merging.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from cloudsearch_shared.document import SourceType
from .planner import QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class SourceRoute:
    """
    A single routing decision: which backends to query + their weights.

    Attributes:
        sources:  Set of SourceType backends to activate.
        weights:  Per-source weight for Fusion Core bias (default 1.0).
        max_results_per_source: Per-source result count limit.
    """
    sources: set[SourceType]
    weights: dict[SourceType, float] = field(default_factory=dict)
    max_results_per_source: int = 10

    def weight_for(self, source: SourceType) -> float:
        return self.weights.get(source, 1.0)


# ─── Default routing table ────────────────────────────────────────────────────

_DEFAULT_ROUTING_TABLE: dict[QueryIntent, SourceRoute] = {
    QueryIntent.FACTUAL: SourceRoute(
        sources={SourceType.WEB, SourceType.INDEXED},
        weights={SourceType.WEB: 1.2, SourceType.INDEXED: 1.0},
    ),
    QueryIntent.RESEARCH: SourceRoute(
        sources={SourceType.WEB, SourceType.INDEXED, SourceType.LOCAL},
        weights={SourceType.WEB: 1.0, SourceType.INDEXED: 1.2, SourceType.LOCAL: 0.8},
    ),
    QueryIntent.CODE: SourceRoute(
        sources={SourceType.CODE, SourceType.INDEXED},
        weights={SourceType.CODE: 1.5, SourceType.INDEXED: 0.7},
        max_results_per_source=15,
    ),
    QueryIntent.GITHUB: SourceRoute(
        sources={SourceType.CODE},
        weights={SourceType.CODE: 1.0},
        max_results_per_source=20,
    ),
    QueryIntent.LOCAL: SourceRoute(
        sources={SourceType.LOCAL},
        weights={SourceType.LOCAL: 1.0},
    ),
    QueryIntent.PRIVATE: SourceRoute(
        sources={SourceType.PRIVATE},
        weights={SourceType.PRIVATE: 1.0},
    ),
    QueryIntent.MATH: SourceRoute(
        sources={SourceType.WEB, SourceType.INDEXED},
        weights={SourceType.WEB: 1.3, SourceType.INDEXED: 1.0},
    ),
    QueryIntent.UNKNOWN: SourceRoute(
        sources={SourceType.WEB, SourceType.INDEXED},
        weights={SourceType.WEB: 1.0, SourceType.INDEXED: 1.0},
    ),
}


class SourceRouter:
    """
    Maps QueryIntent → SourceRoute.

    Supports JSON config file override via ROUTING_CONFIG_PATH env var.
    Disabled source types (via DISABLED_PROVIDERS env var) are filtered out
    so the fan-out never tries to contact unavailable backends.
    """

    def __init__(self) -> None:
        self._table = dict(_DEFAULT_ROUTING_TABLE)
        self._disabled: set[SourceType] = self._load_disabled_providers()
        self._load_config_override()

    def route(self, intent: QueryIntent) -> SourceRoute:
        """
        Return the SourceRoute for the given intent, with disabled
        providers filtered out.
        """
        route = self._table.get(intent, _DEFAULT_ROUTING_TABLE[QueryIntent.UNKNOWN])

        active_sources = route.sources - self._disabled
        if not active_sources:
            logger.warning(
                "All providers disabled for intent %s — falling back to WEB.",
                intent,
            )
            active_sources = {SourceType.WEB} - self._disabled
            if not active_sources:
                active_sources = {SourceType.INDEXED}

        return SourceRoute(
            sources=active_sources,
            weights={k: v for k, v in route.weights.items() if k in active_sources},
            max_results_per_source=route.max_results_per_source,
        )

    def _load_disabled_providers(self) -> set[SourceType]:
        """Read DISABLED_PROVIDERS=WEB,CODE from env."""
        raw = os.getenv("DISABLED_PROVIDERS", "")
        disabled = set()
        for name in raw.split(","):
            name = name.strip().upper()
            if name:
                try:
                    disabled.add(SourceType(name))
                except ValueError:
                    logger.warning("Unknown provider in DISABLED_PROVIDERS: %r", name)
        return disabled

    def _load_config_override(self) -> None:
        """Optionally load routing overrides from a JSON config file."""
        config_path = os.getenv("ROUTING_CONFIG_PATH", "")
        if not config_path:
            return
        try:
            with open(config_path) as f:
                overrides = json.load(f)
            for intent_str, cfg in overrides.items():
                try:
                    intent = QueryIntent(intent_str)
                    sources = {SourceType(s) for s in cfg.get("sources", [])}
                    weights = {SourceType(k): v for k, v in cfg.get("weights", {}).items()}
                    self._table[intent] = SourceRoute(
                        sources=sources,
                        weights=weights,
                        max_results_per_source=cfg.get("max_results_per_source", 10),
                    )
                    logger.info("Routing override applied for intent %s", intent)
                except (ValueError, KeyError) as e:
                    logger.warning("Invalid routing config entry: %s", e)
        except Exception as exc:
            logger.warning("Could not load routing config from %r: %s", config_path, exc)
