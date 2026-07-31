"""
Intent planner — classifies user queries and rewrites them for retrieval.

Two-stage pipeline:
  1. Rule-based fast classification (regex + keyword heuristics)
  2. LLM-assisted rewrite to expand acronyms, normalize phrasing, add context

QueryIntent drives which SearchProvider backends are activated.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTUAL    = "factual"     # "What is X?" — web + indexed
    RESEARCH   = "research"    # "How does X work?" — web + indexed + RAG
    CODE       = "code"        # Code snippets, debugging, APIs — code provider
    GITHUB     = "github"      # Repo-specific queries — GitHub mode
    LOCAL      = "local"       # Offline / private queries — local provider
    PRIVATE    = "private"     # Tenant knowledge base
    MATH       = "math"        # Mathematical / computational queries
    UNKNOWN    = "unknown"     # Fallback — all providers


@dataclass
class PlannerResult:
    """Output of the intent planner."""
    original_query: str
    rewritten_query: str
    intent: QueryIntent
    confidence: float
    mode_override: str | None = None   # explicit user mode (from UI toggle)
    metadata: dict[str, Any] | None = None


# ─── Rule-based classifiers ───────────────────────────────────────────────────

_CODE_PATTERNS = [
    r"\b(def |class |import |function|async def|const |let |var |fn |pub fn)\b",
    r"\b(TypeError|AttributeError|SyntaxError|NullPointerException|segfault)\b",
    r"\bhow (to|do I|can I) (write|implement|code|build|create|fix)\b",
    r"\b(python|javascript|typescript|rust|golang|java|c\+\+|bash|sql) (code|snippet|example|function)\b",
    r"```",
    r"\bAPI (endpoint|key|call|request|response)\b",
    r"\b(git|docker|kubernetes|npm|pip|cargo|gradle)\b",
    r"\b(stack ?overflow|github\.com)\b",
]

_GITHUB_PATTERNS = [
    r"\bgithub\.com/[\w\-]+/[\w\-]+\b",
    r"\b(pull request|PR #\d+|issue #\d+|commit [0-9a-f]{7,})\b",
    r"\b(repo|repository|branch|fork|clone|star)\b.{0,20}\b(github|gitlab|bitbucket)\b",
    r"github (mode|search|repo)",
]

_LOCAL_PATTERNS = [
    r"\b(my|our|internal|private|company|team|local|offline)\b.{0,30}\b(docs?|documents?|files?|notes?)\b",
    r"\blocal (search|mode|file)\b",
]

_MATH_PATTERNS = [
    r"\b(calculate|compute|solve|integrate|differentiate|equation)\b",
    r"\b\d+\s*[\+\-\*/\^]\s*\d+\b",
    r"\b(formula|theorem|proof|algebra|calculus|probability)\b",
]


def classify_intent(query: str, mode_override: str | None = None) -> QueryIntent:
    """
    Rule-based intent classification.

    Args:
        query:         User's raw query string.
        mode_override: Explicit mode from the UI toggle (code/github/local/private).

    Returns:
        QueryIntent enum value.
    """
    # Explicit mode from UI always wins
    if mode_override:
        mode_map = {
            "code":    QueryIntent.CODE,
            "github":  QueryIntent.GITHUB,
            "local":   QueryIntent.LOCAL,
            "private": QueryIntent.PRIVATE,
        }
        if mode_override in mode_map:
            return mode_map[mode_override]

    q = query.strip()

    for pattern in _GITHUB_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return QueryIntent.GITHUB

    for pattern in _CODE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return QueryIntent.CODE

    for pattern in _LOCAL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return QueryIntent.LOCAL

    for pattern in _MATH_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return QueryIntent.MATH

    # Research vs factual: research queries tend to be longer + use "how/why/explain"
    if re.search(r"\b(how|why|explain|describe|what is the difference|compare|overview)\b", q, re.IGNORECASE):
        if len(q.split()) > 6:
            return QueryIntent.RESEARCH

    if re.search(r"\bwhat (is|are|was|were)\b", q, re.IGNORECASE):
        return QueryIntent.FACTUAL

    return QueryIntent.UNKNOWN


def rewrite_query(query: str, intent: QueryIntent) -> str:
    """
    Simple rule-based query rewriting/normalization.

    For each intent, applies expansion heuristics:
    - Expand abbreviations
    - Add domain context words
    - Normalize question phrasing

    A full LLM-assisted rewrite can be plugged in here later.
    """
    q = query.strip()

    if intent == QueryIntent.CODE:
        # Ensure "code" context if not already present
        if not re.search(r"\b(code|implementation|example|snippet)\b", q, re.IGNORECASE):
            q = f"{q} code example implementation"

    elif intent == QueryIntent.RESEARCH:
        # Expand "how does" → include "explanation overview"
        q = re.sub(r"^(how does|how do)\s+", "", q, flags=re.IGNORECASE).strip()
        q = f"{q} explanation overview guide"

    elif intent == QueryIntent.GITHUB:
        if not re.search(r"github", q, re.IGNORECASE):
            q = f"github {q}"

    # Strip trailing punctuation
    q = q.rstrip("?.!")

    return q


class Planner:
    """
    Orchestrates intent detection and query rewriting.

    The planner is stateless and re-entrant — safe to call concurrently.
    """

    def plan(self, query: str, mode_override: str | None = None) -> PlannerResult:
        """
        Classify intent and rewrite the query.

        Args:
            query:         Raw user query.
            mode_override: Explicit mode from UI toggle.

        Returns:
            PlannerResult with original + rewritten query, intent, confidence.
        """
        intent = classify_intent(query, mode_override)
        rewritten = rewrite_query(query, intent)
        confidence = 0.9 if mode_override else 0.7  # explicit mode = high confidence

        logger.debug(
            "Planner: query=%r intent=%s rewritten=%r",
            query[:80], intent, rewritten[:80],
        )

        return PlannerResult(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            confidence=confidence,
            mode_override=mode_override,
        )
