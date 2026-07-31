"use client";

import { useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import SearchBar from "@/components/SearchBar/SearchBar";
import AnswerBox from "@/components/AnswerBox/AnswerBox";
import SourcesGrid from "@/components/SourceCard/SourceCard";
import ErrorBoundary from "@/components/ErrorBoundary/ErrorBoundary";
import { useSearchStore } from "@/lib/search-store";
import type { SearchMode } from "@/lib/types";

/**
 * Inner component — must be in its own file so it can be wrapped
 * in a <Suspense> boundary by the page shell (Next.js 15 requirement
 * for useSearchParams()).
 */
export default function SearchPageInner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const query = searchParams.get("q") ?? "";
  const mode = (searchParams.get("mode") ?? "web") as SearchMode;

  const {
    status,
    sources,
    answerChunks,
    citations,
    error,
    executeSearch,
  } = useSearchStore();

  // Trigger search on page load / param changes
  useEffect(() => {
    if (query) {
      executeSearch(query, mode);
    }
    // executeSearch is stable (Zustand), query/mode drive re-runs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, mode]);

  const handleSearch = (newQuery: string, newMode: SearchMode) => {
    router.push(`/search?q=${encodeURIComponent(newQuery)}&mode=${newMode}`);
  };

  const isSearching = status === "searching";
  const isStreaming = status === "streaming";
  const isDone = status === "done";

  return (
    <>
      {/* Sticky navbar */}
      <nav className="cs-navbar">
        <Link href="/" className="cs-navbar__logo" aria-label="CloudSearch home">
          <div className="cs-navbar__logo-mark" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <circle cx="10" cy="10" r="6" stroke="white" strokeWidth="2.5" />
              <path d="m15 15 4 4" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
          </div>
          <span className="cs-gradient-text">CloudSearch</span>
        </Link>

        <div className="cs-navbar__search">
          <SearchBar
            onSearch={handleSearch}
            initialQuery={query}
            initialMode={mode}
            loading={isSearching}
          />
        </div>
      </nav>

      {/* Main results layout */}
      <div className="cs-results-layout">
        {/* Left: answer + status */}
        <div className="cs-results-main">

          {/* Status bar */}
          {(isSearching || isStreaming) && (
            <div className="cs-status-bar" role="status" aria-live="polite">
              <span className="cs-status-dot searching" />
              {isSearching
                ? `Searching across ${mode} sources…`
                : `Generating answer from ${sources.length} source${sources.length !== 1 ? "s" : ""}…`}
            </div>
          )}

          {isDone && (
            <div className="cs-status-bar">
              <span className="cs-status-dot done" />
              <span>{sources.length} sources · {citations.length} citations</span>
            </div>
          )}

          {/* Answer box */}
          <AnswerBox
            answerChunks={answerChunks}
            citations={citations}
            sources={sources}
            isStreaming={isStreaming}
          />

          {/* Skeleton answer box while searching */}
          {isSearching && (
            <div className="cs-answer-box" aria-hidden="true">
              <div className="cs-section-header">
                <span className="cs-skeleton" style={{ width: 16, height: 16, borderRadius: "50%" }} />
                <span className="cs-skeleton" style={{ width: 80, height: 12 }} />
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", marginTop: "0.75rem" }}>
                {[100, 85, 92, 60].map((w, i) => (
                  <span key={i} className="cs-skeleton" style={{ width: `${w}%`, height: 14 }} />
                ))}
              </div>
            </div>
          )}

          {/* Error state */}
          {error && (
            <div className="cs-answer-box" role="alert" style={{ borderColor: "var(--cs-error)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--cs-error)" }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 8v4M12 16h.01" />
                </svg>
                <strong>Search error</strong>
              </div>
              <p style={{ marginTop: "0.5rem", color: "var(--cs-text-secondary)" }}>{error}</p>
              <button
                onClick={() => executeSearch(query, mode)}
                style={{
                  marginTop: "0.75rem",
                  padding: "0.4rem 1rem",
                  borderRadius: "var(--cs-radius-sm)",
                  border: "1px solid var(--cs-border-hover)",
                  background: "var(--cs-surface-raised)",
                  color: "var(--cs-text-primary)",
                  cursor: "pointer",
                  fontSize: "0.8rem",
                }}
              >
                ↻ Retry
              </button>
            </div>
          )}
        </div>

        {/* Right sidebar: sources */}
        <aside className="cs-results-sidebar" aria-label="Sources">
          <ErrorBoundary>
            <SourcesGrid
              sources={sources}
              isLoading={isSearching}
              skeletonCount={5}
            />
          </ErrorBoundary>
        </aside>
      </div>
    </>
  );
}
