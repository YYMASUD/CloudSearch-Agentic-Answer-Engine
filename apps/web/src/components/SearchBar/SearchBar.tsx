"use client";

import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import type { SearchMode } from "@/lib/types";

const MODES: { id: SearchMode; label: string; icon: string }[] = [
  { id: "web",     label: "Web",     icon: "🌐" },
  { id: "code",    label: "Code",    icon: "💻" },
  { id: "github",  label: "GitHub",  icon: "🐙" },
  { id: "local",   label: "Local",   icon: "📁" },
];

interface SearchBarProps {
  onSearch: (query: string, mode: SearchMode) => void;
  initialQuery?: string;
  initialMode?: SearchMode;
  loading?: boolean;
  autoFocus?: boolean;
  size?: "normal" | "large";
}

export default function SearchBar({
  onSearch,
  initialQuery = "",
  initialMode = "web",
  loading = false,
  autoFocus = false,
  size = "normal",
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery);
  const [mode, setMode] = useState<SearchMode>(initialMode);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync internal state when prop changes (e.g. browser back/forward)
  useEffect(() => {
    setQuery(initialQuery);
  }, [initialQuery]);

  useEffect(() => {
    setMode(initialMode);
  }, [initialMode]);

  useEffect(() => {
    if (autoFocus) inputRef.current?.focus();
  }, [autoFocus]);

  // Keyboard shortcut: "/" focuses search from anywhere
  useEffect(() => {
    const handler = (e: globalThis.KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") inputRef.current?.blur();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const submit = () => {
    if (!query.trim() || loading) return;
    onSearch(query.trim(), mode);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") submit();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {/* Mode bar */}
      <div className="cs-mode-bar" role="group" aria-label="Search mode">
        {MODES.map(({ id, label, icon }) => (
          <button
            key={id}
            id={`mode-pill-${id}`}
            className={`cs-mode-pill${mode === id ? " active" : ""}`}
            onClick={() => setMode(id)}
            aria-pressed={mode === id}
            aria-label={`${label} mode`}
          >
            <span aria-hidden="true">{icon}</span>
            {label}
          </button>
        ))}
      </div>

      {/* Search input */}
      <div
        className="cs-search-bar"
        style={{ padding: size === "large" ? "0 1rem 0 1.5rem" : "0 0.75rem 0 1.25rem" }}
      >
        <svg
          className="cs-search-icon"
          width="18" height="18" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="2"
          strokeLinecap="round" strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>

        <input
          ref={inputRef}
          id="search-input"
          className="cs-search-input"
          type="search"
          placeholder={`Ask anything… (press / to focus)`}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          autoComplete="off"
          spellCheck={false}
          aria-label="Search query"
          style={{ fontSize: size === "large" ? "1.1rem" : "1rem" }}
          disabled={loading}
        />

        <button
          id="search-submit-btn"
          className="cs-search-btn"
          onClick={submit}
          disabled={loading || !query.trim()}
          aria-label="Submit search"
        >
          {loading ? (
            <>
              <span className="cs-spinner" aria-hidden="true" />
              Searching…
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
                strokeLinejoin="round" aria-hidden="true">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
              Search
            </>
          )}
        </button>
      </div>
    </div>
  );
}
