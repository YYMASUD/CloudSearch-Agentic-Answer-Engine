"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import SearchBar from "@/components/SearchBar/SearchBar";

const SUGGESTIONS = [
  "How does retrieval augmented generation work?",
  "What is the difference between BM25 and vector search?",
  "Explain Reciprocal Rank Fusion",
  "How to implement streaming with SSE in Next.js",
  "What is GraphRAG?",
  "Compare Ollama vs OpenAI for local LLMs",
];

export default function HomePage() {
  const router = useRouter();
  const [isNavigating, setIsNavigating] = useState(false);

  const handleSearch = (query: string, mode: string) => {
    if (!query.trim()) return;
    setIsNavigating(true);
    router.push(`/search?q=${encodeURIComponent(query)}&mode=${mode}`);
  };

  return (
    <main className="cs-home">
      {/* Logo */}
      <div className="cs-home__logo">
        <div className="cs-home__logo-mark" aria-hidden="true">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
            <circle cx="10" cy="10" r="6" stroke="white" strokeWidth="2" />
            <path d="m15 15 4 4" stroke="white" strokeWidth="2" strokeLinecap="round" />
            <path d="M10 7v3m0 0v3m0-3h3m-3 0H7" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
        <h1 className="cs-gradient-text">CloudSearch</h1>
      </div>

      <p className="cs-home__tagline">
        Agentic answer engine with citation-grounded responses.
        Multi-source intelligence, one answer.
      </p>

      {/* Search */}
      <div className="cs-home__search-wrap">
        <SearchBar
          autoFocus
          onSearch={handleSearch}
          loading={isNavigating}
          size="large"
        />

        {/* Suggestion chips */}
        <div className="cs-home__suggestions" aria-label="Suggested queries">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              className="cs-suggestion-chip"
              onClick={() => handleSearch(s, "web")}
              aria-label={`Search for: ${s}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Feature tags */}
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", justifyContent: "center", marginTop: "0.5rem" }}>
        {["5 Search Sources", "Inline Citations", "Streaming Answers", "Local LLM Support"].map((tag) => (
          <span key={tag} style={{
            fontSize: "0.75rem", color: "var(--cs-text-muted)",
            border: "1px solid var(--cs-border)", borderRadius: "999px",
            padding: "0.2rem 0.7rem"
          }}>
            {tag}
          </span>
        ))}
      </div>
    </main>
  );
}
