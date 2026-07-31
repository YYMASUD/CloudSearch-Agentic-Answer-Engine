"use client";

import { useState, useRef, useEffect } from "react";
import type { CitationItem, SourceCard } from "@/lib/types";
import MorphicCard from "@/components/GenerativeUI/MorphicCard";


interface CitationProps {
  number: number;
  citations: CitationItem[];
  sources: SourceCard[];
}

/**
 * Inline citation superscript with hover tooltip.
 * Renders [N] with a popup showing the source title, snippet, and URL.
 */
export function Citation({ number, citations, sources }: CitationProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const citation = citations.find((c) => c.number === number);
  const source = sources.find((s) => s.index === number);
  const title = citation?.title ?? source?.title ?? `Source ${number}`;
  const snippet = citation?.snippet ?? source?.snippet ?? "";
  const url = citation?.url ?? source?.url ?? "#";

  useEffect(() => {
    return () => clearTimeout(timer.current);
  }, []);

  const handleMouseEnter = () => {
    clearTimeout(timer.current);
    setOpen(true);
  };

  const handleMouseLeave = () => {
    timer.current = setTimeout(() => setOpen(false), 200);
  };

  const domain = (() => {
    try { return new URL(url).hostname; } catch { return url; }
  })();

  return (
    <span
      ref={ref}
      className="cs-citation"
      role="button"
      tabIndex={0}
      aria-label={`Citation ${number}: ${title}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleMouseEnter}
      onBlur={handleMouseLeave}
      onClick={() => window.open(url, "_blank", "noopener")}
      onKeyDown={(e) => e.key === "Enter" && window.open(url, "_blank", "noopener")}
    >
      {number}

      {open && (
        <span className="cs-citation-tooltip" role="tooltip">
          <span className="cs-citation-tooltip__title">{title}</span>
          {snippet && (
            <span className="cs-citation-tooltip__snippet">{snippet}</span>
          )}
          <span className="cs-citation-tooltip__url">{domain}</span>
        </span>
      )}
    </span>
  );
}

interface AnswerBoxProps {
  answerChunks: string[];
  citations: CitationItem[];
  sources: SourceCard[];
  isStreaming: boolean;
}

/**
 * Renders the streamed LLM answer with inline citation markers.
 * Parses [N] markers and replaces them with interactive Citation components.
 */
export default function AnswerBox({
  answerChunks,
  citations,
  sources,
  isStreaming,
}: AnswerBoxProps) {
  const fullAnswer = answerChunks.join("");

  if (!fullAnswer && !isStreaming) return null;

  return (
    <article className="cs-answer-box" aria-label="AI-generated answer">
      <div className="cs-section-header">
        <svg className="cs-section-header__icon" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          aria-hidden="true">
          <path d="M12 2a10 10 0 1 0 10 10" />
          <path d="M12 8v4l3 3" />
        </svg>
        Answer
        {isStreaming && (
          <span style={{ marginLeft: "auto", color: "var(--cs-accent-2)", fontSize: "0.72rem" }}>
            ● Generating…
          </span>
        )}
      </div>

      <div className="cs-answer-prose" role="region" aria-label="Generated answer text">
        <AnswerRenderer
          text={fullAnswer}
          isStreaming={isStreaming}
          citations={citations}
          sources={sources}
        />
      </div>
    </article>
  );
}

interface AnswerRendererProps {
  text: string;
  isStreaming: boolean;
  citations: CitationItem[];
  sources: SourceCard[];
}

/** Splits answer text into segments and renders [N] as Citation components, code blocks as MorphicCards. */
function AnswerRenderer({ text, isStreaming, citations, sources }: AnswerRendererProps) {
  // Split on fenced code blocks first: ```lang\ncode\n```
  const segments = text.split(/(```[\s\S]*?```)/g);

  return (
    <>
      {segments.map((segment, segIdx) => {
        // Detect fenced code blocks
        const codeMatch = segment.match(/^```(\w*)\n?([\s\S]*?)```$/);
        if (codeMatch) {
          const lang = codeMatch[1] || "code";
          const code = codeMatch[2].trim();
          return (
            <MorphicCard
              key={`morphic-${segIdx}`}
              type="code"
              title="Code Artifact"
              content={code}
              metadata={{ language: lang }}
            />
          );
        }

        // For non-code segments, parse citation markers [N]
        const parts = segment.split(/(\[\d+\])/g);
        return (
          <p key={`seg-${segIdx}`} style={{ margin: 0 }}>
            {parts.map((part, i) => {
              const match = part.match(/^\[(\d+)\]$/);
              if (match) {
                const num = parseInt(match[1], 10);
                return (
                  <Citation key={`${segIdx}-${i}`} number={num} citations={citations} sources={sources} />
                );
              }
              return <span key={`${segIdx}-${i}`}>{part}</span>;
            })}
          </p>
        );
      })}
      {isStreaming && <span className="cs-streaming-cursor" aria-hidden="true" />}
    </>
  );
}

