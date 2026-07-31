"use client";

import { useState } from "react";
import type { SourceCard as SourceCardType } from "@/lib/types";

const SOURCE_TYPE_LABELS: Record<string, string> = {
  WEB: "Web",
  INDEXED: "Indexed",
  CODE: "Code",
  PRIVATE: "Private",
  LOCAL: "Local",
  UNKNOWN: "Source",
};

/** Favicon using plain <img> — avoids next/image domain whitelist for arbitrary URLs. */
function FaviconImg({ url }: { url: string }) {
  const [errored, setErrored] = useState(false);
  if (!url || errored) {
    return (
      <span
        className="cs-source-card__favicon"
        style={{ background: "var(--cs-accent-1)", display: "inline-block", borderRadius: "3px" }}
        aria-hidden="true"
      />
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={url}
      alt=""
      width={16}
      height={16}
      className="cs-source-card__favicon"
      onError={() => setErrored(true)}
    />
  );
}

interface SourceCardProps {
  source: SourceCardType;
  isLoading?: boolean;
}

/** Individual source card with favicon, title, snippet, and source-type badge. */
export function SourceCard({ source, isLoading }: SourceCardProps) {
  if (isLoading) return <SkeletonCard />;

  const domain = (() => {
    try { return new URL(source.url).hostname.replace("www.", ""); }
    catch { return source.url; }
  })();

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="cs-source-card"
      aria-label={`Source ${source.index}: ${source.title}`}
    >
      <div className="cs-source-card__header">
        <FaviconImg url={source.favicon_url} />
        <span className="cs-source-card__domain">{domain}</span>
        <span style={{ marginLeft: "auto", flexShrink: 0 }}>
          <span className="cs-source-card__badge">
            {source.index}
          </span>
        </span>
      </div>

      <div className="cs-source-card__title">{source.title}</div>
      <div className="cs-source-card__snippet">{source.snippet}</div>

      <div style={{ marginTop: "auto", paddingTop: "0.35rem" }}>
        <span className="cs-source-card__badge">
          {SOURCE_TYPE_LABELS[source.source_type] ?? source.source_type}
        </span>
      </div>
    </a>
  );
}

/** Skeleton loading card for the pre-answer state. */
function SkeletonCard() {
  return (
    <div className="cs-source-card" style={{ cursor: "default", pointerEvents: "none" }}>
      <div className="cs-source-card__header">
        <span className="cs-skeleton" style={{ width: 16, height: 16, borderRadius: "3px" }} />
        <span className="cs-skeleton" style={{ width: "60%", height: 10 }} />
      </div>
      <span className="cs-skeleton" style={{ height: 12, width: "90%" }} />
      <span className="cs-skeleton" style={{ height: 10, width: "70%" }} />
    </div>
  );
}

interface SourcesGridProps {
  sources: SourceCardType[];
  isLoading?: boolean;
  skeletonCount?: number;
}

/** Grid of source cards, with skeleton loading state. */
export default function SourcesGrid({ sources, isLoading, skeletonCount = 4 }: SourcesGridProps) {
  if (isLoading && sources.length === 0) {
    return (
      <div>
        <div className="cs-section-header">
          <svg className="cs-section-header__icon" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            aria-hidden="true">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
          </svg>
          Sources
        </div>
        <div className="cs-sources-grid">
          {Array.from({ length: skeletonCount }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (sources.length === 0) return null;

  return (
    <div>
      <div className="cs-section-header" aria-label={`${sources.length} sources`}>
        <svg className="cs-section-header__icon" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          aria-hidden="true">
          <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
          <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
        </svg>
        Sources
        <span style={{ color: "var(--cs-text-muted)", fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>
          ({sources.length})
        </span>
      </div>
      <div className="cs-sources-grid">
        {sources.map((source) => (
          <SourceCard key={source.id} source={source} />
        ))}
      </div>
    </div>
  );
}
