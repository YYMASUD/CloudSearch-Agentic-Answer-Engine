"use client";

import { useState } from "react";

export interface MorphicCardProps {
  type: "code" | "entity" | "data" | "default";
  title: string;
  content: string;
  metadata?: Record<string, any>;
}

/**
 * Morphic Dynamic UI Card — Generative Streaming Artifact Component.
 * Part of Layer 1 (Client / Presentation) in the CloudSearch architecture.
 */
export default function MorphicCard({
  type,
  title,
  content,
  metadata = {},
}: MorphicCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (type === "code") {
    const language = metadata.language || "code";
    return (
      <div className="cs-morphic-card cs-code-card my-4 rounded-xl border border-[var(--color-border)] bg-[#0f172a] text-slate-100 overflow-hidden shadow-lg">
        <div className="flex items-center justify-between px-4 py-2 bg-[#1e293b] border-b border-slate-700/60 text-xs font-mono text-slate-300">
          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block" />
            {title} ({language})
          </span>
          <button
            onClick={handleCopy}
            className="px-2.5 py-1 rounded bg-slate-700/80 hover:bg-slate-600 transition-colors text-slate-200 text-xs font-sans cursor-pointer"
          >
            {copied ? "✓ Copied" : "Copy code"}
          </button>
        </div>
        <pre className="p-4 text-xs font-mono overflow-x-auto text-emerald-300/90 leading-relaxed whitespace-pre-wrap">
          <code>{content}</code>
        </pre>
      </div>
    );
  }

  if (type === "entity") {
    return (
      <div className="cs-morphic-card cs-entity-card my-3 p-4 rounded-xl border border-indigo-500/30 bg-indigo-950/20 text-indigo-100 shadow-md">
        <div className="flex items-center gap-2 mb-2 text-xs font-semibold uppercase tracking-wider text-indigo-400">
          <span className="px-2 py-0.5 rounded bg-indigo-500/20 border border-indigo-500/30">
            {metadata.entity_type || "GraphRAG Entity"}
          </span>
          <span>{title}</span>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">{content}</p>
        {metadata.source_count && (
          <div className="mt-2 text-xs text-indigo-300/70">
            Linked across {metadata.source_count} indexed sources
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="cs-morphic-card my-3 p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-card)] text-slate-200 shadow-sm">
      <h4 className="text-sm font-semibold text-slate-100 mb-1">{title}</h4>
      <p className="text-sm text-slate-300">{content}</p>
    </div>
  );
}
