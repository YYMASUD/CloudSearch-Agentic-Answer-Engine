/**
 * Shared TypeScript types mirroring the gateway's Pydantic schemas.
 */

export interface SourceCard {
  index: number;
  id: string;
  title: string;
  url: string;
  snippet: string;
  source_type: "INDEXED" | "WEB" | "CODE" | "PRIVATE" | "LOCAL" | "UNKNOWN";
  score: number;
  favicon_url: string;
  metadata?: Record<string, unknown>;
}

export interface CitationItem {
  number: number;
  claim?: string;
  url: string;
  title: string;
  snippet: string;
  confidence: number;
  matched_snippet?: string;
}

export interface SearchState {
  query: string;
  mode: SearchMode;
  status: "idle" | "searching" | "streaming" | "done" | "error";
  sources: SourceCard[];
  answerChunks: string[];
  citations: CitationItem[];
  error?: string;
  sessionId: string;
  fusionStats?: { total_raw: number; final_count: number; elapsed_ms: number };
}

export type SearchMode = "web" | "code" | "github" | "local" | "private";

/** SSE event types emitted by the gateway */
export type SSEEventType =
  | "source_card"
  | "answer_chunk"
  | "answer_done"
  | "citations"
  | "error";

export interface SSESourceCardEvent {
  type: "source_card";
  data: SourceCard;
}

export interface SSEAnswerChunkEvent {
  type: "answer_chunk";
  data: { chunk: string; chunk_index: number };
}

export interface SSEAnswerDoneEvent {
  type: "answer_done";
  data: {
    query: string;
    answer: string;
    citations: CitationItem[];
    source_count: number;
  };
}

export interface SSECitationsEvent {
  type: "citations";
  data: CitationItem[];
}

export interface SSEErrorEvent {
  type: "error";
  data: { message: string; recoverable?: boolean };
}

export type SSEEvent =
  | SSESourceCardEvent
  | SSEAnswerChunkEvent
  | SSEAnswerDoneEvent
  | SSECitationsEvent
  | SSEErrorEvent;
