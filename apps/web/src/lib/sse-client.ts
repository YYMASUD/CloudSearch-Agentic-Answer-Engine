/**
 * SSE client — streams search results from the gateway.
 *
 * Uses native EventSource with custom event type parsing.
 * Returns a cleanup function to close the stream.
 */
import type {
  CitationItem,
  SearchMode,
  SourceCard,
  SSEAnswerDoneEvent,
} from "./types";

export interface SSEHandlers {
  onSourceCard: (card: SourceCard) => void;
  onAnswerChunk: (chunk: string, index: number) => void;
  onAnswerDone: (answer: string, citations: CitationItem[]) => void;
  onCitations: (citations: CitationItem[]) => void;
  onError: (message: string, recoverable: boolean) => void;
}

export function startSearchStream(
  query: string,
  mode: SearchMode,
  handlers: SSEHandlers,
  options?: { maxResults?: number; semanticRatio?: number; tenantId?: string }
): () => void {
  const params = new URLSearchParams({
    q: query,
    mode,
    max_results: String(options?.maxResults ?? 10),
    semantic_ratio: String(options?.semanticRatio ?? 0.5),
    tenant_id: options?.tenantId ?? "default",
  });

  const url = `/api/search/stream?${params}`;
  let es: EventSource | null = null;
  let closed = false;
  let retryCount = 0;
  const maxRetries = 3;
  const seenChunkIndexes = new Set<number>();

  const safeClose = () => {
    if (!closed) {
      closed = true;
      es?.close();
    }
  };

  const connect = () => {
    if (closed) return;
    es = new EventSource(url);

  es.addEventListener("source_card", (e: MessageEvent) => {
    try {
      handlers.onSourceCard(JSON.parse(e.data) as SourceCard);
    } catch {/* ignore parse errors */}
  });

  es.addEventListener("answer_chunk", (e: MessageEvent) => {
    try {
      const { chunk, chunk_index } = JSON.parse(e.data) as { chunk: string; chunk_index: number };
      // Deduplicate chunks on reconnect
      if (!seenChunkIndexes.has(chunk_index)) {
        seenChunkIndexes.add(chunk_index);
        handlers.onAnswerChunk(chunk, chunk_index);
      }
    } catch {/* ignore */}
  });

  es.addEventListener("answer_done", (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data) as SSEAnswerDoneEvent["data"];
      handlers.onAnswerDone(data.answer, data.citations);
    } catch {/* ignore */} finally {
      safeClose();
    }
  });

  es.addEventListener("citations", (e: MessageEvent) => {
    try {
      handlers.onCitations(JSON.parse(e.data) as CitationItem[]);
    } catch {/* ignore */}
  });

  // Named "error" event — emitted by the server as a typed SSE event
  es.addEventListener("error", (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data) as { message: string; recoverable?: boolean };
      handlers.onError(data.message ?? "Unknown error", data.recoverable ?? false);
    } catch {
      handlers.onError("Connection lost", false);
    }
    safeClose();
  });

  // Native EventSource connection error — auto-reconnect with backoff
  es.onerror = () => {
    if (closed) return;
    es?.close();
    if (retryCount < maxRetries) {
      retryCount++;
      const delay = Math.min(500 * Math.pow(2, retryCount - 1), 8000);
      console.warn(`[SSE] Connection lost, retrying (${retryCount}/${maxRetries}) in ${delay}ms...`);
      setTimeout(connect, delay);
    } else {
      handlers.onError("Stream connection failed after retries. Is the gateway running?", false);
      safeClose();
    }
  };
  }; // end connect()

  connect();

  // Return cleanup
  return safeClose;
}
