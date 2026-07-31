/**
 * Global search state store using Zustand.
 * Manages the entire search lifecycle from query to answer + citations.
 */
import { create } from "zustand";
import type { CitationItem, SearchMode, SearchState, SourceCard } from "@/lib/types";
import { startSearchStream } from "@/lib/sse-client";

interface SearchStore extends SearchState {
  // Actions
  setQuery: (query: string) => void;
  setMode: (mode: SearchMode) => void;
  executeSearch: (query: string, mode: SearchMode) => void;
  reset: () => void;
  _cleanup: (() => void) | null;
}

/** SSR-safe UUID — returns empty string during server render, real UUID on client. */
function newSessionId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

const initialState: SearchState = {
  query: "",
  mode: "web",
  status: "idle",
  sources: [],
  answerChunks: [],
  citations: [],
  sessionId: "",
};

export const useSearchStore = create<SearchStore>((set, get) => ({
  ...initialState,
  _cleanup: null,

  setQuery: (query) => set({ query }),
  setMode: (mode) => set({ mode }),

  reset: () => {
    const { _cleanup } = get();
    if (_cleanup) _cleanup();
    set({ ...initialState, sessionId: newSessionId(), _cleanup: null });
  },

  executeSearch: (query: string, mode: SearchMode) => {
    // Cancel any existing stream
    const { _cleanup } = get();
    if (_cleanup) _cleanup();

    set({
      query,
      mode,
      status: "searching",
      sources: [],
      answerChunks: [],
      citations: [],
      error: undefined,
      sessionId: newSessionId(),
    });

    const cleanup = startSearchStream(query, mode, {
      onSourceCard: (card: SourceCard) => {
        set((state) => ({
          sources: [...state.sources, card],
          status: "streaming",
        }));
      },

      onAnswerChunk: (chunk: string) => {
        set((state) => ({
          answerChunks: [...state.answerChunks, chunk],
          status: "streaming",
        }));
      },

      onAnswerDone: (_answer: string, citations: CitationItem[]) => {
        set({ citations, status: "done", _cleanup: null });
      },

      onCitations: (citations: CitationItem[]) => {
        set({ citations });
      },

      onError: (message: string) => {
        set({ error: message, status: "error", _cleanup: null });
      },
    });

    set({ _cleanup: cleanup });
  },
}));
