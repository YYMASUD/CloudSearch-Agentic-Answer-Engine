"use client";

import React, { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * ErrorBoundary — catches React render crashes in the search results area.
 * Shows a styled fallback with a retry button instead of a blank/crashed screen.
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary] Caught render error:", error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div
          role="alert"
          style={{
            padding: "2rem",
            borderRadius: "var(--cs-radius-lg)",
            border: "1px solid var(--cs-error)",
            background: "hsla(0, 85%, 60%, 0.08)",
            color: "var(--cs-text-primary)",
            maxWidth: "560px",
            margin: "2rem auto",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.75rem" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
              stroke="var(--cs-error)" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 8v4M12 16h.01" />
            </svg>
            <strong style={{ color: "var(--cs-error)", fontSize: "0.9rem" }}>
              Something went wrong
            </strong>
          </div>
          <p style={{ fontSize: "0.82rem", color: "var(--cs-text-secondary)", marginBottom: "1rem" }}>
            {this.state.error?.message || "An unexpected error occurred while rendering results."}
          </p>
          <button
            onClick={this.handleRetry}
            style={{
              padding: "0.5rem 1.25rem",
              borderRadius: "var(--cs-radius-sm)",
              border: "1px solid var(--cs-border-hover)",
              background: "var(--cs-surface-raised)",
              color: "var(--cs-text-primary)",
              cursor: "pointer",
              fontSize: "0.82rem",
              transition: "background var(--cs-duration-fast)",
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
