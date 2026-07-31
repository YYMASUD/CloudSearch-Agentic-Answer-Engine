"""
Structured JSON logging configuration with OpenTelemetry trace context injection.

Usage:
    from cloudsearch_shared.logging_config import configure_logging
    configure_logging(service_name="my-service")

LOG_FORMAT env var: "json" (production) | "text" (development, default)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON with trace context."""

    def __init__(self, service_name: str = "cloudsearch") -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        # Attempt to read OTel trace context if available
        trace_id = ""
        span_id = ""
        try:
            from opentelemetry import trace
            ctx = trace.get_current_span().get_span_context()
            if ctx and ctx.is_valid:
                trace_id = format(ctx.trace_id, "032x")
                span_id = format(ctx.span_id, "016x")
        except Exception:
            pass

        log_record: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "message": record.getMessage(),
        }
        if trace_id:
            log_record["trace_id"] = trace_id
            log_record["span_id"] = span_id
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via `extra={}`
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName",
            } and not key.startswith("_"):
                log_record[key] = value

        return json.dumps(log_record, default=str)


def configure_logging(
    service_name: str = "cloudsearch",
    level: int | None = None,
) -> None:
    """
    Configure root logger with JSON or text formatting based on LOG_FORMAT env var.
    Call once at application startup before any handlers are registered.
    """
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_level = level or getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if log_format == "json":
        handler.setFormatter(_JSONFormatter(service_name=service_name))
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt=f"%(asctime)s %(levelname)-8s [{service_name}] %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: format=%s level=%s service=%s",
        log_format, logging.getLevelName(log_level), service_name,
    )
