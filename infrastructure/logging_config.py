"""
Structured Logging — JSON-formatted logs with correlation IDs.

Provides a consistent logging setup across all Centurion modules.
All log records include:
  - correlation_id  (from EventBus or request context)
  - module / component name
  - ISO-8601 timestamp
  - structured key-value context

Cloud sinks (opt-in via env vars):
  - **Better Stack / Logtail**: set ``LOGTAIL_TOKEN`` to ship all logs
    to Better Stack for search, dashboards, and alerting.

Usage::

    from infrastructure.logging_config import get_logger

    logger = get_logger("market_data")
    logger.info("Tick received", extra={"symbol": "RELIANCE.NS", "ltp": 2450.0})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Optional


# ── Context holder (thread-local) ────────────────────────────────────

_context = threading.local()


def set_correlation_id(cid: str) -> None:
    _context.correlation_id = cid


def get_correlation_id() -> str:
    return getattr(_context, "correlation_id", "")


# ── JSON Formatter ───────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Emit JSON log lines with standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }
        # Merge extra context
        for key in ("symbol", "ticker", "ltp", "order_id", "latency_ms",
                     "component", "market", "event_topic"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info and record.exc_info[0]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Setup ────────────────────────────────────────────────────────────

_INITIALISED = False


def setup_logging(
    *,
    level: int = logging.INFO,
    json_format: bool = False,
) -> None:
    """
    Configure root logger.  Call once at process start.

    Args:
        level: Root log level.
        json_format: If True, emit JSON lines (for production).
                     If False, use human-readable format (for dev).
    """
    global _INITIALISED
    if _INITIALISED:
        return
    _INITIALISED = True

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root.handlers = [handler]

    # ── Better Stack / Logtail (cloud log sink) ──────────────────────
    logtail_token = os.getenv("LOGTAIL_TOKEN", "")
    if logtail_token:
        try:
            from logtail import LogtailHandler

            lt_handler = LogtailHandler(source_token=logtail_token)
            lt_handler.setLevel(level)
            # Logtail handler sends structured JSON natively — no custom
            # formatter needed.  It captures record.msg, levelname, and
            # any `extra` fields automatically.
            root.addHandler(lt_handler)
            root.info("Logtail (Better Stack) handler attached")
        except ImportError:
            root.debug("logtail-python not installed — cloud logging disabled")
        except Exception as exc:
            root.warning("Logtail handler init failed: %s", exc)


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger."""
    return logging.getLogger(f"centurion.{name}")
