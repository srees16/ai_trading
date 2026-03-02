"""
TimescaleDB tick persistence and OHLC continuous aggregates.

Subscribes to TICK_BATCH events and writes raw ticks into a
``tick_data`` hypertable.  TimescaleDB continuous aggregates
materialise 1-min and 5-min OHLC bars automatically.

Setup
-----
Requires TimescaleDB extension on the PostgreSQL instance:

    CREATE EXTENSION IF NOT EXISTS timescaledb;

The ``ensure_schema()`` method creates the hypertable and
continuous aggregates idempotently.

Usage
-----
>>> handler = TimescaleTickHandler()
>>> handler.ensure_schema()         # one-time DDL
>>> # Then register with the dispatcher:
>>> dispatcher.subscribe("timescale_ticks", [EventType.TICK_BATCH], handler)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS timescaledb;"

_CREATE_TICK_TABLE = """
CREATE TABLE IF NOT EXISTS tick_data (
    ts          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    symbol      TEXT            NOT NULL,
    last_price  DOUBLE PRECISION,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT          DEFAULT 0,
    change_pct  DOUBLE PRECISION DEFAULT 0,
    oi          BIGINT          DEFAULT 0
);
"""

_CREATE_HYPERTABLE = """
SELECT create_hypertable(
    'tick_data', 'ts',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);
"""

_CREATE_INDEX_SYMBOL = """
CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_ts
ON tick_data (symbol, ts DESC);
"""

_CREATE_OHLC_1M = """
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlc_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts) AS bucket,
    symbol,
    first(last_price, ts)       AS open,
    max(last_price)             AS high,
    min(last_price)             AS low,
    last(last_price, ts)        AS close,
    sum(volume)                 AS volume,
    count(*)                    AS trade_count
FROM tick_data
GROUP BY bucket, symbol
WITH NO DATA;
"""

_CREATE_OHLC_5M = """
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlc_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', ts) AS bucket,
    symbol,
    first(last_price, ts)        AS open,
    max(last_price)              AS high,
    min(last_price)              AS low,
    last(last_price, ts)         AS close,
    sum(volume)                  AS volume,
    count(*)                     AS trade_count
FROM tick_data
GROUP BY bucket, symbol
WITH NO DATA;
"""

_CREATE_OHLC_15M = """
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlc_15m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 minutes', ts) AS bucket,
    symbol,
    first(last_price, ts)         AS open,
    max(last_price)               AS high,
    min(last_price)               AS low,
    last(last_price, ts)          AS close,
    sum(volume)                   AS volume,
    count(*)                      AS trade_count
FROM tick_data
GROUP BY bucket, symbol
WITH NO DATA;
"""

_CREATE_OHLC_1H = """
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlc_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)    AS bucket,
    symbol,
    first(last_price, ts)        AS open,
    max(last_price)              AS high,
    min(last_price)              AS low,
    last(last_price, ts)         AS close,
    sum(volume)                  AS volume,
    count(*)                     AS trade_count
FROM tick_data
GROUP BY bucket, symbol
WITH NO DATA;
"""

# Refresh policies — automatically refresh the last 2 intervals
_REFRESH_POLICIES = [
    ("ohlc_1m", "2 minutes", "2 minutes", "1 minute"),
    ("ohlc_5m", "10 minutes", "10 minutes", "5 minutes"),
    ("ohlc_15m", "30 minutes", "30 minutes", "15 minutes"),
    ("ohlc_1h", "2 hours", "2 hours", "1 hour"),
]

_ADD_REFRESH_POLICY = """
SELECT add_continuous_aggregate_policy('{view}',
    start_offset  => INTERVAL '{start}',
    end_offset    => INTERVAL '{end}',
    schedule_interval => INTERVAL '{interval}',
    if_not_exists => TRUE
);
"""

# Retention policy — keep raw tick data for 7 days
_RETENTION_POLICY = """
SELECT add_retention_policy('tick_data',
    INTERVAL '7 days',
    if_not_exists => TRUE
);
"""

# Insert query (batch)
_INSERT_TICK = """
INSERT INTO tick_data (ts, symbol, last_price, open, high, low, close, volume, change_pct, oi)
VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Query OHLC bars
_QUERY_OHLC = """
SELECT bucket, symbol, open, high, low, close, volume, trade_count
FROM {view}
WHERE symbol = %s
ORDER BY bucket DESC
LIMIT %s
"""


# ---------------------------------------------------------------------------
# Tick handler
# ---------------------------------------------------------------------------

class TimescaleTickHandler:
    """
    Writes tick batches to the ``tick_data`` hypertable.

    Throttled to at most one write per ``min_interval`` seconds.
    Pending ticks are buffered and flushed in bulk.
    """

    _OHLC_VIEWS = {
        "1m": "ohlc_1m",
        "5m": "ohlc_5m",
        "15m": "ohlc_15m",
        "1h": "ohlc_1h",
    }

    def __init__(self, min_interval: float = 0.5):
        self._min_interval = min_interval
        self._last_write: float = 0.0
        self._lock = threading.Lock()
        self._pending: List[Dict] = []
        self._schema_ready = False
        self._stats = {
            "ticks_received": 0,
            "ticks_written": 0,
            "write_errors": 0,
        }

    # ── Schema setup ───────────────────────────────────────────

    def ensure_schema(self) -> bool:
        """
        Create the hypertable and continuous aggregates.

        Idempotent — safe to call on every startup.
        Returns True if schema is ready, False on error.
        """
        if self._schema_ready:
            return True

        try:
            conn = self._get_connection()
            conn.autocommit = True
            cur = conn.cursor()

            # TimescaleDB extension
            try:
                cur.execute(_CREATE_EXTENSION)
            except Exception:
                logger.warning("TimescaleDB extension not available — using plain table")

            # Tick data table
            cur.execute(_CREATE_TICK_TABLE)

            # Hypertable (only works with TimescaleDB)
            try:
                cur.execute(_CREATE_HYPERTABLE)
                logger.info("Hypertable tick_data ready")
            except Exception:
                logger.warning("Hypertable creation skipped (TimescaleDB may not be installed)")

            # Index
            cur.execute(_CREATE_INDEX_SYMBOL)

            # Continuous aggregates
            for sql_template, name in [
                (_CREATE_OHLC_1M, "ohlc_1m"),
                (_CREATE_OHLC_5M, "ohlc_5m"),
                (_CREATE_OHLC_15M, "ohlc_15m"),
                (_CREATE_OHLC_1H, "ohlc_1h"),
            ]:
                try:
                    cur.execute(sql_template)
                    logger.info("Continuous aggregate %s ready", name)
                except Exception:
                    logger.warning("Aggregate %s skipped (TimescaleDB required)", name)

            # Refresh policies
            for view, start, end, interval in _REFRESH_POLICIES:
                try:
                    cur.execute(_ADD_REFRESH_POLICY.format(
                        view=view, start=start, end=end, interval=interval,
                    ))
                except Exception:
                    logger.debug("Refresh policy for %s skipped", view)

            # Retention
            try:
                cur.execute(_RETENTION_POLICY)
                logger.info("Retention policy set: 7 days")
            except Exception:
                logger.debug("Retention policy skipped")

            cur.close()
            conn.close()
            self._schema_ready = True
            logger.info("TimescaleDB schema setup complete")
            return True

        except Exception:
            logger.exception("TimescaleDB schema setup failed")
            return False

    # ── Dispatcher callback ────────────────────────────────────

    def __call__(self, event) -> None:
        """WebhookDispatcher callback entry point for TICK_BATCH."""
        from kite_connect.webhooks.events import EventType
        if event.event_type != EventType.TICK_BATCH:
            return

        ticks = event.payload.get("ticks", [])
        if not ticks:
            return

        with self._lock:
            self._pending.extend(ticks)
            self._stats["ticks_received"] += len(ticks)

        now = time.time()
        if now - self._last_write >= self._min_interval:
            self._flush()

    def _flush(self) -> None:
        """Write all pending ticks to the database."""
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()

        self._last_write = time.time()

        try:
            conn = self._get_connection()
            conn.autocommit = True
            cur = conn.cursor()

            for t in batch:
                cur.execute(
                    _INSERT_TICK,
                    (
                        t.get("name", ""),
                        t.get("ltp"),
                        t.get("open"),
                        t.get("high"),
                        t.get("low"),
                        t.get("close"),
                        t.get("volume", 0),
                        t.get("change_pct", 0),
                        t.get("oi", 0),
                    ),
                )

            self._stats["ticks_written"] += len(batch)
            cur.close()
            conn.close()
            logger.debug("TimescaleDB: %d ticks written", len(batch))

        except Exception:
            self._stats["write_errors"] += 1
            logger.exception("TimescaleDB tick write failed")

    # ── OHLC Query ─────────────────────────────────────────────

    def query_ohlc(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Query OHLC bars from a continuous aggregate.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. "RELIANCE").
        interval : str
            One of: "1m", "5m", "15m", "1h".
        limit : int
            Maximum number of bars to return.

        Returns
        -------
        list[dict]
            Each dict has: bucket, symbol, open, high, low, close, volume, trade_count
        """
        view = self._OHLC_VIEWS.get(interval)
        if not view:
            raise ValueError(f"Unsupported interval: {interval}. Use: {list(self._OHLC_VIEWS)}")

        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                _QUERY_OHLC.format(view=view),
                (symbol.upper(), limit),
            )
            columns = ["bucket", "symbol", "open", "high", "low", "close", "volume", "trade_count"]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
            cur.close()
            conn.close()
            # Return in chronological order
            rows.reverse()
            return rows

        except Exception:
            logger.exception("OHLC query failed: %s %s", symbol, interval)
            return []

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _get_connection():
        """
        Get a PostgreSQL connection.  Tries the Kite core DB service
        first, falls back to database.connection.
        """
        try:
            import os, sys
            _kite_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if _kite_root not in sys.path:
                sys.path.append(_kite_root)
            from kite_connect.core.db_service import get_connection
            return get_connection()
        except Exception:
            pass

        try:
            from database.connection import get_db_connection
            return get_db_connection()
        except Exception:
            pass

        # Last resort: psycopg2 direct
        import psycopg2
        from dotenv import load_dotenv
        import os
        load_dotenv()
        return psycopg2.connect(
            host=os.getenv("KITE_DB_HOST", os.getenv("DB_HOST", "localhost")),
            port=int(os.getenv("KITE_DB_PORT", os.getenv("DB_PORT", "9003"))),
            dbname=os.getenv("KITE_DB_NAME", os.getenv("DB_NAME", "centurion")),
            user=os.getenv("KITE_DB_USER", os.getenv("DB_USER", "")),
            password=os.getenv("KITE_DB_PASSWORD", os.getenv("DB_PASSWORD", "")),
        )

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
