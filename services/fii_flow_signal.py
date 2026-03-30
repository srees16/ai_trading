"""
FII Daily Flow Signal — Gap A5.

Uses NSE FII/DII daily trading data as a leading indicator for
next-day price direction. FII flows are the strongest predictor
of short-term returns in Indian equities.

Research basis:
  - Mukherjee et al. (2002): FII flows Granger-cause NIFTY returns
  - Ananthanarayanan et al. (2004): FII trades are information-motivated
  - Sehgal & Tripathi (2009): FII momentum predicts 5–10 day returns

Signal logic:
  1. FII net buy (3-day rolling) > ₹1000 Cr → bullish (positive forecast)
  2. FII net sell (3-day rolling) < -₹1000 Cr → bearish (zero forecast for long-only)
  3. DII net buy during FII selling → accumulation (mildly bullish)

Integration:
  - Generates a market-wide sentiment forecast (same for all stocks)
  - Weight: 5–8% of combined forecast
  - Source: NSE daily FII/DII data or cached values
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_FII_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "fii_flow_cache.json"
FORECAST_CAP = 20.0


@dataclass
class FIIFlowSnapshot:
    """Point-in-time FII/DII flow data."""
    date: str
    fii_net_cr: float          # FII net buy/sell in ₹ Crores
    dii_net_cr: float          # DII net buy/sell in ₹ Crores
    fii_3d_avg: float          # 3-day rolling average FII net
    dii_3d_avg: float          # 3-day rolling average DII net
    fii_5d_avg: float          # 5-day rolling average FII net
    forecast: float            # Carver-scale forecast
    signal: str                # "FII_BULLISH", "FII_BEARISH", "ACCUMULATION", "NEUTRAL"


def compute_fii_forecast(
    fii_daily_net: list[float],
    dii_daily_net: list[float],
    fii_bullish_threshold: float = 1000.0,
    fii_bearish_threshold: float = -1000.0,
) -> FIIFlowSnapshot:
    """Compute FII flow-based forecast from recent daily flows.

    Parameters
    ----------
    fii_daily_net : list[float]
        Last N days of FII net buy/sell in ₹ Crores (most recent last).
    dii_daily_net : list[float]
        Last N days of DII net buy/sell in ₹ Crores.
    fii_bullish_threshold : float
        3-day avg FII net above this = bullish signal (₹ Crores).
    fii_bearish_threshold : float
        3-day avg FII net below this = bearish signal.

    Returns
    -------
    FIIFlowSnapshot
    """
    if not fii_daily_net or len(fii_daily_net) < 3:
        return FIIFlowSnapshot(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            fii_net_cr=0.0, dii_net_cr=0.0,
            fii_3d_avg=0.0, dii_3d_avg=0.0, fii_5d_avg=0.0,
            forecast=0.0, signal="NEUTRAL",
        )

    fii_arr = np.array(fii_daily_net[-10:])
    dii_arr = np.array(dii_daily_net[-10:]) if dii_daily_net else np.zeros_like(fii_arr)

    fii_3d = float(np.mean(fii_arr[-3:])) if len(fii_arr) >= 3 else float(fii_arr[-1])
    dii_3d = float(np.mean(dii_arr[-3:])) if len(dii_arr) >= 3 else float(dii_arr[-1])
    fii_5d = float(np.mean(fii_arr[-5:])) if len(fii_arr) >= 5 else fii_3d
    fii_latest = float(fii_arr[-1])
    dii_latest = float(dii_arr[-1])

    forecast = 0.0
    signal = "NEUTRAL"

    # Strong FII buying
    if fii_3d > fii_bullish_threshold:
        strength = min(1.0, fii_3d / (fii_bullish_threshold * 3))
        forecast = strength * FORECAST_CAP
        signal = "FII_BULLISH"

    # Strong FII selling
    elif fii_3d < fii_bearish_threshold:
        # For long-only: dampen to 0 instead of negative
        # But if DII is buying (accumulation), give mild positive
        if dii_3d > abs(fii_3d) * 0.5:
            forecast = 3.0  # mild positive (DII accumulation)
            signal = "ACCUMULATION"
        else:
            forecast = 0.0
            signal = "FII_BEARISH"

    # Moderate FII buying with trend
    elif fii_3d > 0 and fii_5d > 0:
        strength = min(1.0, fii_3d / fii_bullish_threshold)
        forecast = strength * FORECAST_CAP * 0.5
        signal = "FII_MILD_BULLISH"

    # FII selling but DII strongly buying
    elif fii_3d < 0 and dii_3d > fii_bullish_threshold:
        forecast = 4.0
        signal = "ACCUMULATION"

    forecast = max(0.0, min(FORECAST_CAP, forecast))

    return FIIFlowSnapshot(
        date=datetime.utcnow().strftime("%Y-%m-%d"),
        fii_net_cr=round(fii_latest, 2),
        dii_net_cr=round(dii_latest, 2),
        fii_3d_avg=round(fii_3d, 2),
        dii_3d_avg=round(dii_3d, 2),
        fii_5d_avg=round(fii_5d, 2),
        forecast=round(forecast, 2),
        signal=signal,
    )


def get_fii_flow_forecasts(
    symbols: list[str],
    fii_daily_net: Optional[list[float]] = None,
    dii_daily_net: Optional[list[float]] = None,
) -> Dict[str, float]:
    """Return FII flow forecast for all symbols.

    The FII flow is a market-wide signal — same forecast for all stocks.
    Returns {symbol: forecast} for Carver combiner integration.

    If live data not provided, attempts to load from cache.
    """
    if fii_daily_net is None:
        fii_daily_net, dii_daily_net = _load_cached_flows()

    snap = compute_fii_forecast(
        fii_daily_net or [],
        dii_daily_net or [],
    )

    # Persist for next use
    _persist_flow_cache(snap)

    if snap.forecast == 0.0:
        return {}

    return {sym: snap.forecast for sym in symbols}


def fetch_nse_fii_data() -> tuple:
    """Fetch FII/DII daily data from NSE website.

    Returns (fii_daily_net, dii_daily_net) as lists of ₹ Crores.
    Falls back to cached data on failure.
    """
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        # NSE FII/DII activity endpoint
        url = "https://www.nseindia.com/api/fiidiiActivity"
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            fii_net = []
            dii_net = []
            for entry in data.get("data", []):
                if "FII" in str(entry.get("category", "")):
                    fii_net.append(float(entry.get("netValue", 0)))
                elif "DII" in str(entry.get("category", "")):
                    dii_net.append(float(entry.get("netValue", 0)))
            return fii_net, dii_net
    except Exception as exc:
        logger.debug("NSE FII fetch failed: %s", exc)

    return _load_cached_flows()


def _persist_flow_cache(snap: FIIFlowSnapshot) -> None:
    """Cache latest FII flow snapshot to disk."""
    try:
        data = {
            "date": snap.date,
            "fii_net_cr": snap.fii_net_cr,
            "dii_net_cr": snap.dii_net_cr,
            "fii_3d_avg": snap.fii_3d_avg,
            "dii_3d_avg": snap.dii_3d_avg,
            "forecast": snap.forecast,
            "signal": snap.signal,
        }
        _FII_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FII_CACHE_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        logger.debug("FII cache persist failed: %s", exc)


def _load_cached_flows() -> tuple:
    """Load cached FII/DII flows from disk."""
    if not _FII_CACHE_PATH.exists():
        return [], []
    try:
        data = json.loads(_FII_CACHE_PATH.read_text())
        # Return single-element lists from cache
        fii = [data.get("fii_net_cr", 0.0)]
        dii = [data.get("dii_net_cr", 0.0)]
        return fii, dii
    except Exception:
        return [], []
