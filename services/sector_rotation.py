"""
Sector Rotation Overlay.

Ranks NIFTY sectoral indices by 1-month momentum and identifies
the top 3 sectors. Stocks in top sectors get a score boost,
stocks in bottom sectors get a penalty.

Integration points:
  - NSE screener ``_compute_score()`` → sector momentum bonus/penalty
  - IntegratedScorer → sector rotation weight adjustment
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_CACHE: Optional["SectorRotation"] = None
_CACHE_TS: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=1)

# NIFTY sectoral index tickers for yfinance
SECTOR_INDEX_TICKERS = {
    "NIFTY IT": "^CNXIT",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY PSE": "^CNXPSE",
    "NIFTY FINANCIAL": "^CNXFIN",
}


@dataclass
class SectorMomentum:
    """Momentum data for a single sector."""
    sector_name: str
    return_1m: float = 0.0      # 1-month return
    return_3m: float = 0.0      # 3-month return (for confirmation)
    rank: int = 0               # 1 = best sector
    tier: str = "MID"           # TOP / MID / BOTTOM


@dataclass
class SectorRotation:
    """Complete sector rotation analysis."""
    sectors: Dict[str, SectorMomentum]
    top_sectors: List[str]       # top 3
    bottom_sectors: List[str]    # bottom 3
    analysis_date: str = ""

    def get_sector_tier(self, sector_name: str) -> str:
        """Get tier for a sector name."""
        sm = self.sectors.get(sector_name)
        return sm.tier if sm else "MID"

    def get_score_adjustment(self, sector_name: str) -> float:
        """Score adjustment for screener: +8 for TOP, -3 for BOTTOM."""
        tier = self.get_sector_tier(sector_name)
        if tier == "TOP":
            return 8.0
        if tier == "BOTTOM":
            return -3.0
        return 0.0

    def get_weight_multiplier(self, sector_name: str) -> float:
        """Weight multiplier for IntegratedScorer: 1.10 for TOP, 0.92 for BOTTOM."""
        tier = self.get_sector_tier(sector_name)
        if tier == "TOP":
            return 1.10
        if tier == "BOTTOM":
            return 0.92
        return 1.0


def _compute_sector_rotation(
    sector_indices: Dict[str, List[str]],
    ohlcv_cache: Dict[str, object],
) -> SectorRotation:
    """Compute sector rotation from constituent OHLCV data.

    Falls back to constituent-level returns if sectoral index
    data isn't available (which is common with yfinance for NSE).
    """
    import pandas as pd

    sector_returns: Dict[str, Tuple[float, float]] = {}  # name → (1m, 3m)

    # ── Try sectoral index tickers first ──
    try:
        import yfinance as yf
        index_tickers = list(SECTOR_INDEX_TICKERS.values())
        data = yf.download(
            index_tickers, period="3mo", progress=False, auto_adjust=True,
        )
        if not data.empty:
            for name, ticker in SECTOR_INDEX_TICKERS.items():
                try:
                    if len(index_tickers) > 1:
                        close = data[ticker]["Close"].dropna()
                    else:
                        close = data["Close"].dropna()
                    if len(close) >= 22:
                        ret_1m = float(close.iloc[-1] / close.iloc[-22] - 1)
                        ret_3m = float(close.iloc[-1] / close.iloc[0] - 1) if len(close) >= 60 else ret_1m
                        sector_returns[name] = (ret_1m, ret_3m)
                except (KeyError, IndexError):
                    continue
    except Exception as exc:
        logger.debug("Sectoral index download failed: %s", exc)

    # ── Fallback: compute from constituent OHLCV ──
    if not sector_returns and sector_indices and ohlcv_cache:
        for sector, members in sector_indices.items():
            rets_1m = []
            rets_3m = []
            for sym in members:
                df = ohlcv_cache.get(sym)
                if df is not None and "Close" in df.columns:
                    close = df["Close"].dropna()
                    if len(close) >= 22:
                        rets_1m.append(float(close.iloc[-1] / close.iloc[-22] - 1))
                    if len(close) >= 63:
                        rets_3m.append(float(close.iloc[-1] / close.iloc[-63] - 1))
            if rets_1m:
                avg_1m = float(np.mean(rets_1m))
                avg_3m = float(np.mean(rets_3m)) if rets_3m else avg_1m
                sector_returns[sector] = (avg_1m, avg_3m)

    if not sector_returns:
        return SectorRotation(sectors={}, top_sectors=[], bottom_sectors=[])

    # ── Rank sectors by 1-month momentum ──
    sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1][0], reverse=True)
    n = len(sorted_sectors)
    top_n = max(1, n // 3)  # top third

    sectors: Dict[str, SectorMomentum] = {}
    top_sectors: List[str] = []
    bottom_sectors: List[str] = []

    for rank, (name, (ret_1m, ret_3m)) in enumerate(sorted_sectors, 1):
        if rank <= top_n:
            tier = "TOP"
            top_sectors.append(name)
        elif rank > n - top_n:
            tier = "BOTTOM"
            bottom_sectors.append(name)
        else:
            tier = "MID"

        sectors[name] = SectorMomentum(
            sector_name=name,
            return_1m=ret_1m,
            return_3m=ret_3m,
            rank=rank,
            tier=tier,
        )

    logger.info(
        "Sector rotation: TOP=%s, BOTTOM=%s",
        top_sectors, bottom_sectors,
    )

    return SectorRotation(
        sectors=sectors,
        top_sectors=top_sectors,
        bottom_sectors=bottom_sectors,
        analysis_date=datetime.now().strftime("%Y-%m-%d"),
    )


def get_sector_rotation(
    sector_indices: Optional[Dict[str, List[str]]] = None,
    ohlcv_cache: Optional[Dict[str, object]] = None,
) -> SectorRotation:
    """Get cached sector rotation analysis.

    If no data is available, attempts a fresh computation.
    """
    global _CACHE, _CACHE_TS

    now = datetime.now()
    if _CACHE and _CACHE_TS and (now - _CACHE_TS) < _CACHE_TTL:
        return _CACHE

    if sector_indices is None:
        try:
            from kite_connect.core.config import INDEX_CONSTITUENTS
            sector_indices = dict(INDEX_CONSTITUENTS)
        except Exception:
            sector_indices = {}

    rotation = _compute_sector_rotation(sector_indices or {}, ohlcv_cache or {})
    if rotation.sectors:
        _CACHE = rotation
        _CACHE_TS = now

    return rotation


def get_sector_score_adjustment(sector_name: str) -> float:
    """Return screener score adjustment for a sector.

    Returns:
        +8 for TOP sectors, -3 for BOTTOM, 0 for MID.
    """
    rotation = get_sector_rotation()
    return rotation.get_score_adjustment(sector_name)
