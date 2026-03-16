"""
NSE Stock Screener & Analyser.

Three-stage pipeline executed on every symbol in the NSE universe:

Stage 1 — **Screening Criteria** (fast, eliminates ~80 % of universe)
    • Price  > ₹100
    • Avg daily volume  > threshold (default 500 000)
    • Close > 50-day MA *and* 200-day MA  (bullish trend)
    • Beta > threshold (default 1.0)

Stage 2 — **Methodology Analysis** (on survivors from Stage 1)
    • Pullback:  Close within 2 % of 20-day or 50-day MA in an uptrend
    • Breakout:  Close breaks above 20-day high on volume > 1.5× average
    • Sector:    Top sector tag (Nifty IT / Pharma / Bank / Energy) if leading

Stage 3 — **Technical Analysis** (final ranking)
    • Support / Resistance identification
    • RSI  (14-period)
    • Bollinger Bands  (20, 2σ)

Each symbol is scored 0–100 and tagged with qualifying strategies.
The caller receives a ranked ``pd.DataFrame`` ready for risk management
and order execution.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration defaults
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScreenerConfig:
    """Tuneable parameters for the screener pipeline."""

    # Stage 1 — screening
    min_price: float = 100.0
    min_avg_volume: int = 500_000
    min_beta: float = 1.0
    history_days: int = 250          # ~1 year of trading days

    # Stage 2 — methodology
    pullback_pct: float = 0.02       # within 2 % of MA
    breakout_vol_mult: float = 1.5   # volume must be 1.5× average
    breakout_lookback: int = 20      # days for consolidation high

    # Stage 3 — technicals
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    sr_lookback: int = 60            # days for support / resistance

    # Concurrency
    max_workers: int = 8
    yf_batch_size: int = 50          # tickers per yfinance download batch

    # Sector index constituents (kept in sync with kite_connect/core/config.py)
    sector_indices: Dict[str, List[str]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Data container for a screened stock
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScreenedStock:
    symbol: str
    close: float = 0.0
    avg_volume: float = 0.0
    beta: float = 0.0
    ma_50: float = 0.0
    ma_200: float = 0.0
    ma_20: float = 0.0

    # Methodology flags
    pullback: bool = False
    breakout: bool = False
    sector_leader: bool = False
    sector_name: str = ""

    # Technicals
    rsi: float = 50.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    support: float = 0.0
    resistance: float = 0.0

    # Composite score  (0–100)
    score: float = 0.0
    strategies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "close": round(self.close, 2),
            "avg_volume": int(self.avg_volume),
            "beta": round(self.beta, 2),
            "ma_20": round(self.ma_20, 2),
            "ma_50": round(self.ma_50, 2),
            "ma_200": round(self.ma_200, 2),
            "pullback": self.pullback,
            "breakout": self.breakout,
            "sector_leader": self.sector_leader,
            "sector_name": self.sector_name,
            "rsi": round(self.rsi, 2),
            "bb_upper": round(self.bb_upper, 2),
            "bb_middle": round(self.bb_middle, 2),
            "bb_lower": round(self.bb_lower, 2),
            "support": round(self.support, 2),
            "resistance": round(self.resistance, 2),
            "score": round(self.score, 2),
            "strategies": ", ".join(self.strategies),
        }


# ═══════════════════════════════════════════════════════════════
# Helper — download OHLCV data via yfinance
# ═══════════════════════════════════════════════════════════════

def _download_batch(symbols_ns: List[str], period: str = "1y") -> pd.DataFrame:
    """Download daily OHLCV for a list of ``.NS`` symbols."""
    try:
        df = yf.download(
            symbols_ns,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        return df
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Technical helpers
# ═══════════════════════════════════════════════════════════════

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    if len(delta) < period:
        return 50.0
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_vals = 100 - (100 / (1 + rs))
    last = rsi_vals.dropna()
    return float(last.iloc[-1]) if len(last) > 0 else 50.0


def _bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = _sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _support_resistance(series: pd.Series, lookback: int = 60):
    """Simple min/max based support & resistance over *lookback* days."""
    recent = series.tail(lookback)
    if len(recent) < 5:
        return float(series.min()), float(series.max())
    return float(recent.min()), float(recent.max())


def _beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    """Compute stock beta relative to index returns."""
    aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 1.0
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return 1.0
    return float(cov / var)


# ═══════════════════════════════════════════════════════════════
# Sector performance (for methodology stage)
# ═══════════════════════════════════════════════════════════════

def _compute_sector_leaders(
    sector_map: Dict[str, List[str]],
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> set:
    """
    Return symbols belonging to sectors whose 1-month return
    is in the top 50 % of all sectors tracked.
    """
    sector_returns: Dict[str, float] = {}
    for sector, members in sector_map.items():
        rets = []
        for sym in members:
            df = ohlcv_cache.get(sym)
            if df is not None and len(df) >= 22:
                r = (df["Close"].iloc[-1] / df["Close"].iloc[-22]) - 1
                rets.append(r)
        if rets:
            sector_returns[sector] = float(np.mean(rets))
    if not sector_returns:
        return set()

    median_ret = float(np.median(list(sector_returns.values())))
    leading_sectors = {s for s, r in sector_returns.items() if r >= median_ret}
    leaders: set = set()
    for s in leading_sectors:
        leaders.update(sector_map[s])
    return leaders


# ═══════════════════════════════════════════════════════════════
# Main Screener
# ═══════════════════════════════════════════════════════════════

class NSEScreener:
    """
    Three-stage NSE stock screener.

    Usage::

        from kite_connect.nse.screener import NSEScreener, ScreenerConfig
        screener = NSEScreener(config=ScreenerConfig(min_price=100))
        df = screener.screen(symbols=["RELIANCE", "TCS", ...])
    """

    def __init__(self, config: ScreenerConfig | None = None):
        self.cfg = config or ScreenerConfig()
        if not self.cfg.sector_indices:
            try:
                from kite_connect.core.config import INDEX_CONSTITUENTS
                self.cfg.sector_indices = dict(INDEX_CONSTITUENTS)
            except Exception:
                self.cfg.sector_indices = {}

    # ── Public API ─────────────────────────────────────────────

    def screen(
        self,
        symbols: List[str],
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Run the full three-stage pipeline and return a ranked DataFrame.

        Parameters
        ----------
        symbols : list[str]
            Plain NSE symbols (no ``.NS`` suffix).
        progress_callback : callable | None
            ``callback(message: str)`` for progress reporting.

        Returns
        -------
        pd.DataFrame
            Columns defined by :class:`ScreenedStock.to_dict`.
            Sorted **descending** by ``score``.
        """
        if not symbols:
            return pd.DataFrame()

        _cb = progress_callback or (lambda m: None)

        # ── 0.  Download OHLCV data in batches ────────────────
        _cb(f"Downloading price data for {len(symbols)} symbols")
        ohlcv = self._download_all(symbols)
        if not ohlcv:
            logger.warning("No price data retrieved — aborting screen")
            return pd.DataFrame()

        # Download NIFTY-50 for beta computation
        nifty = self._download_index()

        # ── 1.  Stage 1 — Screening Criteria ──────────────────
        _cb("Stage 1: Applying screening criteria …")
        stage1 = self._stage1_screen(ohlcv, nifty)
        _cb(f"Stage 1 passed: {len(stage1)} / {len(ohlcv)} symbols")
        if not stage1:
            return pd.DataFrame()

        # ── 2.  Stage 2 — Methodology Analysis ────────────────
        _cb("Stage 2: Running methodology analysis …")
        sector_leaders = _compute_sector_leaders(self.cfg.sector_indices, ohlcv)
        for stock in stage1:
            self._stage2_methods(stock, ohlcv.get(stock.symbol), sector_leaders)

        # ── 3.  Stage 3 — Technical Analysis ──────────────────
        _cb("Stage 3: Computing technical indicators …")
        for stock in stage1:
            self._stage3_technicals(stock, ohlcv.get(stock.symbol))

        # ── 4.  Score & rank ──────────────────────────────────
        _cb("Scoring and ranking …")
        for stock in stage1:
            self._compute_score(stock)

        stage1.sort(key=lambda s: s.score, reverse=True)

        rows = [s.to_dict() for s in stage1]
        df = pd.DataFrame(rows)
        _cb(f"Screening complete — {len(df)} stocks ranked")
        return df

    # ── Data download ──────────────────────────────────────────

    def _download_all(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Download OHLCV for all symbols, returns {plain_symbol: df}."""
        cache: Dict[str, pd.DataFrame] = {}
        ns_symbols = [f"{s}.NS" for s in symbols]

        for i in range(0, len(ns_symbols), self.cfg.yf_batch_size):
            batch = ns_symbols[i : i + self.cfg.yf_batch_size]
            raw = _download_batch(batch, period="1y")
            if raw.empty:
                continue

            if len(batch) == 1:
                sym = batch[0].replace(".NS", "")
                if "Close" in raw.columns:
                    cache[sym] = raw.copy()
            else:
                for ns in batch:
                    sym = ns.replace(".NS", "")
                    try:
                        sdf = raw[ns].dropna(how="all")
                        if not sdf.empty and "Close" in sdf.columns:
                            cache[sym] = sdf.copy()
                    except (KeyError, TypeError):
                        continue

        logger.info("Downloaded data for %d / %d symbols", len(cache), len(symbols))
        return cache

    def _download_index(self) -> Optional[pd.Series]:
        """Download NIFTY-50 daily returns for beta calculation."""
        try:
            df = yf.download("^NSEI", period="1y", progress=False, auto_adjust=True)
            if df.empty:
                return None
            return df["Close"].pct_change().dropna()
        except Exception:
            return None

    # ── Stage 1:  Screening Criteria ───────────────────────────

    def _stage1_screen(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        nifty_returns: Optional[pd.Series],
    ) -> List[ScreenedStock]:
        """Filter universe by price, volume, trend, volatility."""
        passed: List[ScreenedStock] = []

        for sym, df in ohlcv.items():
            if len(df) < self.cfg.history_days // 2:
                continue

            close = df["Close"]
            volume = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)
            last_close = float(close.iloc[-1])

            # Price filter
            if last_close < self.cfg.min_price:
                continue

            # Volume filter
            avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else 0
            if avg_vol < self.cfg.min_avg_volume:
                continue

            # Trend filter — above 50-MA & 200-MA
            ma50 = float(_sma(close, 50).iloc[-1]) if len(close) >= 50 else 0
            ma200 = float(_sma(close, 200).iloc[-1]) if len(close) >= 200 else 0
            if ma50 == 0 or ma200 == 0:
                continue
            if last_close < ma50 or last_close < ma200:
                continue

            # Volatility / beta filter
            stock_ret = close.pct_change().dropna()
            b = _beta(stock_ret, nifty_returns) if nifty_returns is not None else 1.0
            if b < self.cfg.min_beta:
                continue

            ma20 = float(_sma(close, 20).iloc[-1]) if len(close) >= 20 else last_close

            stock = ScreenedStock(
                symbol=sym,
                close=last_close,
                avg_volume=avg_vol,
                beta=b,
                ma_50=ma50,
                ma_200=ma200,
                ma_20=ma20,
            )
            passed.append(stock)

        return passed

    # ── Stage 2:  Methodology Analysis ─────────────────────────

    def _stage2_methods(
        self,
        stock: ScreenedStock,
        df: Optional[pd.DataFrame],
        sector_leaders: set,
    ):
        """Tag stock with pullback / breakout / sector-leader flags."""
        if df is None or df.empty:
            return

        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)

        # Pullback — close within pullback_pct of 20-MA or 50-MA in uptrend
        if stock.close > stock.ma_200:  # uptrend confirmed
            dist_20 = abs(stock.close - stock.ma_20) / stock.ma_20 if stock.ma_20 else 1
            dist_50 = abs(stock.close - stock.ma_50) / stock.ma_50 if stock.ma_50 else 1
            if dist_20 <= self.cfg.pullback_pct or dist_50 <= self.cfg.pullback_pct:
                stock.pullback = True
                stock.strategies.append("Pullback")

        # Breakout — close > 20-day high with high volume
        if len(close) >= self.cfg.breakout_lookback + 1:
            high_20 = close.iloc[-(self.cfg.breakout_lookback + 1) : -1].max()
            avg_vol_20 = float(volume.tail(self.cfg.breakout_lookback).mean()) if len(volume) >= self.cfg.breakout_lookback else 0
            today_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0
            if stock.close > high_20 and avg_vol_20 > 0 and today_vol > avg_vol_20 * self.cfg.breakout_vol_mult:
                stock.breakout = True
                stock.strategies.append("Breakout")

        # Sector analysis
        if stock.symbol in sector_leaders:
            stock.sector_leader = True
            # Identify which sector(s)
            for name, members in self.cfg.sector_indices.items():
                if stock.symbol in members:
                    stock.sector_name = name
                    break
            stock.strategies.append("Sector Leader")

    # ── Stage 3:  Technical Analysis ───────────────────────────

    def _stage3_technicals(self, stock: ScreenedStock, df: Optional[pd.DataFrame]):
        """Compute RSI, Bollinger Bands, Support / Resistance."""
        if df is None or df.empty:
            return

        close = df["Close"]

        # RSI
        stock.rsi = _rsi(close, self.cfg.rsi_period)

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = _bollinger(close, self.cfg.bb_period, self.cfg.bb_std)
        if len(bb_upper.dropna()) > 0:
            stock.bb_upper = float(bb_upper.iloc[-1])
            stock.bb_middle = float(bb_mid.iloc[-1])
            stock.bb_lower = float(bb_lower.iloc[-1])

        # Support / Resistance
        stock.support, stock.resistance = _support_resistance(close, self.cfg.sr_lookback)

    # ── Scoring ────────────────────────────────────────────────

    def _compute_score(self, stock: ScreenedStock):
        """
        Composite score 0–100 based on:
          - Trend strength   (20 pts)
          - Volatility/beta  (15 pts)
          - Methodology hits (30 pts: pullback 10, breakout 10, sector 10)
          - RSI zone         (15 pts)
          - Bollinger zone   (10 pts)
          - S/R proximity    (10 pts)
        """
        score = 0.0

        # Trend strength — distance above 200-MA (capped at 20 pts)
        if stock.ma_200 > 0:
            trend_pct = (stock.close - stock.ma_200) / stock.ma_200
            score += min(20, trend_pct * 100)

        # Beta (higher = more swing potential, capped at 15)
        score += min(15, stock.beta * 7.5)

        # Methodology bonuses
        if stock.pullback:
            score += 10
        if stock.breakout:
            score += 10
        if stock.sector_leader:
            score += 10

        # RSI — favour oversold-to-neutral zone (30–50 = best entry)
        if 30 <= stock.rsi <= 50:
            score += 15
        elif 50 < stock.rsi <= 60:
            score += 10
        elif 20 <= stock.rsi < 30:
            score += 12

        # Bollinger — near lower band = better entry
        if stock.bb_upper > stock.bb_lower > 0:
            bb_range = stock.bb_upper - stock.bb_lower
            if bb_range > 0:
                bb_pos = (stock.close - stock.bb_lower) / bb_range
                score += max(0, (1 - bb_pos) * 10)   # 10 at lower band, 0 at upper

        # Support proximity — closer to support = safer entry
        if stock.resistance > stock.support > 0:
            sr_range = stock.resistance - stock.support
            if sr_range > 0:
                sr_pos = (stock.close - stock.support) / sr_range
                score += max(0, (1 - sr_pos) * 10)

        stock.score = min(100, max(0, score))
