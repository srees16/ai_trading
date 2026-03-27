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

    # Index mode — relaxed filters for blue-chip NIFTY50/Next50 universe.
    # When True: lower beta threshold (0.3), require only Close > MA200
    # (allow pullback entries below MA50), lower volume floor.
    index_mode: bool = False

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
    adx: float = 0.0
    volume_ratio: float = 1.0     # current vol / 20-day avg vol
    relative_strength: float = 0.0  # stock return minus NIFTY return (1m)

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
            "adx": round(self.adx, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "rel_strength": round(self.relative_strength, 4),
            "score": round(self.score, 2),
            "strategies": ", ".join(self.strategies),
        }


# ═══════════════════════════════════════════════════════════════
# Helper — download OHLCV data via yfinance
# ═══════════════════════════════════════════════════════════════

def _download_batch(symbols_ns: List[str], period: str = "1y") -> pd.DataFrame:
    """Download daily OHLCV for a list of ``.NS`` symbols.

    Retries up to 3 times with exponential backoff on rate-limit (429)
    or transient network errors.
    """
    import time as _time

    max_retries = 3
    for attempt in range(max_retries):
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
            err_str = str(exc).lower()
            is_retryable = (
                "429" in err_str
                or "rate" in err_str
                or "too many" in err_str
                or "connection" in err_str
                or "timeout" in err_str
            )
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                logger.warning(
                    "yfinance rate-limited (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, wait, exc,
                )
                _time.sleep(wait)
                continue
            logger.error("yfinance download failed: %s", exc)
            return pd.DataFrame()
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


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average Directional Index (ADX) for trend strength."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=1 / period, min_periods=period).mean()
    last = adx_val.iloc[-1]
    return float(last) if np.isfinite(last) else 0.0


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

    def __init__(self, config: ScreenerConfig | None = None, kite=None):
        self.cfg = config or ScreenerConfig()
        self.kite = kite  # Optional Kite instance for live LTP
        if not self.cfg.sector_indices:
            try:
                from kite_connect.core.config import INDEX_CONSTITUENTS
                self.cfg.sector_indices = dict(INDEX_CONSTITUENTS)
            except Exception:
                self.cfg.sector_indices = {}

        # ── Gap A fix: load walk-forward optimised screener params ──
        self._apply_wf_optimal_params()

    def _apply_wf_optimal_params(self) -> None:
        """Override screener config with walk-forward optimal params if available.

        Reads the latest WF-optimised params from ``data/wf_params/``.
        Only numeric screener params (thresholds, lookbacks) are overridden;
        structural config (concurrency, sector indices) is never touched.
        Silently degrades to static defaults if no fresh params exist.
        """
        try:
            from services.walk_forward import load_all_optimal_params

            all_params = load_all_optimal_params()
            if not all_params:
                return

            # Aggregate: for each screener-relevant key, collect the
            # best-performing value across all saved strategy/ticker combos.
            _SCREENER_KEYS = {
                "min_price", "min_avg_volume", "min_volume", "min_beta",
                "pullback_pct", "breakout_vol_mult", "breakout_lookback",
                "rsi_period", "bb_period", "bb_std", "sr_lookback",
            }
            applied = {}
            for key, record in all_params.items():
                params = record.get("params", {})
                for pname, pval in params.items():
                    if pname in _SCREENER_KEYS and isinstance(pval, (int, float)):
                        # Keep the value from the record with the highest OOS Sharpe
                        prev = applied.get(pname)
                        if prev is None or record.get("oos_sharpe", 0) > prev[1]:
                            applied[pname] = (pval, record.get("oos_sharpe", 0))

            if not applied:
                return

            count = 0
            for pname, (pval, _sharpe) in applied.items():
                # Map aliases
                attr = "min_avg_volume" if pname == "min_volume" else pname
                if hasattr(self.cfg, attr):
                    old_val = getattr(self.cfg, attr)
                    setattr(self.cfg, attr, type(old_val)(pval))
                    count += 1
                    logger.debug(
                        "WF param override: %s %s → %s", attr, old_val, pval,
                    )
            if count:
                logger.info(
                    "Applied %d walk-forward optimal params to screener config",
                    count,
                )
        except Exception as exc:
            logger.debug("WF param loading skipped: %s", exc)

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

        # Seed sector rotation cache with the OHLCV data we already have
        try:
            from services.sector_rotation import get_sector_rotation
            get_sector_rotation(self.cfg.sector_indices, ohlcv)
        except Exception:
            pass

        for stock in stage1:
            self._stage2_methods(stock, ohlcv.get(stock.symbol), sector_leaders)

        # ── 3.  Stage 3 — Technical Analysis ──────────────────
        _cb("Stage 3: Computing technical indicators …")
        # Compute NIFTY 1-month return for relative strength comparison
        nifty_1m_ret = 0.0
        if nifty is not None and len(nifty) >= 20:
            nifty_1m_ret = float(((1 + nifty.tail(20)).prod() - 1).iloc[0])
        for stock in stage1:
            self._stage3_technicals(stock, ohlcv.get(stock.symbol))
            # Relative strength: stock 1-month return minus NIFTY return
            sdf = ohlcv.get(stock.symbol)
            if sdf is not None and len(sdf) >= 20:
                stock_1m_ret = float(sdf["Close"].iloc[-1] / sdf["Close"].iloc[-20] - 1)
                stock.relative_strength = stock_1m_ret - nifty_1m_ret

        # ── 4.  Score & rank ──────────────────────────────────
        _cb("Scoring and ranking …")
        for stock in stage1:
            self._compute_score(stock)

        stage1.sort(key=lambda s: s.score, reverse=True)

        # ── 4b. Live price refresh via Kite LTP (#4) ──────────
        # During market hours, replace stale yfinance close with
        # Kite LTP for top-80 screened stocks.
        if self.kite is not None:
            top_80 = stage1[:80]
            self._refresh_with_kite_ltp(top_80, _cb)

        rows = [s.to_dict() for s in stage1]
        df = pd.DataFrame(rows)
        _cb(f"Screening complete — {len(df)} stocks ranked")
        return df

    # ── Live LTP refresh via Kite (Feature #4) ────────────────

    def _refresh_with_kite_ltp(self, stocks: List[ScreenedStock], _cb) -> None:
        """Refresh close prices for screened stocks using Kite batch quote()."""
        if not stocks or self.kite is None:
            return
        try:
            # Batch in groups of 200 (Kite API limit)
            symbols = [s.symbol for s in stocks]
            updated = 0
            for i in range(0, len(symbols), 200):
                batch = symbols[i:i + 200]
                keys = [f"NSE:{s}" for s in batch]
                ltp_data = self.kite.ltp(keys)
                for stock in stocks:
                    key = f"NSE:{stock.symbol}"
                    if key in ltp_data:
                        live = ltp_data[key].get("last_price")
                        if live and live > 0:
                            stock.close = float(live)
                            updated += 1
            if updated > 0:
                _cb(f"Refreshed {updated}/{len(stocks)} prices with Kite LTP")
        except Exception as exc:
            logger.warning("Kite LTP refresh failed (using yfinance close): %s", exc)

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

        logger.info("Downloaded data for %d / %d symbols via yfinance", len(cache), len(symbols))

        # ── Bhavcopy fallback for missed symbols ──────────────
        missed = [s for s in symbols if s not in cache]
        if missed:
            try:
                from datetime import date as _date, timedelta as _td
                from services.bhavcopy_fetcher import fetch_ohlcv_batch

                end_dt = _date.today()
                start_dt = end_dt - _td(days=self.cfg.history_days + 30)
                bhav = fetch_ohlcv_batch(missed, start=start_dt, end=end_dt)
                for sym, df in bhav.items():
                    # fetch_ohlcv_batch returns original ticker; strip .NS if present
                    plain = sym.replace(".NS", "").replace(".BO", "")
                    if not df.empty:
                        cache[plain] = df
                if bhav:
                    logger.info(
                        "Bhavcopy filled %d / %d missed symbols",
                        len(bhav), len(missed),
                    )
            except Exception as exc:
                logger.warning("Bhavcopy fallback in screener failed: %s", exc)

        logger.info("Total data: %d / %d symbols after fallbacks", len(cache), len(symbols))

        # ── Survivorship bias filter ──────────────────────────
        # Remove delisted / suspended / dead tickers before any
        # technical analysis is computed.
        try:
            from services.survivorship_filter import filter_valid_tickers
            valid_syms, rejected = filter_valid_tickers(
                list(cache.keys()), market="IND",
                ohlcv_cache=cache, kite=self.kite,
            )
            if rejected:
                for r in rejected:
                    cache.pop(r.ticker, None)
                logger.info(
                    "Survivorship filter: removed %d delisted/suspended symbols",
                    len(rejected),
                )
        except Exception as exc:
            logger.debug("Survivorship filter skipped: %s", exc)

        # ── Corporate action adjustment (#2) ──────────────────
        # Adjust OHLCV for pending splits/bonuses so technical
        # indicators are computed on adjusted prices.
        try:
            from services.corporate_actions import get_actions_for_symbols, adjust_ohlcv_for_action
            pending = get_actions_for_symbols(list(cache.keys()))
            for sym, action in pending.items():
                if sym in cache:
                    cache[sym] = adjust_ohlcv_for_action(cache[sym], action)
                    logger.info("Adjusted %s OHLCV for %s (%s)",
                                sym, action.action_type, action.description[:60])
        except Exception as exc:
            logger.debug("Corporate action adjustment skipped: %s", exc)

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

            # Volume filter (relaxed in index mode: 200K floor)
            vol_floor = 200_000 if self.cfg.index_mode else self.cfg.min_avg_volume
            avg_vol = float(volume.tail(20).mean()) if len(volume) >= 20 else 0
            if avg_vol < vol_floor:
                continue

            # Trend filter
            ma50 = float(_sma(close, 50).iloc[-1]) if len(close) >= 50 else 0
            ma200 = float(_sma(close, 200).iloc[-1]) if len(close) >= 200 else 0
            if ma200 == 0:
                continue
            if self.cfg.index_mode:
                # Index mode: only require Close > MA200 (allow pullback below MA50)
                if last_close < ma200:
                    continue
            else:
                # Full mode: require Close > MA50 AND Close > MA200
                if ma50 == 0:
                    continue
                if last_close < ma50 or last_close < ma200:
                    continue

            # Volatility / beta filter (relaxed in index mode: min 0.3)
            beta_floor = 0.3 if self.cfg.index_mode else self.cfg.min_beta
            stock_ret = close.pct_change().dropna()
            b = _beta(stock_ret, nifty_returns) if nifty_returns is not None else 1.0
            if b < beta_floor:
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
        """Compute RSI, Bollinger Bands, Support / Resistance, ADX, volume ratio."""
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

        # ADX (trend strength)
        stock.adx = _adx(df) if len(df) > 28 else 0.0

        # Volume ratio (today’s volume vs 20-day average)
        if "Volume" in df.columns and len(df) >= 20:
            avg_vol = float(df["Volume"].tail(20).mean())
            if avg_vol > 0:
                stock.volume_ratio = float(df["Volume"].iloc[-1]) / avg_vol

    # ── Scoring ────────────────────────────────────────────────

    def _compute_score(self, stock: ScreenedStock):
        """
        Composite score 0–100 based on:
          - Trend strength   (15 pts)
          - Volatility/beta  (10 pts)
          - Methodology hits (25 pts: pullback 10, breakout 10, sector 5)
          - RSI zone         (15 pts)
          - Bollinger zone   (10 pts)
          - S/R proximity    (10 pts)
          - ADX trend filter (  8 pts) — NEW
          - Volume surge     (  7 pts) — NEW
        """
        score = 0.0

        # Trend strength — distance above 200-MA (capped at 15 pts)
        if stock.ma_200 > 0:
            trend_pct = (stock.close - stock.ma_200) / stock.ma_200
            score += min(15, trend_pct * 100)

        # Beta (higher = more swing potential, capped at 10)
        score += min(10, stock.beta * 5.0)

        # Methodology bonuses
        if stock.pullback:
            score += 10
        if stock.breakout:
            score += 10
        if stock.sector_leader:
            score += 5

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

        # ADX trend strength — strong trend = higher conviction
        if stock.adx >= 25:
            score += 8    # strong trend
        elif stock.adx >= 20:
            score += 5    # moderate trend
        elif stock.adx < 15:
            score -= 3    # choppy market penalty

        # Volume surge — above-average volume confirms move
        if stock.volume_ratio >= 2.0:
            score += 7    # strong volume confirmation
        elif stock.volume_ratio >= 1.5:
            score += 5
        elif stock.volume_ratio >= 1.2:
            score += 3

        # Relative strength vs NIFTY (if computed)
        if stock.relative_strength > 0.05:
            score += 5    # outperforming index by > 5%
        elif stock.relative_strength < -0.05:
            score -= 3    # underperforming index

        # Delivery volume conviction — high delivery % = institutional buying
        try:
            from services.delivery_volume import get_delivery_score
            score += get_delivery_score(stock.symbol)
        except Exception:
            pass  # degrade gracefully

        # Sector rotation overlay — boost top-momentum sectors
        if stock.sector_name:
            try:
                from services.sector_rotation import get_sector_score_adjustment
                score += get_sector_score_adjustment(stock.sector_name)
            except Exception:
                pass

        # ── Circuit limit penalty — stock at/near circuit can't be traded ──
        if stock.close > 0 and hasattr(stock, 'open') and stock.open > 0:
            daily_chg_pct = abs((stock.close - stock.open) / stock.open * 100)
            for band in (5.0, 10.0, 20.0):
                if abs(daily_chg_pct - band) < 0.5:
                    score -= 30  # heavy penalty — order unlikely to fill
                    stock.at_circuit = True
                    break

        stock.score = min(100, max(0, score))
