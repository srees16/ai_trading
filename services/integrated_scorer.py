"""
Integrated Multi-Layer Stock Evaluation Pipeline.

Orchestrates three primary layers into a single per-ticker verdict:

    Layer 1 — Core Analysis   (fundamentals, technicals, macro, IND overlays)
    Layer 2 — Strategy Consensus + Robustness   (backtest strategies, walk-forward,
              CSCV, bootstrap, permutation — merged for efficiency)
    Layer 3 — ML Feature Enrichment   (AFML fractional diff, structural breaks,
              microstructure — opt-in, skipped by default for IND market)

Design decisions (March 2026 consolidation):
    • Layer 1 no longer re-runs the full AlgoTradingSystem pipeline
      (news scraping + sentiment). It directly computes fundamental,
      technical, and macro scores via MetricsCalculator + DecisionEngine,
      saving 15–25 s per ticker.
    • Layer 4 (Robustness) merged into Layer 2 — walk-forward validation
      was already partially computed there; the CSCV, bootstrap, and
      permutation tests now run as part of strategy consensus, producing
      a single robustness-adjusted score.
    • Layer 3 (ML/AFML) is skipped by default for IND market — the
      features (fractional diff, microstructure, SADF) are academic and
      rarely change the final classification for swing/positional trades.
      Users can opt in by removing 'ml_features' from skip_layers.
    • RAG layer remains disabled (neutral placeholder).

Each layer is independently skippable — the pipeline degrades gracefully.
"""

import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights (sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "core": 0.45,
    "strategy": 0.55,       # includes merged robustness validation
}

# Extended weights when ML layer is explicitly enabled
DEFAULT_WEIGHTS_WITH_ML = {
    "core": 0.40,
    "strategy": 0.45,
    "ml_features": 0.15,
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class StockVerdict:
    """Final evaluation result for a single ticker."""

    ticker: str
    market: str  # "US" or "IND"
    final_score: float  # −1 … +1
    classification: str  # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    layer_scores: Dict[str, Optional[float]]  # {"core": 0.6, …}
    layer_details: Dict[str, Any]  # per-layer breakdown
    confidence: float  # 0–1 based on data completeness
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _classify(score: float) -> str:
    if score >= Config.STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    if score >= Config.BUY_THRESHOLD:
        return "BUY"
    if score <= Config.STRONG_SELL_THRESHOLD:
        return "STRONG_SELL"
    if score <= Config.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _safe_mean(values: list) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _fetch_ohlcv(
    ticker: str,
    market: str,
    *,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data, routing by market.

    US tickers go directly through yfinance (avoids the wasted Bhavcopy
    probe inside ``download_ind_ohlcv``).  IND tickers use the existing
    helper which tries Bhavcopy first, then yfinance as fallback.
    """
    if market == "IND":
        from utils import download_ind_ohlcv
        if period:
            return download_ind_ohlcv(ticker, period=period)
        return download_ind_ohlcv(ticker, start=start, end=end)
    # US / other markets — yfinance directly
    import yfinance as yf
    kw: Dict[str, Any] = {}
    if period:
        kw["period"] = period
    if start:
        kw["start"] = start
    if end:
        kw["end"] = end
    try:
        df = yf.download(ticker, progress=False, **kw)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Lazy DB / MinIO accessors (match project conventions)
# ---------------------------------------------------------------------------

def _get_db_service():
    try:
        from database.service import get_database_service
        return get_database_service()
    except Exception:
        return None


def _get_minio():
    try:
        from services.storage.minio_service import get_minio_service
        return get_minio_service()
    except Exception:
        return None


def _load_module(file_path: str, module_name: str):
    """Load a Python module directly from its file path using importlib.

    This avoids the ``applied`` package collision between financial_ML
    and testune_trade_sys (both contain ``applied/__init__.py``).

    The parent directory of the file is temporarily inserted at the
    front of ``sys.path`` and colliding ``sys.modules`` entries
    (``sample_data``, ``applied``) are saved and restored so each
    package resolves its own neighbours correctly.
    """
    import importlib.util

    parent_dir = str(Path(file_path).resolve().parent.parent)

    # Temporarily override sys.path and module cache
    _colliding_keys = ("sample_data", "applied")
    saved_modules = {}
    for key in _colliding_keys:
        if key in sys.modules:
            saved_modules[key] = sys.modules.pop(key)

    inserted = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        inserted = True

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if inserted:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass
        # Remove any modules that were loaded during this call to
        # avoid polluting subsequent loads from a different package.
        for key in _colliding_keys:
            sys.modules.pop(key, None)
        # Restore previously saved modules
        sys.modules.update(saved_modules)


# Resolved root directories
_FML_APPLIED = Path(__file__).resolve().parent.parent / "financial_ML" / "applied"
_TTS_APPLIED = Path(__file__).resolve().parent.parent / "testune_trade_sys" / "applied"

# ---------------------------------------------------------------------------
# Layer 1 cache — avoid recomputing core scores within a short window.
# L1: in-memory dict for sub-ms hot path
# L2: CacheService (Redis/Upstash) for cross-restart persistence
# Key: (ticker, market)  →  (timestamp, result_dict)
# ---------------------------------------------------------------------------
_LAYER1_CACHE: Dict[tuple, tuple] = {}
_LAYER1_TTL = 900  # 15 minutes


def _get_cached_core(ticker: str, market: str) -> Optional[Dict[str, Any]]:
    key = (ticker.upper(), market.upper())
    # L1: in-memory
    entry = _LAYER1_CACHE.get(key)
    if entry is not None:
        ts, result = entry
        if time.time() - ts <= _LAYER1_TTL:
            return result
        _LAYER1_CACHE.pop(key, None)
    # L2: Redis (survives restarts)
    try:
        from infrastructure.cache import cache as _redis_cache
        redis_val = _redis_cache.get(f"l1:{market.upper()}:{ticker.upper()}")
        if redis_val is not None:
            _LAYER1_CACHE[key] = (time.time(), redis_val)
            return redis_val
    except Exception:
        pass
    return None


def _set_cached_core(ticker: str, market: str, result: Dict[str, Any]) -> None:
    key = (ticker.upper(), market.upper())
    _LAYER1_CACHE[key] = (time.time(), result)
    try:
        from infrastructure.cache import cache as _redis_cache
        _redis_cache.set(f"l1:{market.upper()}:{ticker.upper()}", result, ttl=_LAYER1_TTL)
    except Exception:
        pass


# ===================================================================
# Layer 1 — Core Analysis (lightweight — no full pipeline re-run)
# ===================================================================

def _run_layer_core(ticker: str, market: str) -> Dict[str, Any]:
    """Compute fundamental + technical + macro scores directly.

    Unlike the previous implementation this does **not** re-run the full
    ``AlgoTradingSystem.run()`` pipeline (news scraping, sentiment, etc.).
    Instead it instantiates ``MetricsCalculator`` and ``DecisionEngine``
    directly and scores the ticker in ~3-5 s — saving 15-25 s per ticker.
    """
    cached = _get_cached_core(ticker, market)
    if cached is not None:
        logger.debug("Layer 1 cache hit for %s (%s)", ticker, market)
        return cached

    try:
        from services.metrics import MetricsCalculator
        from services.decision_engine import DecisionEngine

        calculator = MetricsCalculator()
        engine = DecisionEngine()

        metrics = calculator.get_stock_metrics(ticker)

        fundamental_score = 0.0
        technical_score = 0.0
        macro_score = 0.0

        if metrics:
            fundamental_score = engine._calculate_fundamental_score(metrics)
            technical_score = engine._calculate_technical_score(metrics)
        macro_score = engine._calculate_macro_score() or 0.0

        # Weighted core score (replicates DecisionEngine logic)
        w = {
            "fund": Config.FUNDAMENTAL_WEIGHT,
            "tech": Config.TECHNICAL_WEIGHT,
            "macro": Config.MACRO_WEIGHT,
        }
        total_w = sum(w.values())
        core_score = (
            fundamental_score * w["fund"]
            + technical_score * w["tech"]
            + macro_score * w["macro"]
        ) / total_w

        # ── IND-only overlays (skip for US — sources are NSE-specific) ──
        delivery_mult = 1.0
        earnings_boost = 0.0
        if market == "IND":
            try:
                from services.delivery_volume import get_delivery_conviction
                delivery_mult = get_delivery_conviction(ticker)
                core_score *= delivery_mult
            except Exception:
                pass

            try:
                from services.earnings_momentum import get_post_earnings_boost
                earnings_boost = get_post_earnings_boost(ticker)
                core_score = _clamp(core_score + earnings_boost)
            except Exception:
                pass

        result = {
            "score": _clamp(core_score),
            "details": {
                "fundamental": round(fundamental_score, 4),
                "technical": round(technical_score, 4),
                "macro": round(macro_score, 4),
                "delivery_conviction": round(delivery_mult, 3),
                "earnings_momentum_boost": round(earnings_boost, 4),
                "combined": round(core_score, 4),
            },
        }
        _set_cached_core(ticker, market, result)
        return result
    except Exception as exc:
        logger.warning("Layer 1 (Core) failed for %s: %s", ticker, exc)
        return {"score": None, "details": {"error": str(exc)}}


# ===================================================================
# Layer 2 — Strategy Consensus + Robustness (merged)
# ===================================================================

def _run_layer_strategy(
    ticker: str, market: str, date_range: tuple,
    trade_horizon: str = "swing",
) -> Dict[str, Any]:
    """Run registered strategies, aggregate consensus, and apply robustness validation.

    Robustness tests (previously a separate Layer 4) are now integrated:
    walk-forward, CSCV, BCa bootstrap, and permutation tests run on the
    best-performing strategy's returns, producing a single robustness-
    adjusted score.

    Parameters
    ----------
    trade_horizon : str
        ``"swing"`` (3-10 day) or ``"positional"`` (2-6 week).
        Strategies more suited to the chosen horizon receive a 1.5×
        weight multiplier in the Sharpe-weighted consensus vote.
    """
    try:
        from strategies import StrategyRegistry, load_all_strategies

        load_all_strategies()
        all_strategies = StrategyRegistry._strategies

        if not all_strategies:
            return {"score": None, "details": {"error": "No strategies registered"}}

        start_date, end_date = date_range
        buy_votes = 0
        sell_votes = 0
        sharpes: List[float] = []
        drawdowns: List[float] = []
        strategy_results: Dict[str, Any] = {}

        # Strategies that require ≥2 tickers (cointegration / pairs)
        _MULTI_TICKER_STRATEGIES = {"pairs trading", "mean reversion (z-score)"}
        # Strategies unsuitable for Indian (NSE) stocks — cointegration and
        # mean-reversion assumptions break down due to lower liquidity,
        # operator-driven moves, and fragmented order books.
        _IND_EXCLUDED = _MULTI_TICKER_STRATEGIES | {
            "pairs trading", "mean reversion", "mean reversion (z-score)",
            "statistical arbitrage",
        }

        # ── P1-P2: Trade-horizon weight multipliers ──────────
        # Strategies more suited to the chosen horizon get a 1.5× boost
        # in the Sharpe-weighted consensus vote; less-suited get 0.8×.
        _SWING_PREFERRED = {
            "liquidity sweep", "order flow imbalance", "rsi pattern",
            "swing combo",
        }
        _POSITIONAL_PREFERRED = {
            "volume profile", "anchored vwap", "liquidity sweep",
            "positional combo", "macd oscillator", "parabolic sar",
        }
        _horizon_preferred = (
            _SWING_PREFERRED if trade_horizon == "swing" else _POSITIONAL_PREFERRED
        )

        # ── Gap A fix: load WF-optimised params per strategy ──
        _wf_params_map: Dict[str, dict] = {}
        try:
            from services.walk_forward import load_optimal_params as _load_wf
            for name in all_strategies:
                wfp = _load_wf(name, ticker)
                if wfp:
                    _wf_params_map[name.lower()] = wfp
        except Exception:
            pass  # degrade — use default params

        for name, strategy_cls in all_strategies.items():
            # Skip crypto strategies when evaluating US/IND stocks
            if "crypto" in name.lower():
                continue
            # Skip multi-ticker strategies in per-ticker evaluation
            if name.lower() in _MULTI_TICKER_STRATEGIES:
                strategy_results[name] = {"skipped": "requires multiple tickers"}
                continue
            # Skip IND-incompatible strategies
            if market == "IND" and name.lower() in _IND_EXCLUDED:
                strategy_results[name] = {"skipped": "excluded for IND market"}
                continue
            try:
                strategy = strategy_cls()
                # Use WF-optimised params if available for this strategy
                wf_kwargs = _wf_params_map.get(name.lower(), {})
                result = strategy.run(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date,
                    capital=10000,
                    **wf_kwargs,
                )
                if not result.success:
                    strategy_results[name] = {"error": result.error_message}
                    continue

                # Apply transaction costs to portfolio if available
                if (result.portfolio is not None and not result.portfolio.empty
                        and result.signals is not None and not result.signals.empty):
                    from strategies.utils import apply_transaction_costs
                    try:
                        result.portfolio = apply_transaction_costs(
                            result.portfolio, result.signals,
                            cost_pct=result.transaction_cost_pct,
                        )
                    except Exception:
                        pass  # degrade gracefully

                # Extract signal direction from the last row of signals
                strat_buy = 0
                strat_sell = 0
                if result.signals is not None and not result.signals.empty:
                    last_signal = result.signals.iloc[-1]
                    sig_col = next(
                        (c for c in ("signal", "Signal", "position", "Position",
                                     "signals")
                         if c in result.signals.columns),
                        None,
                    )
                    if sig_col is not None:
                        val = last_signal[sig_col]
                        if val > 0:
                            strat_buy = 1
                        elif val < 0:
                            strat_sell = 1
                buy_votes += strat_buy
                sell_votes += strat_sell

                # Extract metrics — strategies return a nested dict
                # keyed by ticker; unwrap to get the flat metrics dict.
                raw_metrics = result.metrics
                if isinstance(raw_metrics, dict) and ticker in raw_metrics:
                    flat_metrics = raw_metrics[ticker]
                elif isinstance(raw_metrics, dict) and "aggregate" in raw_metrics:
                    flat_metrics = raw_metrics["aggregate"]
                else:
                    flat_metrics = raw_metrics  # already flat

                sr = (
                    flat_metrics.get("sharpe_ratio")
                    or flat_metrics.get("sharpe")
                    or flat_metrics.get("avg_sharpe")
                )
                md = flat_metrics.get("max_drawdown")
                if sr is not None and np.isfinite(sr):
                    sharpes.append(float(sr))
                if md is not None and np.isfinite(md):
                    drawdowns.append(float(md))

                strat_signal = (
                    "BUY" if strat_buy > strat_sell
                    else "SELL" if strat_sell > strat_buy
                    else "NEUTRAL"
                )

                # Collect position vector for MC permutation test
                _pos_vec = None
                if result.signals is not None and not result.signals.empty and sig_col is not None:
                    try:
                        _pos_vec = result.signals[sig_col].fillna(0).values.astype(float)
                    except Exception:
                        pass

                strategy_results[name] = {
                    "sharpe": sr,
                    "max_drawdown": md,
                    "last_signal": strat_signal,
                    "_position_vector": _pos_vec,
                }
            except Exception as e:
                strategy_results[name] = {"error": str(e)}

        total_votes = buy_votes + sell_votes
        if total_votes == 0:
            consensus = 0.0
        else:
            consensus = (buy_votes - sell_votes) / total_votes  # −1 … +1

        median_sharpe = float(np.median(sharpes)) if sharpes else 0.0
        worst_dd = min(drawdowns) if drawdowns else 0.0

        # ── S2: Correlation-weighted voting ──────────────────
        # Weight each strategy's vote by its Sharpe ratio so that
        # better-performing strategies have more influence on the
        # final consensus.  Falls back to equal-weight if no
        # Sharpe data is available.
        weighted_consensus = consensus  # default to equal-weight
        if sharpes and total_votes > 0:
            # Build per-strategy weighted vote
            # Only strategies above the minimum Sharpe floor get a vote,
            # preventing low-quality strategies from diluting consensus.
            #
            # Tier 1 Gap 1: REJECT overfit / random strategies outright.
            # If degradation_ratio or permutation p-value indicate the
            # strategy is overfit or no better than random, it is
            # excluded from the consensus vote entirely.
            weighted_buy = 0.0
            weighted_sell = 0.0
            total_weight = 0.0
            _rejected_strategies: List[str] = []
            for name, res in strategy_results.items():
                if isinstance(res, dict) and "sharpe" in res and res["sharpe"] is not None:
                    sr_val = float(res["sharpe"])
                    if sr_val < Config.MIN_STRATEGY_SHARPE:
                        continue  # below quality floor — excluded
                    # P1-P2: horizon-aware weight multiplier
                    horizon_mult = 1.5 if name.lower() in _horizon_preferred else 0.8
                    w = sr_val * horizon_mult
                    sig = res.get("last_signal", "NEUTRAL")
                    if sig == "BUY":
                        weighted_buy += w
                    elif sig == "SELL":
                        weighted_sell += w
                    total_weight += w
            if total_weight > 0:
                weighted_consensus = (weighted_buy - weighted_sell) / total_weight
                weighted_consensus = max(-1.0, min(1.0, weighted_consensus))

        # Sharpe bonus/penalty (clamped to ±0.3)
        sharpe_adj = _clamp(median_sharpe / 5.0, -0.3, 0.3)
        # Drawdown penalty (worst drawdown, negative = bad)
        dd_adj = _clamp(worst_dd / 2.0, -0.3, 0.0) if worst_dd < -0.10 else 0.0

        # ── P9: Sector-aware combo weighting ─────────────────
        # If the ticker's sector is in a strong momentum regime
        # (positive 20-day return), boost the combo strategies'
        # contribution.  Weak-sector tickers get a small penalty.
        sector_adj = 0.0
        sector_details: Dict[str, Any] = {}
        try:
            from services.sector_momentum import get_sector_momentum
            sm = get_sector_momentum(ticker, market)
            if sm is not None:
                sector_details["sector"] = sm.sector_name
                sector_details["sector_return_20d"] = round(sm.return_20d, 4)
                if sm.return_20d > 0.03:       # sector up > 3%
                    sector_adj = 0.05           # mild bullish boost
                elif sm.return_20d < -0.03:    # sector down > 3%
                    sector_adj = -0.05          # mild bearish drag
        except Exception:
            pass  # sector data unavailable — no adjustment

        # ── Walk-forward degradation penalty (#1) ────────────
        # Run walk-forward validation on top-voted strategies to
        # detect overfitting.  A degradation ratio < 0.5 (OOS Sharpe
        # is less than half of IS Sharpe) penalises the score.
        wf_adj = 0.0
        wf_details: Dict[str, Any] = {}
        try:
            from services.walk_forward import walk_forward_validate
            # Pick the single best strategy by Sharpe for WF validation
            best_strat_name = None
            best_strat_sharpe = -999.0
            for name, res in strategy_results.items():
                if isinstance(res, dict) and "sharpe" in res and res["sharpe"] is not None:
                    sr_val = float(res["sharpe"])
                    if sr_val > best_strat_sharpe:
                        best_strat_sharpe = sr_val
                        best_strat_name = name
            if best_strat_name and best_strat_name in all_strategies:
                wf_summary = walk_forward_validate(
                    strategy_cls=all_strategies[best_strat_name],
                    ticker=ticker,
                    capital=10000,
                    train_days=252,
                    test_days=63,
                    total_days=756,
                )
                wf_details = wf_summary.to_dict()
                deg = wf_summary.degradation_ratio
                if deg < 0.3:
                    wf_adj = -0.2  # heavy overfitting penalty
                elif deg < 0.5:
                    wf_adj = -0.1  # moderate penalty
                elif deg > 0.8:
                    wf_adj = 0.05  # slight bonus for robust strategy
                wf_details["adjustment"] = round(wf_adj, 3)

                # Tier 1 Gap 1: Hard rejection gate — if degradation < 0.5,
                # strategy is likely overfit; demote score to suppress signal.
                if deg < 0.5:
                    wf_details["rejected_overfit"] = True
                    logger.info(
                        "WF rejection gate: %s on %s deg=%.2f < 0.5 → overfit",
                        best_strat_name, ticker, deg,
                    )
        except Exception as e:
            wf_details = {"error": str(e)}

        score = _clamp(weighted_consensus * 0.6 + sharpe_adj + dd_adj + wf_adj + sector_adj)

        # ── Merged robustness validation (ex-Layer 4) ────────
        # Run CSCV, BCa bootstrap, and permutation tests on price
        # returns to validate strategy edge is not noise/overfitting.
        robustness_details: Dict[str, Any] = {}
        robustness_adj = 0.0
        try:
            rob_data = _fetch_ohlcv(ticker, market, start=date_range[0], end=date_range[1])
            if not rob_data.empty:
                rob_close = rob_data["Close"].squeeze().dropna()
                rob_returns = rob_close.pct_change().dropna().values
                if len(rob_returns) >= 200:
                    rob_sub_scores: List[float] = []

                    # CSCV overfitting probability
                    try:
                        ch05_tts = _load_module(
                            str(_TTS_APPLIED / "ch05_estimating_future_performance_unbiased.py"),
                            "tts_ch05",
                        )
                        cscv_superiority = ch05_tts.cscv_superiority
                        n_configs = 5
                        slices = [rob_returns[i::n_configs] for i in range(n_configs)]
                        min_len = min(len(s) for s in slices)
                        if min_len > 10:
                            ret_matrix = np.row_stack([s[:min_len] for s in slices])
                            if ret_matrix.ndim == 2 and ret_matrix.shape[0] >= 2:
                                cscv = cscv_superiority(ret_matrix, n_blocks=4)
                                pbo = cscv.get("pbo", 0.5)
                                cscv_score = _clamp((1.0 - pbo * 2) * 0.5)
                                robustness_details["cscv_pbo"] = round(float(pbo), 4)
                                rob_sub_scores.append(cscv_score)
                    except Exception as e:
                        robustness_details["cscv_error"] = str(e)

                    # BCa bootstrap confidence interval
                    try:
                        ch06_tts = _load_module(
                            str(_TTS_APPLIED / "ch06_estimating_future_performance_trade_analysis.py"),
                            "tts_ch06",
                        )
                        bca_bootstrap = ch06_tts.bca_bootstrap
                        bca = bca_bootstrap(rob_returns, n_boot=1000, confidence=0.95, seed=42)
                        lower = bca.get("lower", 0)
                        bca_score = _clamp(float(np.sign(lower)) * 0.5)
                        robustness_details["bca_lower_95"] = round(float(lower), 6)
                        robustness_details["bca_upper_95"] = round(float(bca.get("upper", 0)), 6)
                        rob_sub_scores.append(bca_score)
                    except Exception as e:
                        robustness_details["bca_error"] = str(e)

                    # MC Permutation test (Timothy Masters position-shuffle)
                    try:
                        from services.mc_permutation_test import MCPermutationTest
                        mc_engine = MCPermutationTest(
                            n_perms=getattr(Config, 'MC_PERMUTATION_N_REPS', 5000),
                            center_returns=getattr(Config, 'MC_CENTER_RETURNS', True),
                            normalize_time=getattr(Config, 'MC_NORMALIZE_TIME', True),
                            significance_level=getattr(Config, 'MC_SIGNIFICANCE_LEVEL', 0.05),
                            seed=42,
                        )

                        # Collect actual position vectors from strategy results
                        _collected_positions = []
                        _collected_names = []
                        for _sname, _sres in strategy_results.items():
                            _pvec = _sres.get("_position_vector")
                            if _pvec is not None and len(_pvec) > 0:
                                # Align to rob_returns length
                                _plen = min(len(_pvec), len(rob_returns))
                                if _plen >= 30:
                                    _collected_positions.append(_pvec[-_plen:])
                                    _collected_names.append(_sname)

                        if _collected_positions:
                            # Use the BEST strategy's position vector for single-system test
                            _best_idx = 0
                            _best_sr = -999
                            for _ci, _cn in enumerate(_collected_names):
                                _csr = strategy_results[_cn].get("sharpe") or 0
                                if _csr and _csr > _best_sr:
                                    _best_sr = _csr
                                    _best_idx = _ci

                            _best_pos = _collected_positions[_best_idx]
                            _n_aligned = min(len(_best_pos), len(rob_returns))
                            mc_result = mc_engine.test_single_system(
                                rob_returns[-_n_aligned:], _best_pos[-_n_aligned:],
                            )
                            p_value = mc_result.p_value
                            perm_score = _clamp((0.5 - p_value) * 2)
                            robustness_details["perm_p_value"] = round(float(p_value), 4)
                            robustness_details["perm_z_score"] = round(mc_result.z_score, 3)
                            robustness_details["perm_significant"] = mc_result.significant
                            rob_sub_scores.append(perm_score)

                            # Tier 1 Gap 1: If permutation p-value > 0.10,
                            # the strategy is no better than random — suppress score.
                            if p_value > 0.10:
                                robustness_details["rejected_random"] = True
                                # Apply heavy penalty to drive score toward HOLD
                                robustness_adj = -0.30
                                logger.info(
                                    "Perm rejection gate: %s p=%.4f > 0.10 → random",
                                    ticker, p_value,
                                )

                            # Skill vs luck decomposition
                            sl = mc_engine.partition_skill_luck(
                                rob_returns[-_n_aligned:], _best_pos[-_n_aligned:],
                            )
                            robustness_details["skill_fraction"] = round(sl.skill_fraction, 4)
                            robustness_details["luck_fraction"] = round(sl.luck_fraction, 4)
                            robustness_details["skill_p_value"] = round(sl.p_value, 4)

                            # Best-of-N correction if multiple strategies
                            if len(_collected_positions) > 1:
                                _aligned_vecs = []
                                _min_len = len(rob_returns)
                                for _pv in _collected_positions:
                                    _ml = min(len(_pv), _min_len)
                                    _min_len = _ml
                                for _pv in _collected_positions:
                                    _aligned_vecs.append(_pv[-_min_len:])
                                bon = mc_engine.test_best_of_n(
                                    rob_returns[-_min_len:], _aligned_vecs, _collected_names,
                                )
                                robustness_details["best_of_n_p"] = round(bon.corrected_p_value, 4)
                                robustness_details["best_of_n_significant"] = bon.significant
                        else:
                            # Fallback: old-style permutation with SMA proxy
                            ch07_tts = _load_module(
                                str(_TTS_APPLIED / "ch07_permutation_tests.py"), "tts_ch07",
                            )
                            permutation_test = ch07_tts.permutation_test

                            def _sma_strat_returns(rets):
                                short, long_ = 10, 50
                                if len(rets) < long_:
                                    return float(np.mean(rets))
                                fast = np.convolve(rets, np.ones(short) / short, "valid")
                                slow = np.convolve(rets, np.ones(long_) / long_, "valid")
                                ml = min(len(fast), len(slow))
                                sig = np.where(fast[-ml:] > slow[-ml:], 1.0, -1.0)
                                return float(np.mean(rets[-ml:] * sig))

                            perm = permutation_test(rob_returns, _sma_strat_returns, n_perms=200, seed=42)
                            p_value = perm.get("p_value", 1.0)
                            perm_score = _clamp((0.5 - p_value) * 2)
                            robustness_details["perm_p_value"] = round(float(p_value), 4)
                            robustness_details["perm_fallback"] = True
                            rob_sub_scores.append(perm_score)
                    except Exception as e:
                        robustness_details["perm_error"] = str(e)

                    if rob_sub_scores:
                        robustness_adj = _safe_mean(rob_sub_scores) * 0.15
                        robustness_details["adjustment"] = round(robustness_adj, 4)
        except Exception as e:
            robustness_details["error"] = str(e)

        score = _clamp(score + robustness_adj)

        # Strip internal _position_vector from per-strategy output
        _clean_results = {}
        for _k, _v in strategy_results.items():
            if isinstance(_v, dict):
                _clean_results[_k] = {k2: v2 for k2, v2 in _v.items() if k2 != "_position_vector"}
            else:
                _clean_results[_k] = _v

        return {
            "score": score,
            "details": {
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "total_strategies": len(all_strategies),
                "median_sharpe": round(median_sharpe, 4),
                "worst_max_drawdown": round(worst_dd, 4),
                "consensus_raw": round(consensus, 4),
                "trade_horizon": trade_horizon,
                "sector": sector_details,
                "walk_forward": wf_details,
                "robustness": robustness_details,
                "per_strategy": _clean_results,
            },
        }

        # ── Benchmark alpha overlay (IND only) ──
        # Adds benchmark comparison data to the details dict for
        # transparency. Does NOT alter the score — informational.
        if market == "IND":
            try:
                from services.benchmark_tracker import compare_strategy_to_nifty
                bcomp = compare_strategy_to_nifty(ticker, start_date, end_date)
                if bcomp is not None:
                    result["details"]["benchmark"] = {
                        "nifty_return_pct": bcomp.benchmark_return_pct,
                        "stock_return_pct": bcomp.portfolio_return_pct,
                        "excess_return_pct": bcomp.excess_return_pct,
                        "jensens_alpha": bcomp.jensens_alpha,
                        "beta_vs_nifty": bcomp.portfolio_beta,
                        "information_ratio": bcomp.information_ratio,
                    }
            except Exception as e:
                result["details"]["benchmark"] = {"error": str(e)}

        return result
    except Exception as exc:
        logger.warning("Layer 2 (Strategy+Robustness) failed for %s: %s", ticker, exc)
        return {"score": None, "details": {"error": str(exc)}}


# ===================================================================
# Layer 3 — ML Feature Enrichment
# ===================================================================

def _run_layer_ml(ticker: str, market: str = "IND") -> Dict[str, Any]:
    """Compute AFML-based feature scores for a ticker."""
    import concurrent.futures

    def _run_with_timeout(fn, timeout_sec=30):
        """Run *fn* in a thread; return result or raise TimeoutError."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn)
            return fut.result(timeout=timeout_sec)

    try:
        data = _fetch_ohlcv(ticker, market, period="2y")
        if data.empty:
            return {"score": None, "details": {"error": "No price data"}}

        close = data["Close"].squeeze()
        close = close.dropna()
        if close.empty:
            return {"score": None, "details": {"error": "No close data"}}

        details: Dict[str, Any] = {}
        sub_scores: List[float] = []

        # ── ch05: Fractional Differentiation ──
        try:
            ch05_fml = _load_module(
                str(_FML_APPLIED / "ch05_fractionally_differentiated_features.py"),
                "fml_ch05",
            )
            fracDiff_FFD = ch05_fml.fracDiff_FFD

            close_df = close.to_frame("close")
            diff_series = fracDiff_FFD(close_df, d=0.4, thres=1e-5)
            last_val = diff_series.iloc[-1].values[0] if not diff_series.empty else 0
            # Normalise: positive fracdiff ≈ uptrend → bullish
            fd_score = _clamp(float(np.sign(last_val)) * 0.5)
            details["frac_diff_d04"] = round(float(last_val), 6)
            sub_scores.append(fd_score)
        except Exception as e:
            details["frac_diff_error"] = str(e)

        # ── ch17: Structural Breaks (SADF) ──
        try:
            ch17 = _load_module(
                str(_FML_APPLIED / "ch17_structural_breaks.py"), "fml_ch17",
            )
            sadf_series = ch17.sadf_series

            log_close = np.log(close).to_frame("logP")
            # Subsample to max 200 points to keep SADF O(n^2) tractable
            if len(log_close) > 200:
                step = len(log_close) // 200
                log_close = log_close.iloc[::step]

            def _compute_sadf():
                return sadf_series(log_close, minSL=20, constant="nc", lags=1)

            sadf = _run_with_timeout(_compute_sadf, timeout_sec=30)
            peak_sadf = float(sadf.max()) if len(sadf) > 0 else 0
            # SADF > 1.0 ⇒ explosiveness ⇒ potential bubble (bearish signal)
            sb_score = _clamp(-peak_sadf / 3.0)
            details["sadf_peak"] = round(peak_sadf, 4)
            sub_scores.append(sb_score)
        except concurrent.futures.TimeoutError:
            details["sadf_error"] = "timed out (30s)"
        except Exception as e:
            details["sadf_error"] = str(e)

        # ── ch19: Microstructural Features ──
        try:
            ch19 = _load_module(
                str(_FML_APPLIED / "ch19_microstructural_features.py"), "fml_ch19",
            )
            roll_model = ch19.roll_model
            amihud_lambda = ch19.amihud_lambda

            rm = roll_model(close)
            spread = rm.get("spread", 0)
            # Tight spread = liquid = healthy → positive
            spread_score = _clamp(1.0 - min(spread / 0.02, 2.0), -1, 1) * 0.5
            details["roll_spread"] = round(float(spread), 6)
            sub_scores.append(spread_score)

            if "Volume" in data.columns:
                vol = data["Volume"].squeeze().dropna()
                if len(vol) == len(close):
                    amihud = amihud_lambda(close, vol, window=50)
                    # Low Amihud = liquid = healthy
                    last_amihud = float(amihud.iloc[-1]) if len(amihud) else 0
                    amihud_score = _clamp(-last_amihud * 100, -0.5, 0.5)
                    details["amihud_lambda_last"] = round(last_amihud, 8)
                    sub_scores.append(amihud_score)
        except Exception as e:
            details["micro_error"] = str(e)

        # ── ch14: Backtest Statistics (Sharpe, PSR) ──
        try:
            ch14 = _load_module(
                str(_FML_APPLIED / "ch14_backtest_statistics.py"), "fml_ch14",
            )
            sharpeRatio = ch14.sharpeRatio
            probabilisticSharpeRatio = ch14.probabilisticSharpeRatio
            computeDD_TuW = ch14.computeDD_TuW

            returns = close.pct_change().dropna()
            sr = sharpeRatio(returns)
            psr = probabilisticSharpeRatio(returns)
            dd, _ = computeDD_TuW(returns)
            max_dd = float(dd.min()) if len(dd) else 0

            # Good SR and PSR → bullish
            stat_score = _clamp((sr / 3.0) + (psr - 0.5), -1, 1) * 0.5
            details["sharpe_ratio"] = round(float(sr), 4)
            details["psr"] = round(float(psr), 4)
            details["max_drawdown"] = round(max_dd, 4)
            sub_scores.append(stat_score)
        except Exception as e:
            details["backtest_stats_error"] = str(e)

        # ── ch03: Triple-Barrier Labeling ──
        try:
            ch03 = _load_module(
                str(_FML_APPLIED / "ch03_labeling.py"), "fml_ch03",
            )
            getDailyVol = ch03.getDailyVol
            applyPtSlOnT1 = ch03.applyPtSlOnT1
            getEvents = ch03.getEvents
            getBins = ch03.getBins

            close_series = close.copy()
            close_series.index = pd.to_datetime(close_series.index)

            def _compute_triple_barrier():
                daily_vol = getDailyVol(close_series, span0=50)

                # Use simple CUSUM-like trigger: price crosses 1 std dev
                t_events = close_series.index[50:]  # skip warmup

                # Get events with symmetric barriers (pt_sl = [1, 1])
                events = getEvents(
                    close_series, tEvents=t_events, ptSl=[1, 1],
                    trgt=daily_vol, minRet=0.0,
                    t1=pd.Series(
                        data=[t_events[-1]] * len(t_events),
                        index=t_events,
                    ),
                )
                if events is not None and not events.empty:
                    bins = getBins(events, close_series)
                    if bins is not None and not bins.empty:
                        # Recent label distribution → bullish/bearish signal
                        recent = bins.tail(20)
                        buy_ratio = (recent["bin"] == 1).mean()
                        sell_ratio = (recent["bin"] == -1).mean()
                        return buy_ratio, sell_ratio
                return None, None

            buy_ratio, sell_ratio = _run_with_timeout(_compute_triple_barrier, timeout_sec=30)
            if buy_ratio is not None:
                tb_score = _clamp((buy_ratio - sell_ratio) * 2)
                details["triple_barrier_buy_pct"] = round(float(buy_ratio), 4)
                details["triple_barrier_sell_pct"] = round(float(sell_ratio), 4)
                sub_scores.append(tb_score)
        except concurrent.futures.TimeoutError:
            details["triple_barrier_error"] = "timed out (30s)"
        except Exception as e:
            details["triple_barrier_error"] = str(e)

        # ── ch07: Purged K-Fold CV (strategy robustness) ──
        try:
            ch07 = _load_module(
                str(_FML_APPLIED / "ch07_cross_validation_in_finance.py"), "fml_ch07",
            )
            PurgedKFold = ch07.PurgedKFold

            returns = close.pct_change().dropna()
            if len(returns) >= 200:
                from sklearn.linear_model import SGDClassifier

                X_vals = returns.values[:-1].reshape(-1, 1)
                y = (returns.values[1:] > 0).astype(int)
                X = pd.DataFrame(
                    X_vals,
                    index=returns.index[:-1],
                    columns=["ret"],
                )
                t1 = pd.Series(
                    data=returns.index[1:],
                    index=returns.index[:-1],
                )

                def _compute_purged_cv():
                    pkf = PurgedKFold(n_splits=5, t1=t1, pctEmbargo=0.01)
                    cv_scores = []
                    for train_idx, test_idx in pkf.split(X):
                        clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=200)
                        clf.fit(X_vals[train_idx], y[train_idx])
                        cv_scores.append(float(clf.score(X_vals[test_idx], y[test_idx])))
                    return cv_scores

                cv_scores = _run_with_timeout(_compute_purged_cv, timeout_sec=20)

                mean_cv = float(np.mean(cv_scores))
                # CV accuracy > 0.55 = meaningful edge
                cv_score = _clamp((mean_cv - 0.5) * 4)
                details["purged_cv_mean_accuracy"] = round(mean_cv, 4)
                details["purged_cv_scores"] = [round(s, 4) for s in cv_scores]
                sub_scores.append(cv_score)
        except concurrent.futures.TimeoutError:
            details["purged_cv_error"] = "timed out (20s)"
        except Exception as e:
            details["purged_cv_error"] = str(e)

        # ── ch10: Bet Sizing (Kelly confidence) ──
        try:
            ch10 = _load_module(
                str(_FML_APPLIED / "ch10_bet_sizing.py"), "fml_ch10",
            )
            discreteSignal = ch10.discreteSignal

            returns = close.pct_change().dropna()
            if len(returns) >= 100:
                # Use recent Sharpe as signal strength proxy
                recent_ret = returns.tail(60)
                signal_strength = float(
                    recent_ret.mean() / (recent_ret.std() + 1e-10) * np.sqrt(252)
                )
                # Bound signal to reasonable range before passing to Kelly
                bounded_signal = max(-3.0, min(3.0, signal_strength))
                # discreteSignal(signal0, stepSize) — rounds & clips
                # signal to multiples of stepSize in [-1, 1].
                raw_signal = pd.Series([bounded_signal / 3.0])  # normalise to ~[-1,1]
                bet_size_s = discreteSignal(
                    signal0=raw_signal, stepSize=0.1,
                )
                bet_size = float(bet_size_s.iloc[0])
                # bet_size in [-1,1]: positive = bullish, negative = bearish
                kelly_score = _clamp(bet_size * 0.5)
                details["kelly_bet_size"] = round(float(bet_size), 4)
                details["kelly_signal_strength"] = round(signal_strength, 4)
                sub_scores.append(kelly_score)
        except Exception as e:
            details["bet_sizing_error"] = str(e)

        score = _safe_mean(sub_scores) if sub_scores else None
        return {"score": _clamp(score) if score is not None else None, "details": details}

    except Exception as exc:
        logger.warning("Layer 3 (ML) failed for %s: %s", ticker, exc)
        return {"score": None, "details": {"error": str(exc)}}


# ===================================================================
# RAG Knowledge Augmentation (disabled — kept for future use)
# ===================================================================

def _run_layer_rag(ticker: str) -> Dict[str, Any]:
    """Query RAG engine for qualitative insights on the ticker.

    .. note::

       This layer is **disabled by default** in the verdict pipeline.
       The RAG engine calls Anthropic / Ollama LLMs which adds significant
       latency (~5-15 s per ticker) while the scoring contribution is
       minimal: the full LLM answer is reduced to naive keyword counting
       and capped at 10 % weight.  The vector store also contains
       user-uploaded domain documents, not per-ticker equity research,
       so retrieval relevance is low.

       Enable manually from the UI if the ChromaDB store has been
       populated with relevant per-stock research.
    """
    # Short-circuit: no LLM calls.  Return a neutral placeholder so the
    # layer is scored as "skipped" and its weight is redistributed.
    return {
        "score": None,
        "details": {
            "note": (
                "RAG layer disabled — LLM calls (Anthropic/Ollama) "
                "provide minimal scoring value for the verdict pipeline. "
                "Remove 'rag' from skip_layers to re-enable."
            ),
        },
    }


# ===================================================================
# Main Scorer
# ===================================================================

class IntegratedScorer:
    """Orchestrates the consolidated evaluation pipeline.

    Default layers:
        core     — fundamentals + technicals + macro + IND overlays
        strategy — strategy consensus + merged robustness validation

    Optional layer (opt-in, skipped by default for IND):
        ml_features — AFML fractional diff, structural breaks, microstructure
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        # Add RL layer weight when RL is enabled
        if Config.RL_ENABLED and "rl_bot" not in self.weights:
            self.weights["rl_bot"] = Config.RL_LAYER_WEIGHT
        if weights:
            self.weights.update(weights)
        # Normalise
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    def evaluate(
        self,
        tickers: List[str],
        market: str = "US",
        date_range: Optional[tuple] = None,
        skip_layers: Optional[List[str]] = None,
        max_workers: int = 4,
        trade_horizon: str = "swing",
    ) -> List[StockVerdict]:
        """
        Run the multi-layer pipeline for *tickers*.

        Layers:
            core     — fundamentals, technicals, macro, IND overlays (always)
            strategy — strategy consensus + robustness validation (always)
            ml_features — AFML features (opt-in; skipped by default for IND)

        Args:
            tickers: Stock symbols to evaluate.
            market: 'US' or 'IND'.
            date_range: (start_date_str, end_date_str) for backtests.
            skip_layers: Layer names to skip (e.g. ['ml_features']).
            max_workers: Thread-pool size for parallel layer execution.
            trade_horizon: 'swing' (3-10 day) or 'positional' (2-6 week).
                Strategy weights are adjusted based on horizon suitability.

        Returns:
            List of StockVerdict objects.
        """
        from datetime import date, timedelta

        if date_range is None:
            end = date.today()
            start = end - timedelta(days=365)
            date_range = (start.isoformat(), end.isoformat())

        skip = set(skip_layers or [])

        # ── Skip ML layer by default (opt-in for all markets) ──
        # AFML features (fractional diff, SADF, microstructure) are
        # academic and rarely change the final classification for
        # swing/positional trades.  Users can opt in by explicitly
        # passing skip_layers without 'ml_features'.
        if skip_layers is None:
            skip.add("ml_features")

        # If ML features are enabled, use extended weights
        if "ml_features" not in skip and "ml_features" not in self.weights:
            self.weights = dict(DEFAULT_WEIGHTS_WITH_ML)
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}

        run_id = str(uuid.uuid4())
        verdicts: List[StockVerdict] = []

        # ── Normalise Indian tickers to yfinance format (.NS) ──
        # Raw NSE symbols (e.g. "BPCL") won't resolve in yfinance
        # and won't trigger the Indian-specific code paths in
        # MetricsCalculator (_is_indian_ticker checks for .NS/.BO).
        if market == "IND":
            from utils import yf_nse_symbol
            tickers = [
                yf_nse_symbol(t) if not t.upper().endswith((".NS", ".BO")) else t
                for t in tickers
            ]

        # ── Survivorship bias gate ─────────────────────────────
        # Reject delisted / suspended tickers before running
        # expensive layer evaluations.
        try:
            from services.survivorship_filter import filter_valid_tickers
            valid_tickers, rejected = filter_valid_tickers(
                tickers, market=market,
            )
            for r in rejected:
                verdicts.append(StockVerdict(
                    ticker=r.ticker, market=market,
                    final_score=0.0, classification="HOLD",
                    layer_scores={}, layer_details={"survivorship_rejected": r.reason},
                    confidence=0.0, run_id=run_id,
                ))
            tickers = valid_tickers
        except Exception as exc:
            logger.debug("Survivorship filter skipped in scorer: %s", exc)

        for ticker in tickers:
            t0 = time.time()
            logger.info("IntegratedScorer: evaluating %s (%s)", ticker, market)

            layer_results: Dict[str, Dict[str, Any]] = {}

            # Core and strategy run in parallel; ML is optional.
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scorer") as pool:
                if "core" not in skip:
                    futures["core"] = pool.submit(_run_layer_core, ticker, market)
                if "strategy" not in skip:
                    futures["strategy"] = pool.submit(
                        _run_layer_strategy, ticker, market, date_range,
                        trade_horizon,
                    )
                if "ml_features" not in skip:
                    futures["ml_features"] = pool.submit(_run_layer_ml, ticker, market)
                # RL Bot layer (opt-in via Config.RL_ENABLED)
                if "rl_bot" not in skip and Config.RL_ENABLED:
                    try:
                        from services.rl_bot.rl_signal_integrator import run_rl_layer
                        futures["rl_bot"] = pool.submit(run_rl_layer, ticker, market)
                    except ImportError:
                        pass

                for layer_name, fut in futures.items():
                    try:
                        layer_results[layer_name] = fut.result(timeout=120)
                    except Exception as exc:
                        logger.warning("Layer %s timed out / failed for %s: %s",
                                       layer_name, ticker, exc)
                        layer_results[layer_name] = {
                            "score": None,
                            "details": {"error": str(exc)},
                        }

            # ── Regime-adaptive weight overrides ──────────────────
            regime_info: Dict[str, Any] = {}
            try:
                from services.regime_detector import regime_detector
                snapshot = regime_detector.detect()
                regime_info = {
                    "regime": snapshot.regime.value,
                    "position_scale": snapshot.position_scale,
                    "vix_level": snapshot.vix_panic,
                }
                # During turbulence, favour strategy layer (which now
                # includes robustness validation) over core.
                if snapshot.regime.value in ("HIGH_VOLATILITY", "CRISIS"):
                    effective_weights = dict(self.weights)
                    effective_weights["strategy"] = self.weights.get("strategy", 0.55) * 1.2
                    effective_weights["core"] = self.weights.get("core", 0.45) * 0.8
                    tw = sum(effective_weights.values())
                    effective_weights = {k: v / tw for k, v in effective_weights.items()}
                else:
                    effective_weights = dict(self.weights)
            except Exception:
                effective_weights = dict(self.weights)

            # ── Fundamental freshness adjustment (#9) ─────────
            freshness_adj = 0.0
            freshness_info: Dict[str, Any] = {}
            try:
                from services.fundamental_freshness import get_freshness_adjustment
                fadj = get_freshness_adjustment(ticker)
                freshness_adj = fadj.adjustment_score
                freshness_info = {
                    "adjustment": round(freshness_adj, 4),
                    "bulk_deals": fadj.bulk_deal_detected,
                    "pledge_change": round(fadj.promoter_pledge_change_pct, 2),
                    "mf_change": round(fadj.mf_holding_change_pct, 2),
                }
            except Exception:
                pass  # freshness data unavailable — degrade gracefully

            # ── Aggregate (with effective weights) ──
            layer_scores: Dict[str, Optional[float]] = {}
            layer_details_out: Dict[str, Any] = {}
            available_weight = 0.0
            weighted_sum = 0.0

            for layer_name, w in effective_weights.items():
                res = layer_results.get(layer_name, {})
                sc = res.get("score")
                layer_scores[layer_name] = round(sc, 4) if sc is not None else None
                layer_details_out[layer_name] = res.get("details", {})

                if sc is not None:
                    # Apply freshness adjustment to core layer
                    adjusted_sc = sc
                    if layer_name == "core" and freshness_adj != 0:
                        adjusted_sc = _clamp(sc + freshness_adj)
                    weighted_sum += adjusted_sc * w
                    available_weight += w

            if available_weight > 0:
                final_score = _clamp(weighted_sum / available_weight)
            else:
                final_score = 0.0

            # ── FII/DII gating (#10) — suppress BUY during heavy outflows ──
            fii_info: Dict[str, Any] = {}
            try:
                from scrapers.macro.fii_dii_tracker import compute_fii_dii_signal
                fii_signal = compute_fii_dii_signal()
                fii_info = {
                    "sentiment_score": round(fii_signal.sentiment_score, 3),
                    "consecutive_fii_selling": fii_signal.consecutive_fii_selling_days,
                    "is_heavy": fii_signal.is_heavy_fii_selling,
                }
                if fii_signal.is_heavy_fii_selling and final_score > 0:
                    final_score = min(final_score, Config.BUY_THRESHOLD - 0.01)
                elif fii_signal.is_fii_selling_pressure and final_score > 0:
                    final_score *= 0.8
                    final_score = _clamp(final_score)
            except Exception:
                pass

            # Merge extra details
            layer_details_out["regime"] = regime_info
            layer_details_out["fii_dii"] = fii_info
            if freshness_info:
                layer_details_out["freshness"] = freshness_info

            # Confidence = fraction of layers that returned data
            active_layers = sum(1 for s in layer_scores.values() if s is not None)
            total_layers = len(effective_weights)
            confidence = active_layers / total_layers if total_layers else 0.0

            verdict = StockVerdict(
                ticker=ticker,
                market=market,
                final_score=round(final_score, 4),
                classification=_classify(final_score),
                layer_scores=layer_scores,
                layer_details=layer_details_out,
                confidence=round(confidence, 2),
                run_id=run_id,
            )
            verdicts.append(verdict)
            elapsed = time.time() - t0
            logger.info(
                "IntegratedScorer: %s → %s (%.2f) in %.1fs [confidence=%.0f%%]",
                ticker, verdict.classification, final_score, elapsed,
                confidence * 100,
            )

        # ── Persist verdicts ──
        self._persist(verdicts, run_id, market, date_range)

        return verdicts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, verdicts: List[StockVerdict], run_id: str, market: str, date_range: tuple):
        """Save verdicts to PostgreSQL and MinIO."""
        db = _get_db_service()
        if db and db.is_available:
            try:
                analysis_run_id = db.start_analysis_run(
                    run_type="integrated_verdict",
                    tickers=[v.ticker for v in verdicts],
                    parameters={"weights": self.weights},
                    market=market,
                )
                for v in verdicts:
                    result_dict = {
                        "strategy_id": "integrated_verdict",
                        "strategy_name": "Integrated Verdict",
                        "tickers": [v.ticker],
                        "start_date": date_range[0],
                        "end_date": date_range[1],
                        "initial_capital": 0,
                        "total_return": v.final_score,
                        "sharpe_ratio": v.layer_scores.get("ml_features"),
                        "max_drawdown": None,
                        "parameters": {"weights": self.weights},
                        "metrics": {
                            "final_score": v.final_score,
                            "classification": v.classification,
                            "layer_scores": v.layer_scores,
                            "layer_details": v.layer_details,
                            "confidence": v.confidence,
                            "run_id": v.run_id,
                        },
                    }
                    db.save_backtest_result(result_dict, analysis_run_id, market=market)

                db.complete_analysis_run(analysis_run_id, total_signals=len(verdicts))
                logger.info("Verdicts persisted to DB (run=%s)", analysis_run_id)
            except Exception as exc:
                logger.warning("DB persistence failed: %s", exc)

        # MinIO — save radar chart if matplotlib available
        minio = _get_minio()
        if minio and minio.is_available:
            for v in verdicts:
                try:
                    img_bytes = _render_radar_chart(v)
                    if img_bytes:
                        minio.save_backtest_image(
                            run_id=f"verdict_{run_id}",
                            image_data=img_bytes,
                            filename=f"{v.ticker}_radar.png",
                            strategy_name="integrated_verdict",
                            ticker=v.ticker,
                            chart_title=f"{v.ticker} — {v.classification}",
                        )
                except Exception as exc:
                    logger.debug("MinIO radar save failed for %s: %s", v.ticker, exc)


def _render_radar_chart(verdict: StockVerdict) -> Optional[bytes]:
    """Render a radar chart of layer scores as PNG bytes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(verdict.layer_scores.keys())
        values = [verdict.layer_scores.get(l) or 0 for l in labels]
        n = len(labels)
        if n < 3:
            return None

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.plot(angles, values, color="steelblue", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(-1, 1)
        ax.set_title(
            f"{verdict.ticker}  {verdict.classification}  ({verdict.final_score:+.2f})",
            fontsize=12, pad=20,
        )

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None
