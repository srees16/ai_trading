"""
Portfolio Allocator — Dual-Strategy Orchestration.

Runs Centurion Compounder (CC) and Centurion Harvest (CH) backtests
with a user-defined capital split and returns combined results.
"""

import logging
import os
import time
from typing import Dict, Optional

from config import Config
from services.forecast_combiner import DEFAULT_FORECAST_WEIGHTS

logger = logging.getLogger(__name__)

# Absolute path to centurion_core/ root (matches Kaggle runner convention)
_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Harvest parameter presets
HARVEST_PRESETS = {
    "conservative": {"inject_pct": 0.10, "book_pct": 0.05, "sustain_days": 90,
                     "min_gain_to_book": 0.15, "inject_cooldown_days": 300},
    "balanced":     {"inject_pct": 0.15, "book_pct": 0.10, "sustain_days": 60,
                     "min_gain_to_book": 0.10, "inject_cooldown_days": 200},
    "aggressive":   {"inject_pct": 0.20, "book_pct": 0.15, "sustain_days": 30,
                     "min_gain_to_book": 0.10, "inject_cooldown_days": 200},
}




def _configure_base(bt_mod):
    """Set R21A base configuration and weights on the backtest module."""
    # Enable R21A regime scaling
    bt_mod._R21A_REGIME_VOL = True
    bt_mod._R21A_REGIME_BOOST = 1.25
    bt_mod._R21A_REGIME_DEFEND = 0.15

    # Apply weights: prefer optimized pkl, fall back to DEFAULT_FORECAST_WEIGHTS (R21A)
    import pickle
    opt_path = os.path.join(_CORE_DIR, "data", "r21a_optimization_results.pkl")
    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt = pickle.load(f)
        weights = opt["best_weights"]
        # Overwrite DEFAULT_FORECAST_WEIGHTS from pkl
        for fw in DEFAULT_FORECAST_WEIGHTS:
            if fw.name in weights:
                fw.weight = weights[fw.name]
            elif fw.weight > 0:
                fw.weight = 0.0  # zero out any signal not in optimized set
    else:
        # DEFAULT_FORECAST_WEIGHTS already contains R21A-optimized values
        logger.warning("No r21a_optimization_results.pkl found — using DEFAULT_FORECAST_WEIGHTS (R21A)")
        weights = {fw.name: fw.weight for fw in DEFAULT_FORECAST_WEIGHTS}

    return weights


def _run_single(bt_mod, capital: float, start_date: str, end_date: str, label: str) -> Optional[Dict]:
    """Run a single backtest with current module configuration."""
    try:
        return bt_mod.run_full_backtest(
            tickers=None,
            capital=capital,
            period="13y",
            market="IND",
            verbose=False,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error("Backtest failed for %s: %s", label, e)
        return None


def _extract_metrics(result: Dict, name: str, capital: float) -> Dict:
    """Extract standardized metrics dict from backtest result."""
    return {
        "strategy_name": name,
        "capital_allocated": capital,
        "final_equity": result.get("daily_equity", [capital])[-1] if result.get("daily_equity") else capital,
        "sharpe": result.get("sharpe", 0.0),
        "sortino": result.get("sortino", 0.0),
        "calmar": result.get("calmar", 0.0),
        "cagr_pct": result.get("annual_return_pct", 0.0),
        "max_drawdown_pct": result.get("max_drawdown_pct", 0.0),
        "total_return_pct": result.get("total_return_pct", 0.0),
        "total_trades": result.get("n_trades", 0),
        "win_rate": result.get("win_rate", 0.0),
        "profit_factor": result.get("profit_factor", 0.0),
    }


def run_dual_backtest(
    total_capital: float = 1_000_000,
    compounder_pct: float = 50.0,
    harvest_params: Optional[Dict] = None,
    start_date: str = "2012-01-01",
    end_date: str = "2025-12-31",
) -> Dict:
    """
    Run Centurion Compounder + Centurion Harvest backtests with a capital split.

    Returns a dict matching PortfolioBacktestResponse schema.
    """
    t0 = time.time()
    harvest_pct = 100.0 - compounder_pct
    cc_capital = total_capital * (compounder_pct / 100.0)
    ch_capital = total_capital * (harvest_pct / 100.0)

    # Resolve harvest preset
    hp = dict(HARVEST_PRESETS["aggressive"])  # default
    if harvest_params:
        if harvest_params.get("preset") and harvest_params["preset"] in HARVEST_PRESETS:
            hp = dict(HARVEST_PRESETS[harvest_params["preset"]])
        # Override with explicit params
        for k in ("inject_pct", "book_pct", "sustain_days", "min_gain_to_book", "inject_cooldown_days"):
            if k in harvest_params and harvest_params[k] is not None:
                hp[k] = harvest_params[k]

    # ── Run 1: Centurion Compounder (CC) ──
    import services.full_pipeline_backtest as bt_mod
    _configure_base(bt_mod)
    # Explicitly disable ALL Harvest flags (defensive — mirrors R21A baseline exactly)
    bt_mod._HARVEST_DIP_BUYER = False
    bt_mod._HARVEST_PROFIT_TAKER = False
    bt_mod._HARVEST_ENABLED = False
    bt_mod._HARVEST_MR_BEAR_VOL_MULT = 3.33   # default (unused when DIP_BUYER=False)
    bt_mod._HARVEST_BULL_STOP_SIGMA = 6.0      # default (unused when PROFIT_TAKER=False)
    bt_mod._HARVEST_INJECT_PCT = 0.20
    bt_mod._HARVEST_BOOK_PCT = 0.15
    bt_mod._HARVEST_BULL_SUSTAIN_DAYS = 30
    bt_mod._HARVEST_MIN_GAIN_TO_BOOK = 0.10
    bt_mod._HARVEST_INJECT_COOLDOWN_DAYS = 200
    # Isolate checkpoint path so CC/CH never collide
    os.environ["CENTURION_BT_CHECKPOINT"] = os.path.join(
        _CORE_DIR, "data", "backtest_checkpoint_cc.pkl")
    logger.info("Running Centurion Compounder (CC) with ₹%s ...", f"{cc_capital:,.0f}")
    cc_result = _run_single(bt_mod, cc_capital, start_date, end_date, Config.STRATEGY_COMPOUNDER)

    # ── Run 2: Centurion Harvest (CH) ──
    import importlib
    importlib.reload(bt_mod)   # Clean module state between runs
    _configure_base(bt_mod)    # Same R21A base + weights as CC
    bt_mod._HARVEST_DIP_BUYER = True
    bt_mod._HARVEST_PROFIT_TAKER = True
    bt_mod._HARVEST_ENABLED = True
    bt_mod._HARVEST_MR_BEAR_VOL_MULT = hp.get("mr_bear_vol_mult", 3.33)
    bt_mod._HARVEST_BULL_STOP_SIGMA = hp.get("bull_stop_sigma", 6.0)
    bt_mod._HARVEST_INJECT_PCT = hp["inject_pct"]
    bt_mod._HARVEST_BOOK_PCT = hp["book_pct"]
    bt_mod._HARVEST_BULL_SUSTAIN_DAYS = hp["sustain_days"]
    bt_mod._HARVEST_MIN_GAIN_TO_BOOK = hp["min_gain_to_book"]
    bt_mod._HARVEST_INJECT_COOLDOWN_DAYS = hp["inject_cooldown_days"]
    # Isolate checkpoint path for CH
    os.environ["CENTURION_BT_CHECKPOINT"] = os.path.join(
        _CORE_DIR, "data", "backtest_checkpoint_ch.pkl")
    logger.info("Running Centurion Harvest (CH) with ₹%s ...", f"{ch_capital:,.0f}")
    ch_result = _run_single(bt_mod, ch_capital, start_date, end_date, Config.STRATEGY_HARVEST)

    elapsed = time.time() - t0

    # ── Build response ──
    cc_metrics = _extract_metrics(cc_result, Config.STRATEGY_COMPOUNDER, cc_capital) if cc_result else None
    ch_metrics = _extract_metrics(ch_result, Config.STRATEGY_HARVEST, ch_capital) if ch_result else None

    # Equity curves (sampled to max 500 points for frontend)
    def _sample_equity(daily_eq, max_pts=500):
        if not daily_eq:
            return []
        step = max(1, len(daily_eq) // max_pts)
        return [{"day": i, "equity": round(daily_eq[i], 2)}
                for i in range(0, len(daily_eq), step)]

    cc_equity = _sample_equity(cc_result.get("daily_equity", [])) if cc_result else []
    ch_equity = _sample_equity(ch_result.get("daily_equity", [])) if ch_result else []

    # Harvest events
    harvest_summary = None
    cr = ch_result.get("capital_rotation") if ch_result else None
    if cr:
        def _to_events(events, etype):
            return [{"day": ev[0], "amount": round(ev[1], 2),
                     "equity_before": round(ev[2], 2), "equity_after": round(ev[3], 2),
                     "event_type": etype} for ev in events]

        harvest_summary = {
            "total_injected": cr["total_injected"],
            "total_booked": cr["total_booked"],
            "net_extracted": cr["total_booked"] - cr["total_injected"],
            "inject_events": _to_events(cr.get("inject_events", []), "inject"),
            "book_events": _to_events(cr.get("book_events", []), "book"),
        }

    # Combined wealth
    cc_final = cc_metrics["final_equity"] if cc_metrics else 0
    ch_final = ch_metrics["final_equity"] if ch_metrics else 0
    ch_net_extracted = harvest_summary["net_extracted"] if harvest_summary else 0
    combined_wealth = cc_final + ch_final + ch_net_extracted
    combined_return_pct = ((combined_wealth / total_capital) - 1) * 100 if total_capital > 0 else 0

    return {
        "total_capital": total_capital,
        "compounder_pct": compounder_pct,
        "harvest_pct": harvest_pct,
        "compounder": cc_metrics,
        "harvest": ch_metrics,
        "combined_wealth": round(combined_wealth, 2),
        "combined_return_pct": round(combined_return_pct, 1),
        "compounder_equity": cc_equity,
        "harvest_equity": ch_equity,
        "harvest_summary": harvest_summary,
        "execution_time_sec": round(elapsed, 1),
    }
