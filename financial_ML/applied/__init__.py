"""
financial_ML.applied — chapter registry, async batch runner, and progress tracking.
"""

import asyncio
import io
import logging
import runpy
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_APPLIED_DIR = Path(__file__).resolve().parent

# ── Chapter registry ─────────────────────────────────────────────────────

_CHAPTERS = [
    # Data Structures
    {"key": "ch02", "title": "Financial Data Structures",
     "category": "Data Structures",
     "description": "PCA weights, roll-gap adjustment, CUSUM filter"},
    {"key": "ch03", "title": "Labeling (Triple-Barrier)",
     "category": "Data Structures",
     "description": "Daily vol, triple-barrier labels, CUSUM events"},
    {"key": "ch04", "title": "Sample Weights",
     "category": "Data Structures",
     "description": "Concurrent events, uniqueness, sequential bootstrap"},
    # Features
    {"key": "ch05", "title": "Fractional Differentiation",
     "category": "Features",
     "description": "FFD stationarity transform — find min d for ADF"},
    {"key": "ch08", "title": "Feature Importance",
     "category": "Features",
     "description": "MDI, MDA, SFI importance with PCA orthogonalisation"},
    {"key": "ch17", "title": "Structural Breaks",
     "category": "Features",
     "description": "SADF, Chu-Stinchcombe-White CUSUM, SMT tests"},
    {"key": "ch18", "title": "Entropy Features",
     "category": "Features",
     "description": "Plug-in, Lempel-Ziv, Kontoyiannis entropy estimators"},
    {"key": "ch19", "title": "Microstructural Features",
     "category": "Features",
     "description": "Corwin-Schultz spread, tick rule, Kyle/Amihud lambda, VPIN"},
    # Modeling
    {"key": "ch06", "title": "Ensemble Methods",
     "category": "Modeling",
     "description": "Bagging accuracy, three RF setups, cross-validated scores"},
    {"key": "ch07", "title": "Cross-Validation",
     "category": "Modeling",
     "description": "Purged K-Fold, embargo, time-aware CV"},
    {"key": "ch09", "title": "Hyper-Parameter Tuning",
     "category": "Modeling",
     "description": "Grid & random search with purged CV"},
    {"key": "ch10", "title": "Bet Sizing",
     "category": "Modeling",
     "description": "Signal → position sizing, discretisation, limit prices"},
    # Backtesting
    {"key": "ch11", "title": "Dangers of Backtesting",
     "category": "Backtesting",
     "description": "Selection bias, deflated SR, probability of backtest overfitting"},
    {"key": "ch13", "title": "Synthetic Backtesting",
     "category": "Backtesting",
     "description": "Ornstein-Uhlenbeck optimal trading rules"},
    {"key": "ch14", "title": "Backtest Statistics",
     "category": "Backtesting",
     "description": "Sharpe, PSR, DSR, drawdowns, HHI concentration"},
    {"key": "ch15", "title": "Strategy Risk",
     "category": "Backtesting",
     "description": "Implied precision, betting frequency, failure probability"},
    # Portfolio
    {"key": "ch16", "title": "ML Asset Allocation",
     "category": "Portfolio",
     "description": "Hierarchical Risk Parity (HRP) vs CLA vs IVP"},
    # Computation
    {"key": "ch20", "title": "Multiprocessing",
     "category": "Computation",
     "description": "Vectorisation benchmarks, barrier touch, parallel partitioning"},
    {"key": "ch21", "title": "Brute Force & Quantum",
     "category": "Computation",
     "description": "Combinatorial portfolio optimisation, dynamic vs static SR"},
]

# Map chapter key → script filename
_CHAPTER_SCRIPTS: Dict[str, str] = {}
for f in sorted(_APPLIED_DIR.glob("ch*.py")):
    key = f.stem.split("_", 1)[0]  # e.g. "ch02"
    _CHAPTER_SCRIPTS[key] = f.name


def get_chapters() -> List[Dict[str, str]]:
    """Return the chapter registry for the API."""
    return _CHAPTERS


# ── Batch progress tracking ──────────────────────────────────────────────

_batch_lock = threading.Lock()
_batch_progress: Dict[str, Dict[str, Any]] = {}


def get_batch_progress(batch_id: str) -> Optional[Dict[str, Any]]:
    with _batch_lock:
        return _batch_progress.get(batch_id)


def _execute_chapter(
    ch_key: str,
    tickers: Optional[List[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single chapter script and capture output + figures."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result: Dict[str, Any] = {
        "chapter_key": ch_key,
        "status": "done",
        "figures": [],
        "tables": [],
        "text_output": "",
        "error_message": None,
    }

    script_file = _CHAPTER_SCRIPTS.get(ch_key)
    if not script_file:
        result["status"] = "error"
        result["error_message"] = f"Unknown chapter: {ch_key}"
        return result

    script_path = _APPLIED_DIR / script_file
    if not script_path.exists():
        result["status"] = "error"
        result["error_message"] = f"Script not found: {script_file}"
        return result

    figs_before = set(plt.get_fignums())
    stdout_capture = io.StringIO()

    # Inject tickers / date range as env vars so chapter scripts can read them
    env_overrides = {}
    if tickers:
        env_overrides["FML_TICKERS"] = ",".join(tickers)
    if date_start:
        env_overrides["FML_DATE_START"] = date_start
    if date_end:
        env_overrides["FML_DATE_END"] = date_end

    old_env = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v

    # Ensure `from sample_data import ...` resolves to financial_ML/sample_data.py
    # rather than testune_trade_sys/sample_data.py which lacks some functions.
    fml_dir = str(_APPLIED_DIR.parent)
    path_inserted = fml_dir not in sys.path
    if path_inserted:
        sys.path.insert(0, fml_dir)

    try:
        with redirect_stdout(stdout_capture):
            runpy.run_path(str(script_path), run_name="__main__")
    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = str(exc)
        logger.exception("FML chapter %s failed", ch_key)
    finally:
        if path_inserted and fml_dir in sys.path:
            sys.path.remove(fml_dir)
        # Restore env vars
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    result["text_output"] = stdout_capture.getvalue()

    # Capture new figures as base64 PNG
    import base64
    new_figs = set(plt.get_fignums()) - figs_before
    for fig_num in sorted(new_figs):
        try:
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            result["figures"].append(base64.b64encode(buf.read()).decode())
            plt.close(fig)
        except Exception:
            pass

    return result


async def run_chapters_async(
    batch_id: str,
    chapter_keys: List[str],
    tickers: Optional[List[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
):
    """Run chapters in background and update progress."""
    total = len(chapter_keys)
    with _batch_lock:
        _batch_progress[batch_id] = {
            "batch_id": batch_id,
            "total": total,
            "completed": 0,
            "chapters": {
                k: {"chapter_key": k, "status": "pending", "figures": [], "tables": [], "text_output": "", "error_message": None}
                for k in chapter_keys
            },
        }

    for ch_key in chapter_keys:
        with _batch_lock:
            _batch_progress[batch_id]["chapters"][ch_key]["status"] = "running"

        result = await asyncio.to_thread(_execute_chapter, ch_key, tickers, date_start, date_end)

        with _batch_lock:
            _batch_progress[batch_id]["chapters"][ch_key] = result
            _batch_progress[batch_id]["completed"] += 1
