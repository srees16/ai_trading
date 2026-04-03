"""
vince.applied — chapter registry, async batch runner, and progress tracking.

Ralph Vince — Leverage Space Risk Lab.
"""

import asyncio
import io
import logging
import os
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
    # Position Sizing
    {"key": "ch01", "title": "Optimal f & TWR",
     "category": "Position Sizing",
     "description": "Growth-optimal fraction, Terminal Wealth Relative, geometric mean maximization"},
    # Portfolio
    {"key": "ch02", "title": "Leverage Space Portfolio",
     "category": "Portfolio",
     "description": "Multi-asset optimal f allocation with drawdown constraints"},
    # Monte Carlo
    {"key": "ch03", "title": "Monte Carlo Risk Simulation",
     "category": "Monte Carlo",
     "description": "Probability of profit, CAGR distribution, max drawdown at N-year horizon"},
]

_CHAPTER_SCRIPTS: Dict[str, str] = {}
for f in sorted(_APPLIED_DIR.glob("ch*.py")):
    key = f.stem.split("_", 1)[0]
    _CHAPTER_SCRIPTS[key] = f.name


def get_chapters() -> List[Dict[str, str]]:
    return _CHAPTERS


# ── Batch progress tracking ──────────────────────────────────────────────

_batch_lock = threading.Lock()
_batch_progress: Dict[str, Dict[str, Any]] = {}
_abort_flags: set = set()


def get_batch_progress(batch_id: str) -> Optional[Dict[str, Any]]:
    with _batch_lock:
        return _batch_progress.get(batch_id)


def abort_batch(batch_id: str) -> bool:
    with _batch_lock:
        if batch_id not in _batch_progress:
            return False
        _abort_flags.add(batch_id)
        return True


def _execute_chapter(
    ch_key: str,
    tickers: Optional[List[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result: Dict[str, Any] = {
        "chapter_key": ch_key, "status": "done",
        "figures": [], "tables": [], "text_output": "", "error_message": None,
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

    env_overrides = {}
    if tickers:
        env_overrides["VINCE_TICKERS"] = ",".join(tickers)
    if date_start:
        env_overrides["VINCE_DATE_START"] = date_start
    if date_end:
        env_overrides["VINCE_DATE_END"] = date_end

    old_env = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v

    vince_dir = str(_APPLIED_DIR.parent)
    if vince_dir in sys.path:
        sys.path.remove(vince_dir)
    sys.path.insert(0, vince_dir)

    for _key in [k for k in sys.modules if k == "sample_data" or k.startswith("sample_data.")]:
        sys.modules.pop(_key, None)
    import importlib
    _sd = importlib.import_module("sample_data")
    if tickers:
        _sd.SYMBOLS = list(tickers)
    if date_start:
        _sd.DEFAULT_START = date_start
    if date_end:
        _sd.DEFAULT_END = date_end

    try:
        with redirect_stdout(stdout_capture):
            runpy.run_path(str(script_path), run_name="__main__")
    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = str(exc)
        logger.exception("Vince chapter %s failed", ch_key)
    finally:
        if vince_dir in sys.path:
            sys.path.remove(vince_dir)
        sys.modules.pop("sample_data", None)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    result["text_output"] = stdout_capture.getvalue()

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
    total = len(chapter_keys)
    with _batch_lock:
        _batch_progress[batch_id] = {
            "batch_id": batch_id, "total": total, "completed": 0,
            "chapters": {
                k: {"chapter_key": k, "status": "pending", "figures": [], "tables": [], "text_output": "", "error_message": None}
                for k in chapter_keys
            },
        }

    for ch_key in chapter_keys:
        with _batch_lock:
            if batch_id in _abort_flags:
                for remaining in chapter_keys[chapter_keys.index(ch_key):]:
                    _batch_progress[batch_id]["chapters"][remaining]["status"] = "cancelled"
                _batch_progress[batch_id]["status"] = "aborted"
                _abort_flags.discard(batch_id)
                return
            _batch_progress[batch_id]["chapters"][ch_key]["status"] = "running"

        result = await asyncio.to_thread(_execute_chapter, ch_key, tickers, date_start, date_end)

        with _batch_lock:
            _batch_progress[batch_id]["chapters"][ch_key] = result
            _batch_progress[batch_id]["completed"] += 1

    # Persist completed batch to DB + R2
    try:
        from references.batch_history import save_batch_history
        with _batch_lock:
            snapshot = dict(_batch_progress.get(batch_id, {}))
        await asyncio.to_thread(
            save_batch_history, batch_id, "vince", chapter_keys, snapshot,
            tickers, date_start, date_end,
        )
    except Exception:
        logger.warning("Failed to save Vince batch history", exc_info=True)
