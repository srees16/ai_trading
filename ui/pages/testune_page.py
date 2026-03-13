"""
TestTune Trading System Page Module for Centurion Capital LLC.

Interactive UI for *Testing and Tuning Market Trading Systems* (Timothy
Masters, 2018) chapter analyses.  Users select stock tickers (or use
defaults), then explore results across tabbed categories that map to the
``testune_trade_sys/applied/`` scripts.
"""

import io
import json
import sys
import logging
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from io import StringIO
from contextlib import redirect_stdout

import streamlit as st

from config import Config
from ui.components import (
    render_header,
    render_footer,
    spinner_html,
)
from ui.styles import Colors
from utils import parse_ticker_csv, validate_tickers

logger = logging.getLogger(__name__)

DB_AVAILABLE = Config.is_database_configured()
MINIO_AVAILABLE = True


def _get_db_service():
    """Lazy import of database.service."""
    from database.service import get_database_service
    return get_database_service()


def _get_minio():
    """Lazy import of storage.minio_service."""
    from storage.minio_service import get_minio_service
    return get_minio_service()

# ── Path setup: allow imports from testune_trade_sys/ ──────────
_TTS_ROOT = Path(__file__).resolve().parent.parent.parent / "testune_trade_sys"
if str(_TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TTS_ROOT))

# Default tickers (same as testune sample_data.SYMBOLS)
TTS_DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]

# ── Tab / chapter registry ──────────────────────────────────────────
# Grouped into logical analysis categories.  Each entry:
#   (tab_label, [(chapter_key, display_name, description, needs_tickers), ...])

ANALYSIS_TABS: List[tuple] = [
    ("Foundations", [
        ("ch01", "Introduction",
         "Log/simple returns, future leak detection, percent wins analysis", True),
        ("ch02", "Pre-Optimization Issues",
         "Stationarity, entropy, indicator oscillation, tail cleaning", True),
    ]),
    ("Optimization", [
        ("ch03", "Optimization Issues",
         "Elastic-net coordinate descent, differential evolution, CV lambda search", True),
        ("ch04", "Post-Optimization Issues",
         "StocBias debiasing, parameter relationships, sensitivity curves", True),
    ]),
    ("Performance Estimation", [
        ("ch05", "Unbiased Performance Estimation",
         "Walk-forward, trading CV, CSCV superiority, nested walk-forward", True),
        ("ch06", "Trade-Based Analysis",
         "BCa bootstrap, parametric confidence, drawdown bounds", True),
    ]),
    ("Statistical Testing", [
        ("ch07", "Permutation Tests",
         "Return/price/bar permutation, walk-forward permutation, partition return", True),
    ]),
]

# Chapter-key → display-name lookup (for progress messages)
_CH_NAME_MAP: Dict[str, str] = {
    ch_key: ch_name
    for _, chapters in ANALYSIS_TABS
    for ch_key, ch_name, _, _ in chapters
}

# Chapter-key → short acronym for MinIO folder/file naming clarity
_CH_ACRONYM: Dict[str, str] = {
    "ch01": "intro",   # Introduction
    "ch02": "preopt",  # Pre-Optimization Issues
    "ch03": "opt",     # Optimization Issues
    "ch04": "postopt", # Post-Optimization Issues
    "ch05": "ubperf",  # Unbiased Performance Estimation
    "ch06": "trade",   # Trade-Based Analysis
    "ch07": "perm",    # Permutation Tests
}


def _ch_tag(ch_key: str) -> str:
    """Return `ch05_ubperf` style tag for MinIO paths."""
    acr = _CH_ACRONYM.get(ch_key)
    return f"{ch_key}_{acr}" if acr else ch_key

# ── Async pre-computation state (thread-safe, module-level) ─────────
_ASYNC_LOCK = threading.Lock()
_ASYNC_JOBS: Dict[str, Dict[str, Any]] = {}


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════
def render_testune_page():
    """Render the TestTune Trading System module page."""
    _user = st.session_state.get("username", "unknown")
    logger.info("[user=%s] Viewing TestTune Trading System page", _user)

    render_header()

    # ── Title ───────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center; padding:0.4rem 0 0.2rem;">
            <span style="font-size:1.5rem; font-weight:700;
                         color:{Colors.TEXT_PRIMARY};">
                Testing and Tuning Market Trading Systems
            </span><br>
            <span style="font-size:0.85rem; color:{Colors.TEXT_MUTED};">
                Based on the book by Timothy Masters (2018)
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── History navigation ──────────────────────────────────────
    if st.button("View Saved Results", key="tts_goto_history"):
        st.session_state["current_page"] = "tts_history"
        st.rerun()

    # ── Ticker input (left) + settings (right) ─────────────────
    _inject_compact_css()
    col1, col2 = st.columns([1, 1])

    with col1:
        tickers = _render_ticker_selection()

    with col2:
        date_start, date_end = _render_date_settings()

    # Store in session
    st.session_state["tts_tickers"] = tickers
    st.session_state["tts_date_start"] = date_start
    st.session_state["tts_date_end"] = date_end

    # ── Run All button ──────────────────────────────────────────
    _render_run_all_button(tickers, date_start, date_end)

    # ── Async progress (auto-refreshing fragment) ───────────────
    if st.session_state.get("tts_async_batch_id"):
        _render_async_progress()

    # ── Batch-completion banner ─────────────────────────────────
    _tts_banner = st.session_state.pop("_tts_batch_banner", None)
    if _tts_banner:
        getattr(st, _tts_banner["kind"])(_tts_banner["msg"])

    st.markdown("---")

    # ── Analysis tabs ───────────────────────────────────────────
    _render_analysis_tabs(tickers, date_start, date_end)

    render_footer()

    if st.session_state.pop("_tts_needs_rerun", False):
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# Ticker selection
# ════════════════════════════════════════════════════════════════════════
def _render_ticker_selection() -> List[str]:
    st.markdown("**Select Stocks**")
    ticker_mode = st.radio(
        "Input method:",
        ["Default Tickers", "Manual Entry", "Upload CSV"],
        help="Choose how to specify the stock tickers for analysis",
        horizontal=True,
        key="tts_ticker_mode",
    )
    if ticker_mode == "Default Tickers":
        return _handle_default_tickers()
    elif ticker_mode == "Manual Entry":
        return _handle_manual_entry()
    else:
        return _handle_csv_upload()


def _handle_default_tickers() -> List[str]:
    with st.expander("View default tickers"):
        st.write(", ".join(TTS_DEFAULT_TICKERS))
    return list(TTS_DEFAULT_TICKERS)


def _handle_manual_entry() -> List[str]:
    ticker_input = st.text_area(
        "Enter tickers (comma-separated):",
        value="SPY, QQQ, IWM, DIA",
        height=80,
        help="Enter US stock symbols separated by commas.",
        key="tts_manual_tickers",
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    return tickers


def _handle_csv_upload() -> List[str]:
    with st.expander("View CSV format example"):
        sample = "Ticker\nSPY\nQQQ\nIWM\nDIA\n"
        st.code(sample, language="csv")
        st.download_button(
            label="Download Sample CSV",
            data=sample,
            file_name="sample_tts_tickers.csv",
            mime="text/csv",
            key="tts_csv_download",
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV with ticker symbols",
        key="tts_csv_upload",
    )
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            parsed = parse_ticker_csv(content)
            if parsed:
                valid, invalid = validate_tickers(parsed)
                st.success(f"Found {len(valid)} valid ticker(s)")
                if invalid:
                    st.warning(f"Skipped {len(invalid)} invalid ticker(s)")
                with st.expander("View uploaded tickers"):
                    st.write(", ".join(valid))
                return valid
            else:
                st.error("No valid tickers found in CSV")
        except Exception as e:
            logger.error("CSV parse error: %s", e)
            st.error(f"Error parsing CSV: {e}")
    return []


# ════════════════════════════════════════════════════════════════════════
# Date / settings panel
# ════════════════════════════════════════════════════════════════════════
def _render_date_settings():
    import datetime as _dt

    col_a, col_b = st.columns(2)
    with col_a:
        date_start = st.date_input(
            "Start date",
            value=_dt.date(2020, 1, 1),
            key="tts_start_date",
        )
    with col_b:
        date_end = st.date_input(
            "End date",
            value=_dt.date(2024, 12, 31),
            key="tts_end_date",
        )

    st.caption(
        "Date range applies only to chapters that fetch live market data. "
        "Chapters using generated / synthetic data are unaffected."
    )
    return str(date_start), str(date_end)


# ════════════════════════════════════════════════════════════════════════
# Run All – pre-compute every chapter
# ════════════════════════════════════════════════════════════════════════
def _render_run_all_button(tickers: List[str], date_start: str, date_end: str):
    async_running = bool(st.session_state.get("tts_async_batch_id"))
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        run_all = st.button(
            "Run Analyses",
            type="primary",
            width="stretch",
            disabled=not tickers or async_running,
            help="Pre-compute all chapter analyses in the background" if not async_running
                 else "Computation in progress",
            key="tts_run_all",
        )

    if run_all and tickers and not async_running:
        _start_async_precompute(tickers, date_start, date_end)
        st.rerun()


_PRECOMPUTE_SKIP: set = set()  # No chapters to skip by default


# ────────────────────────────────────────────────────────────────────────
# Async pre-computation (non-blocking)
# ────────────────────────────────────────────────────────────────────────
def _start_async_precompute(
    tickers: List[str], date_start: str, date_end: str,
):
    """Launch pre-computation in a background thread."""
    batch_id = uuid.uuid4().hex[:12]

    all_chapters = [
        (ch_key, ch_name, ch_desc, needs_tickers)
        for _, chapters in ANALYSIS_TABS
        for ch_key, ch_name, ch_desc, needs_tickers in chapters
    ]

    cached_keys: set = {
        ch_key for ch_key, *_ in all_chapters
        if st.session_state.get(f"tts_result_{ch_key}")
    }

    job: Dict[str, Any] = {
        "status": "running",
        "total": len(all_chapters),
        "completed": 0,
        "succeeded": 0,
        "skipped": 0,
        "chapters": {},
        "tickers": list(tickers),
        "run_id": _build_tts_run_id(tickers),
    }

    with _ASYNC_LOCK:
        _ASYNC_JOBS[batch_id] = job

    st.session_state["tts_async_batch_id"] = batch_id
    st.session_state["tts_async_synced"] = set()

    thread = threading.Thread(
        target=_async_worker,
        args=(batch_id, all_chapters, tickers, date_start, date_end, cached_keys),
        daemon=True,
    )
    thread.start()
    logger.info("Async TTS pre-computation started — batch %s", batch_id)


def _async_worker(
    batch_id: str,
    all_chapters: list,
    tickers: List[str],
    date_start: str,
    date_end: str,
    cached_keys: set = None,
):
    """Background worker — runs chapters sequentially, stores results."""
    job = _ASYNC_JOBS.get(batch_id)
    if job is None:
        return
    run_id = job["run_id"]
    cached_keys = cached_keys or set()

    for idx, (ch_key, ch_name, _, needs_tickers) in enumerate(all_chapters):
        with _ASYNC_LOCK:
            job["chapters"][ch_key] = {"status": "running", "ch_name": ch_name}

        logger.info("[async] TTS computing %d/%d: %s", idx + 1, job["total"], ch_key)

        if ch_key in cached_keys:
            with _ASYNC_LOCK:
                job["chapters"][ch_key] = {"status": "skipped", "ch_name": ch_name, "reason": "cached"}
                job["skipped"] += 1
                job["completed"] += 1
            continue

        if ch_key in _PRECOMPUTE_SKIP:
            with _ASYNC_LOCK:
                job["chapters"][ch_key] = {"status": "skipped", "ch_name": ch_name}
                job["skipped"] += 1
                job["completed"] += 1
            continue

        if needs_tickers and not tickers:
            with _ASYNC_LOCK:
                job["chapters"][ch_key] = {"status": "skipped", "ch_name": ch_name, "reason": "no_tickers"}
                job["skipped"] += 1
                job["completed"] += 1
            continue

        try:
            result = _execute_chapter(ch_key, tickers, date_start, date_end, needs_tickers)
            fig_count = _save_tts_figures_to_minio(run_id, ch_key, ch_name, result)
            json_ok = _save_tts_text_to_minio(run_id, ch_key, ch_name, result)
            db_ok = _save_tts_to_database(ch_key, ch_name, tickers, date_start, date_end, result)

            with _ASYNC_LOCK:
                job["chapters"][ch_key] = {
                    "status": "done",
                    "ch_name": ch_name,
                    "result": result,
                    "fig_count": fig_count,
                    "json_ok": json_ok,
                    "db_ok": db_ok,
                }
                job["succeeded"] += 1
                job["completed"] += 1
        except Exception as exc:
            logger.error("Async pre-compute failed for %s: %s", ch_key, exc)
            with _ASYNC_LOCK:
                job["chapters"][ch_key] = {
                    "status": "error",
                    "ch_name": ch_name,
                    "result": {"text": f"Error: {exc}", "tables": [], "figures": []},
                }
                job["completed"] += 1

    with _ASYNC_LOCK:
        job["status"] = "done"
    logger.info("Async TTS batch %s finished — %d/%d succeeded",
                batch_id, job["succeeded"], job["total"] - job["skipped"])


@st.fragment(run_every=timedelta(seconds=3))
def _render_async_progress():
    """Auto-refreshing fragment that syncs background results into session_state."""
    batch_id = st.session_state.get("tts_async_batch_id")
    if not batch_id:
        return

    with _ASYNC_LOCK:
        job = _ASYNC_JOBS.get(batch_id)
    if not job:
        st.session_state.pop("tts_async_batch_id", None)
        st.session_state.pop("tts_async_synced", None)
        return

    synced: set = st.session_state.get("tts_async_synced", set())

    with _ASYNC_LOCK:
        for ch_key, ch_data in list(job["chapters"].items()):
            if ch_key in synced:
                continue
            if ch_data["status"] in ("done", "error"):
                st.session_state[f"tts_result_{ch_key}"] = ch_data["result"]
                synced.add(ch_key)

    st.session_state["tts_async_synced"] = synced

    total = job["total"]
    skipped = job.get("skipped", 0)
    effective_total = max(total - skipped, 1)
    effective_done = job["completed"] - skipped
    progress = min(effective_done / effective_total, 1.0)

    running_ch = None
    with _ASYNC_LOCK:
        for ch_key, ch_data in job["chapters"].items():
            if ch_data.get("status") == "running":
                running_ch = ch_key
                break

    if job["status"] == "done":
        succeeded = job["succeeded"]
        total = job["total"]
        skipped = job.get("skipped", 0)
        failed = total - skipped - succeeded
        parts: List[str] = [f"{succeeded} chapter(s) completed"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
        kind = "success" if not failed else "warning"
        st.session_state["_tts_batch_banner"] = {
            "kind": kind,
            "msg": f"TestTune batch done — {', '.join(parts)}",
        }
        st.session_state.pop("tts_async_batch_id", None)
        st.session_state.pop("tts_async_synced", None)
        st.session_state["tts_run_id"] = st.session_state.get("tts_run_id", 0) + 1
        with _ASYNC_LOCK:
            _ASYNC_JOBS.pop(batch_id, None)
        st.rerun(scope="app")
    else:
        running_name = _CH_NAME_MAP.get(running_ch, running_ch) if running_ch else "…"
        pct = int(progress * 100)
        st.markdown(
            spinner_html(f"Running {running_name} — {pct}%"),
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════
# Analysis tabs
# ════════════════════════════════════════════════════════════════════════
def _render_analysis_tabs(tickers: List[str], date_start: str, date_end: str):
    tab_labels = [t[0] for t in ANALYSIS_TABS]
    tabs = st.tabs(tab_labels)

    async_ch_status: Dict[str, str] = {}
    batch_id = st.session_state.get("tts_async_batch_id")
    if batch_id:
        with _ASYNC_LOCK:
            job = _ASYNC_JOBS.get(batch_id)
            if job:
                async_ch_status = {k: v.get("status", "") for k, v in job["chapters"].items()}

    for tab, (_, chapters) in zip(tabs, ANALYSIS_TABS):
        with tab:
            for ch_key, ch_name, ch_desc, needs_tickers in chapters:
                result = st.session_state.get(f"tts_result_{ch_key}")

                col_info, col_btn, _ = st.columns([4, 1, 4], vertical_alignment="center")
                with col_info:
                    a_status = async_ch_status.get(ch_key, "")
                    if result:
                        icon = ""
                    elif a_status == "running":
                        icon = "[running]"
                    elif a_status == "error":
                        icon = "[error]"
                    elif a_status == "skipped":
                        icon = "[skipped]"
                    else:
                        icon = ""
                    prefix = f"{icon} " if icon else ""
                    st.markdown(
                        f"<span style='font-size:0.92rem;'>{prefix}<b>{ch_name}</b> — {ch_desc}</span>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if needs_tickers and not tickers:
                        st.button(
                            "Re-run", key=f"tts_run_{ch_key}",
                            type="secondary", disabled=True,
                        )
                    else:
                        _render_chapter_rerun_button(
                            ch_key, ch_name, tickers,
                            date_start, date_end, needs_tickers,
                        )

                if needs_tickers and not tickers:
                    st.caption("Select at least one ticker to enable.")

                if result:
                    with st.expander("Results", expanded=False):
                        _display_result(result, ch_key)

                st.markdown("---")


@st.fragment
def _render_chapter_rerun_button(
    ch_key: str,
    ch_name: str,
    tickers: List[str],
    date_start: str,
    date_end: str,
    needs_tickers: bool,
):
    """Re-run a single chapter (overrides the precomputed cache)."""
    btn_key = f"tts_run_{ch_key}"
    if st.button("Re-run", key=btn_key, type="secondary"):
        rerun_spinner = st.empty()
        rerun_spinner.markdown(
            spinner_html("Running"),
            unsafe_allow_html=True,
        )
        try:
            result = _execute_chapter(
                ch_key, tickers, date_start, date_end, needs_tickers,
            )
            st.session_state[f"tts_result_{ch_key}"] = result
            run_id = _build_tts_run_id(tickers)
            fig_count = _save_tts_figures_to_minio(run_id, ch_key, ch_name, result)
            json_ok = _save_tts_text_to_minio(run_id, ch_key, ch_name, result)
            db_ok = _save_tts_to_database(ch_key, ch_name, tickers, date_start, date_end, result)
            rerun_spinner.empty()
            _report_persistence_status(ch_name, fig_count, json_ok, db_ok)
        except Exception as exc:
            logger.exception("TestTune %s failed", ch_key)
            rerun_spinner.empty()
            st.error(f"Error: {exc}")
        st.session_state["_tts_needs_rerun"] = True


# ════════════════════════════════════════════════════════════════════════
# Chapter execution engine
# ════════════════════════════════════════════════════════════════════════

_CHAPTER_RUNNERS: Dict[str, Any] = {}


def _get_runner(ch_key: str):
    """Lazy-import and cache the chapter module."""
    if ch_key in _CHAPTER_RUNNERS:
        return _CHAPTER_RUNNERS[ch_key]

    # Ensure testune_trade_sys root is on sys.path (may have been
    # evicted by Streamlit reruns or app.py path management).
    _tts_root = str(_TTS_ROOT)
    if _tts_root not in sys.path:
        sys.path.insert(0, _tts_root)

    module_map = {
        "ch01": "applied.ch01_introduction",
        "ch02": "applied.ch02_pre_optimization_issues",
        "ch03": "applied.ch03_optimization_issues",
        "ch04": "applied.ch04_post_optimization_issues",
        "ch05": "applied.ch05_estimating_future_performance_unbiased",
        "ch06": "applied.ch06_estimating_future_performance_trade_analysis",
        "ch07": "applied.ch07_permutation_tests",
    }

    import importlib
    mod_path = module_map[ch_key]
    mod = importlib.import_module(mod_path)
    _CHAPTER_RUNNERS[ch_key] = mod
    return mod


def _execute_chapter(
    ch_key: str,
    tickers: List[str],
    date_start: str,
    date_end: str,
    needs_tickers: bool,
) -> Dict[str, Any]:
    """Execute a chapter analysis and capture results.

    Returns a dict with keys:
        text    – captured stdout as string
        tables  – list of (title, DataFrame) pairs
        figures – list of (title, PNG-bytes) pairs
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import runpy

    mod = _get_runner(ch_key)

    # Patch sample_data globals if user supplied custom tickers / dates
    import sample_data as _sd
    orig_symbols = _sd.SYMBOLS
    orig_start = _sd.DEFAULT_START
    orig_end = _sd.DEFAULT_END

    _orig_show = plt.show
    _orig_close = plt.close
    _orig_savefig = plt.savefig
    plt.show = lambda *a, **kw: None
    plt.close = lambda *a, **kw: None
    plt.savefig = lambda *a, **kw: None

    try:
        if needs_tickers and tickers:
            _sd.SYMBOLS = list(tickers)
        _sd.DEFAULT_START = date_start
        _sd.DEFAULT_END = date_end

        _orig_close("all")
        pre_fignums = set(plt.get_fignums())

        buf = StringIO()
        with redirect_stdout(buf):
            if hasattr(mod, "main") and callable(mod.main):
                mod.main()
            else:
                script_path = str(Path(mod.__file__).resolve())
                runpy.run_path(script_path, run_name="__main__")

        new_figs = [
            (f"Figure {num}", plt.figure(num))
            for num in plt.get_fignums()
            if num not in pre_fignums
        ]

        result = {
            "text": buf.getvalue(),
            "tables": [],
            "figures": new_figs,
        }

        # Convert matplotlib figures to PNG bytes
        png_figs = []
        for title, fig in result.get("figures", []):
            fbuf = io.BytesIO()
            fig.savefig(fbuf, format="png", dpi=120, bbox_inches="tight")
            png_figs.append((title, fbuf.getvalue()))
            fbuf.close()
        result["figures"] = png_figs

        for num in plt.get_fignums():
            if num not in pre_fignums:
                _orig_close(num)

        return result
    finally:
        _sd.SYMBOLS = orig_symbols
        _sd.DEFAULT_START = orig_start
        _sd.DEFAULT_END = orig_end
        plt.show = _orig_show
        plt.close = _orig_close
        plt.savefig = _orig_savefig


# ════════════════════════════════════════════════════════════════════════
# Persistence – MinIO (figures) + PostgreSQL (results)
# ════════════════════════════════════════════════════════════════════════

def _build_tts_run_id(tickers: List[str]) -> str:
    """Build a unique run_id for MinIO storage."""
    short_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tts_{short_id}_{ts}"


def _save_tts_figures_to_minio(
    run_id: str,
    ch_key: str,
    ch_name: str,
    result: Dict[str, Any],
) -> int:
    """Save matplotlib figures from a chapter result to MinIO as PNGs."""
    if not MINIO_AVAILABLE:
        return 0
    figures = result.get("figures", [])
    if not figures:
        return 0
    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            return 0
        minio_svc.ensure_bucket_ready()
    except Exception as e:
        logger.error("MinIO not available for TTS figure save: %s", e)
        return 0

    saved = 0
    for idx, (title, fig_data) in enumerate(figures):
        try:
            image_bytes = fig_data if isinstance(fig_data, bytes) else fig_data
            tag = _ch_tag(ch_key)
            path = minio_svc.save_backtest_image(
                run_id=run_id,
                image_data=image_bytes,
                filename=f"{tag}_fig{idx}.png",
                strategy_name=f"tts_{tag}",
                chart_title=title or ch_name,
                chart_type="matplotlib",
                content_type="image/png",
            )
            if path:
                saved += 1
        except Exception as e:
            logger.error("Failed to save TTS figure %s/%d to MinIO: %s", ch_key, idx, e)

    if saved:
        logger.info("Saved %d figure(s) to MinIO for %s", saved, ch_key)
    return saved


def _save_tts_text_to_minio(
    run_id: str,
    ch_key: str,
    ch_name: str,
    result: Dict[str, Any],
) -> bool:
    """Save text output and table data to MinIO as JSON/HTML."""
    text = result.get("text", "")
    tables = result.get("tables", [])
    if not text and not tables:
        return False
    if not MINIO_AVAILABLE:
        return False
    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            return False
        minio_svc.ensure_bucket_ready()
    except Exception as e:
        logger.error("MinIO not available for TTS text save: %s", e)
        return False

    table_payload: list = []
    for title, df in tables:
        try:
            table_payload.append({
                "title": title,
                "data": df.to_dict(orient="records") if hasattr(df, "to_dict") else [],
            })
        except Exception:
            table_payload.append({"title": title, "data": []})

    payload = {
        "chapter": ch_key,
        "chapter_name": ch_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "text_output": text,
        "tables": table_payload,
    }

    try:
        json_str = json.dumps(payload, default=str, ensure_ascii=False, indent=2)
        safe_json = (
            json_str
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{ch_name}</title>"
            "<style>"
            "*{box-sizing:border-box}"
            "body{margin:0;padding:1.5rem 2rem;font-family:'Cascadia Code','Fira Code',"
            "'Consolas',monospace;font-size:13px;line-height:1.55;"
            "background:#1e1e1e;color:#d4d4d4}"
            "h1{font-size:1.1rem;color:#569cd6;margin:0 0 .3rem;font-weight:600}"
            ".meta{color:#6a9955;font-size:0.85em;margin-bottom:1rem}"
            "pre{white-space:pre-wrap;word-break:break-word;margin:0;"
            "padding:1rem;background:#252526;border:1px solid #3c3c3c;border-radius:6px;"
            "overflow-x:auto}"
            ".s{color:#ce9178}"
            ".n{color:#b5cea8}"
            ".k{color:#569cd6}"
            ".b{color:#569cd6}"
            ".null{color:#808080}"
            ".p{color:#808080}"
            "</style></head><body>"
            f"<h1>{ch_name}</h1>"
            f"<div class='meta'>Chapter {ch_key} &middot; {payload['run_id']} "
            f"&middot; {payload['timestamp']}</div>"
            f"<pre id='j'>{safe_json}</pre>"
            "<script>"
            "const e=document.getElementById('j'),"
            "t=e.innerHTML;"
            "e.innerHTML=t"
            ".replace(/&quot;([^&]*?)&quot;\\s*:/g,"
            "'<span class=\"k\">&quot;$1&quot;</span>:')"
            ".replace(/:&#32;*&quot;((?:[^&]|&(?!quot;))*)&quot;/g,"
            "':<span class=\"s\">&quot;$1&quot;</span>')"
            ".replace(/&quot;((?:[^&]|&(?!quot;))*)&quot;/g,"
            "'<span class=\"s\">&quot;$1&quot;</span>')"
            ".replace(/\\b(-?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?)\\b/g,"
            "'<span class=\"n\">$1</span>')"
            ".replace(/\\b(true|false)\\b/g,"
            "'<span class=\"b\">$1</span>')"
            ".replace(/\\bnull\\b/g,"
            "'<span class=\"null\">null</span>');"
            "</script></body></html>"
        )
        html_bytes = html.encode("utf-8")
        tag = _ch_tag(ch_key)
        path = minio_svc.save_backtest_image(
            run_id=run_id,
            image_data=html_bytes,
            filename=f"{tag}_results.html",
            strategy_name=f"tts_{tag}",
            chart_title=f"{ch_name} — text/tables",
            chart_type="json",
            content_type="text/html",
        )
        if path:
            logger.info("Saved text/table JSON to MinIO for %s: %s", ch_key, path)
            return True
        return False
    except Exception as e:
        logger.error("Failed to save TTS text JSON %s to MinIO: %s", ch_key, e)
        return False


def _save_tts_to_database(
    ch_key: str,
    ch_name: str,
    tickers: List[str],
    date_start: str,
    date_end: str,
    result: Dict[str, Any],
) -> bool:
    """Save a TestTune chapter result to the backtest_results table."""
    if not DB_AVAILABLE:
        return False
    try:
        db_service = _get_db_service()

        table_payload: list = []
        for title, df in result.get("tables", []):
            try:
                table_payload.append({
                    "title": title,
                    "data": df.to_dict(orient="records") if hasattr(df, "to_dict") else [],
                })
            except Exception:
                table_payload.append({"title": title, "data": []})

        backtest_data = {
            "strategy_id": f"tts_{ch_key}",
            "strategy_name": f"TTS: {ch_name}",
            "tickers": list(tickers) if tickers else [],
            "start_date": (
                datetime.strptime(date_start, "%Y-%m-%d") if date_start else None
            ),
            "end_date": (
                datetime.strptime(date_end, "%Y-%m-%d") if date_end else None
            ),
            "initial_capital": 0,
            "parameters": {
                "chapter": ch_key,
                "chapter_name": ch_name,
                "output_lines": len(result.get("text", "").splitlines()),
                "figure_count": len(result.get("figures", [])),
            },
            "metrics": {
                "text_output": result.get("text", ""),
                "tables": table_payload,
            },
            "total_trades": 0,
        }
        saved = db_service.save_backtest_result(backtest_data, market="US")
        if saved:
            logger.info("Saved TTS %s result to database", ch_key)
        return saved
    except Exception as e:
        logger.error("Failed to save TTS %s to database: %s", ch_key, e)
        return False


def _report_persistence_status(
    ch_name: str,
    fig_count: int,
    json_ok: bool,
    db_ok: bool,
):
    """Show a toast notification summarising what was persisted."""
    parts: List[str] = []
    if fig_count:
        parts.append(f"{fig_count} figure(s) -> MinIO")
    if json_ok:
        parts.append("text/tables -> MinIO")
    if db_ok:
        parts.append("results -> PostgreSQL")

    if parts:
        st.toast(f"{ch_name}: {', '.join(parts)}", icon=":material/check_circle:")
    else:
        warnings: List[str] = []
        if not fig_count:
            warnings.append("MinIO figures")
        if not json_ok:
            warnings.append("MinIO text/tables")
        if not db_ok:
            warnings.append("PostgreSQL")
        st.toast(
            f"{ch_name}: persistence skipped — {', '.join(warnings)}",
            icon=":material/warning:",
        )


# ════════════════════════════════════════════════════════════════════════
# Result display
# ════════════════════════════════════════════════════════════════════════
def _display_result(result: Dict[str, Any], ch_key: str):
    """Render captured analysis output."""
    if result.get("figures"):
        for title, fig_bytes in result["figures"]:
            st.image(fig_bytes, width="stretch")

    if result.get("tables"):
        for title, df in result["tables"]:
            st.markdown(f"**{title}**")
            st.dataframe(df, width="stretch")

    text = result.get("text", "")
    if text.strip():
        st.code(text, language="text")


# ════════════════════════════════════════════════════════════════════════
# Compact CSS
# ════════════════════════════════════════════════════════════════════════
def _inject_compact_css():
    st.markdown(
        """<style>
        [data-testid="stRadio"] { margin-top: -0.5rem; margin-bottom: -0.8rem; }
        [data-testid="stExpander"] { margin-top: -0.6rem; margin-bottom: -0.6rem; }
        [data-testid="stTextArea"] { margin-top: -0.2rem; }
        [data-testid="stFileUploader"] { margin-top: -0.6rem; }
        [data-testid="stSelectbox"] { margin-bottom: -0.8rem; }
        [data-testid="stCheckbox"] { margin-top: -0.5rem; margin-bottom: -0.5rem; }
        [data-testid="stHorizontalBlock"] + [data-testid="stElementContainer"],
        [data-testid="stHorizontalBlock"] + div {
            margin-top: -1.5rem !important;
        }
        [data-testid="stAlert"] { margin-top: -0.5rem !important;
                                   margin-bottom: -0.5rem !important; }
        </style>""",
        unsafe_allow_html=True,
    )
