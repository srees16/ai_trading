"""
Financial ML Page Module for Centurion Capital LLC.

Interactive UI for *Advances in Financial Machine Learning* (AFML) chapter
analyses.  Users select stock tickers (or use defaults), then explore
results across tabbed categories that map to the ``applied/`` scripts.
"""

import io
import json
import sys
import base64
import logging
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
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

# ── Path setup: allow imports from financial_ML/ ────────────────────────
_FINANCIAL_ML_ROOT = Path(__file__).resolve().parent.parent.parent / "financial_ML"
if str(_FINANCIAL_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_FINANCIAL_ML_ROOT))

# Default tickers (same as sample_data.SYMBOLS)
FML_DEFAULT_TICKERS = ["MSFT", "GOOG", "NVDA", "AMD"]

# ── Tab / chapter registry ──────────────────────────────────────────────
# Grouped into logical analysis categories.  Each entry:
#   (tab_label, [(chapter_key, display_name, description, needs_tickers), ...])

ANALYSIS_TABS: List[tuple] = [
    ("Data Structures", [
        ("ch02", "Financial Data Structures",
         "PCA weights, roll-gap adjustment, CUSUM filter", True),
        ("ch03", "Labeling (Triple-Barrier)",
         "Daily vol, triple-barrier labels, CUSUM events", True),
        ("ch04", "Sample Weights",
         "Concurrent events, uniqueness, sequential bootstrap", False),
    ]),
    ("Features", [
        ("ch05", "Fractional Differentiation",
         "FFD stationarity transform — find min d for ADF", True),
        ("ch08", "Feature Importance",
         "MDI, MDA, SFI importance with PCA orthogonalisation", False),
        ("ch17", "Structural Breaks",
         "SADF, Chu-Stinchcombe-White CUSUM, SMT tests", True),
        ("ch18", "Entropy Features",
         "Plug-in, Lempel-Ziv, Kontoyiannis entropy estimators", True),
        ("ch19", "Microstructural Features",
         "Corwin-Schultz spread, tick rule, Kyle/Amihud lambda, VPIN", True),
    ]),
    ("Modeling", [
        ("ch06", "Ensemble Methods",
         "Bagging accuracy, three RF setups, cross-validated scores", False),
        ("ch07", "Cross-Validation",
         "Purged K-Fold, embargo, time-aware CV", False),
        ("ch09", "Hyper-Parameter Tuning",
         "Grid & random search with purged CV", False),
        ("ch10", "Bet Sizing",
         "Signal → position sizing, discretisation, limit prices", False),
    ]),
    ("Backtesting", [
        ("ch11", "Dangers of Backtesting",
         "Selection bias, deflated SR, probability of backtest overfitting", False),
        ("ch13", "Synthetic Backtesting",
         "Ornstein-Uhlenbeck optimal trading rules", False),
        ("ch14", "Backtest Statistics",
         "Sharpe, PSR, DSR, drawdowns, HHI concentration", False),
        ("ch15", "Strategy Risk",
         "Implied precision, betting frequency, failure probability", False),
    ]),
    ("Portfolio", [
        ("ch16", "ML Asset Allocation",
         "Hierarchical Risk Parity (HRP) vs CLA vs IVP — requires  >2 tickers", True),
    ]),
    ("Computation", [
        ("ch20", "Multiprocessing",
         "Vectorisation benchmarks, barrier touch, parallel partitioning", False),
        ("ch21", "Brute Force & Quantum",
         "Combinatorial portfolio optimisation, dynamic vs static SR", False),
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
    "ch02": "fds",   # Financial Data Structures
    "ch03": "lbl",   # Labeling (Triple-Barrier)
    "ch04": "sw",    # Sample Weights
    "ch05": "fd",    # Fractional Differentiation
    "ch06": "em",    # Ensemble Methods
    "ch07": "cv",    # Cross-Validation
    "ch08": "fi",    # Feature Importance
    "ch09": "hpt",   # Hyper-Parameter Tuning
    "ch10": "bs",    # Bet Sizing
    "ch11": "dob",   # Dangers of Backtesting
    "ch13": "sb",    # Synthetic Backtesting
    "ch14": "bts",   # Backtest Statistics
    "ch15": "sr",    # Strategy Risk
    "ch16": "mla",   # ML Asset Allocation
    "ch17": "stb",   # Structural Breaks
    "ch18": "ef",    # Entropy Features
    "ch19": "mf",    # Microstructural Features
    "ch20": "mp",    # Multiprocessing
    "ch21": "bfq",   # Brute Force & Quantum
}


def _ch_tag(ch_key: str) -> str:
    """Return `ch05_fd` style tag for MinIO paths."""
    acr = _CH_ACRONYM.get(ch_key)
    return f"{ch_key}_{acr}" if acr else ch_key

# ── Async pre-computation state (thread-safe, module-level) ─────────
_ASYNC_LOCK = threading.Lock()
_ASYNC_JOBS: Dict[str, Dict[str, Any]] = {}


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════
def render_finance_ml_page():
    """Render the Financial ML module page."""
    _user = st.session_state.get("username", "unknown")
    logger.info("[user=%s] Viewing Financial ML page", _user)

    render_header()

    # ── Title ───────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center; padding:0.4rem 0 0.2rem;">
            <span style="font-size:1.5rem; font-weight:700;
                         color:{Colors.TEXT_PRIMARY};">
                Advanced Financial Machine Learning Techniques
            </span><br>
            <span style="font-size:0.85rem; color:{Colors.TEXT_MUTED};">
                Based on the book by - Marcos López de Prado
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── History navigation ──────────────────────────────────────
    if st.button("View Saved Results", key="fml_goto_history"):
        st.session_state["current_page"] = "fml_history"
        st.rerun()

    # ── Ticker input (left) + settings (right) ─────────────────
    _inject_compact_css()
    col1, col2 = st.columns([1, 1])

    with col1:
        tickers = _render_ticker_selection()

    with col2:
        date_start, date_end = _render_date_settings()

    # Store in session
    st.session_state["fml_tickers"] = tickers
    st.session_state["fml_date_start"] = date_start
    st.session_state["fml_date_end"] = date_end

    # ── Run All button ──────────────────────────────────────────
    _render_run_all_button(tickers, date_start, date_end)

    # ── Async progress (auto-refreshing fragment) ───────────────
    if st.session_state.get("fml_async_batch_id"):
        _render_async_progress()

    st.markdown("---")

    # ── Analysis tabs ───────────────────────────────────────────
    _render_analysis_tabs(tickers, date_start, date_end)

    render_footer()

    # Deferred rerun: fragments signal completion via a sentinel
    # so the rerun happens outside fragment scope (avoids orphan-tick warnings).
    if st.session_state.pop("_fml_needs_rerun", False):
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# Ticker selection (mirrors Ind Stocks pattern)
# ════════════════════════════════════════════════════════════════════════
def _render_ticker_selection() -> List[str]:
    st.markdown("**Select Stocks**")
    ticker_mode = st.radio(
        "Input method:",
        ["Default Tickers", "Manual Entry", "Upload CSV"],
        help="Choose how to specify the stock tickers for analysis",
        horizontal=True,
        key="fml_ticker_mode",
    )
    if ticker_mode == "Default Tickers":
        return _handle_default_tickers()
    elif ticker_mode == "Manual Entry":
        return _handle_manual_entry()
    else:
        return _handle_csv_upload()


def _handle_default_tickers() -> List[str]:
    with st.expander("View default tickers"):
        st.write(", ".join(FML_DEFAULT_TICKERS))
    return list(FML_DEFAULT_TICKERS)


def _handle_manual_entry() -> List[str]:
    ticker_input = st.text_area(
        "Enter tickers (comma-separated):",
        value="MSFT, GOOG, NVDA, AMD",
        height=80,
        help="Enter US stock symbols separated by commas.",
        key="fml_manual_tickers",
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    return tickers


def _handle_csv_upload() -> List[str]:
    with st.expander("View CSV format example"):
        sample = "Ticker\nMSFT\nGOOG\nNVDA\nAMD\n"
        st.code(sample, language="csv")
        st.download_button(
            label="Download Sample CSV",
            data=sample,
            file_name="sample_fml_tickers.csv",
            mime="text/csv",
            key="fml_csv_download",
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV with ticker symbols",
        key="fml_csv_upload",
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
            key="fml_start_date",
        )
    with col_b:
        date_end = st.date_input(
            "End date",
            value=_dt.date(2024, 12, 31),
            key="fml_end_date",
        )

    st.caption(
        "Date range applies only to chapters that fetch live market data. "
        "Chapters using generated / synthetic data are unaffected."
    )
    return str(date_start), str(date_end)


# ════════════════════════════════════════════════════════════════════════
# Run All – pre-compute every chapter (mirrors backtesting strategy tab)
# ════════════════════════════════════════════════════════════════════════


def _render_run_all_button(tickers: List[str], date_start: str, date_end: str):
    async_running = bool(st.session_state.get("fml_async_batch_id"))
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        run_all = st.button(
            "Run Analyses",
            type="primary",
            use_container_width=True,
            disabled=not tickers or async_running,
            help="Pre-compute all chapter analyses in the background" if not async_running
                 else "Computation in progress",
            key="fml_run_all",
        )

    if run_all and tickers and not async_running:
        _start_async_precompute(tickers, date_start, date_end)
        st.rerun()


_PRECOMPUTE_SKIP = {"ch13", "ch16"}  # ch13: Synthetic Backtesting (slow), ch16: HRP (needs 2+ tickers)


# ────────────────────────────────────────────────────────────────────────
# Async pre-computation (non-blocking)
# ────────────────────────────────────────────────────────────────────────

def _start_async_precompute(
    tickers: List[str], date_start: str, date_end: str,
):
    """Launch pre-computation in a background thread.

    Chapters execute sequentially in the worker thread (avoiding
    global-state conflicts with matplotlib / sample_data).  Results are
    written to the module-level ``_ASYNC_JOBS`` dict which the Streamlit
    fragment ``_render_async_progress`` polls every few seconds.
    """
    batch_id = uuid.uuid4().hex[:12]

    all_chapters = [
        (ch_key, ch_name, ch_desc, needs_tickers)
        for _, chapters in ANALYSIS_TABS
        for ch_key, ch_name, ch_desc, needs_tickers in chapters
    ]

    # Chapters already cached in session_state — skip them
    cached_keys: set = {
        ch_key for ch_key, *_ in all_chapters
        if st.session_state.get(f"fml_result_{ch_key}")
    }

    job: Dict[str, Any] = {
        "status": "running",
        "total": len(all_chapters),
        "completed": 0,
        "succeeded": 0,
        "skipped": 0,
        "chapters": {},
        "tickers": list(tickers),
        "run_id": _build_fml_run_id(tickers),
    }

    with _ASYNC_LOCK:
        _ASYNC_JOBS[batch_id] = job

    st.session_state["fml_async_batch_id"] = batch_id
    st.session_state["fml_async_synced"] = set()

    thread = threading.Thread(
        target=_async_worker,
        args=(batch_id, all_chapters, tickers, date_start, date_end, cached_keys),
        daemon=True,
    )
    thread.start()
    logger.info("Async FML pre-computation started — batch %s", batch_id)


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
    user = "async"
    cached_keys = cached_keys or set()

    for idx, (ch_key, ch_name, _, needs_tickers) in enumerate(all_chapters):
        # Mark chapter as running
        with _ASYNC_LOCK:
            job["chapters"][ch_key] = {"status": "running", "ch_name": ch_name}

        logger.info("[%s] Async computing %d/%d: %s", user, idx + 1, job["total"], ch_key)

        # --- skip already-cached results ---
        if ch_key in cached_keys:
            with _ASYNC_LOCK:
                job["chapters"][ch_key] = {"status": "skipped", "ch_name": ch_name, "reason": "cached"}
                job["skipped"] += 1
                job["completed"] += 1
            continue

        # --- skip logic ---
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

        # --- execute ---
        try:
            result = _execute_chapter(ch_key, tickers, date_start, date_end, needs_tickers)
            fig_count = _save_fml_figures_to_minio(run_id, ch_key, ch_name, result)
            json_ok = _save_fml_text_to_minio(run_id, ch_key, ch_name, result)
            db_ok = _save_fml_to_database(ch_key, ch_name, tickers, date_start, date_end, result)

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
    logger.info("Async FML batch %s finished — %d/%d succeeded",
                batch_id, job["succeeded"], job["total"] - job["skipped"])


@st.fragment(run_every=timedelta(seconds=3))
def _render_async_progress():
    """Auto-refreshing fragment that syncs background results into session_state.

    Runs every 3 s while a batch is active.  When new chapter results are
    detected the fragment copies them to ``session_state`` and triggers a
    full page rerun so the analysis tabs re-render with the new data.
    """
    batch_id = st.session_state.get("fml_async_batch_id")
    if not batch_id:
        return

    with _ASYNC_LOCK:
        job = _ASYNC_JOBS.get(batch_id)
    if not job:
        # Job vanished (e.g. server restart) — clean up
        st.session_state.pop("fml_async_batch_id", None)
        st.session_state.pop("fml_async_synced", None)
        return

    # ── Sync completed results to session_state ──────────────
    synced: set = st.session_state.get("fml_async_synced", set())
    new_completions: List[Dict[str, Any]] = []

    with _ASYNC_LOCK:
        for ch_key, ch_data in list(job["chapters"].items()):
            if ch_key in synced:
                continue
            if ch_data["status"] in ("done", "error"):
                st.session_state[f"fml_result_{ch_key}"] = ch_data["result"]
                synced.add(ch_key)
                new_completions.append(ch_data)

    st.session_state["fml_async_synced"] = synced

    # ── Progress bar ─────────────────────────────────────────
    total = job["total"]
    skipped = job.get("skipped", 0)
    effective_total = max(total - skipped, 1)
    effective_done = job["completed"] - skipped
    progress = min(effective_done / effective_total, 1.0)

    # Identify the currently-running chapter
    running_ch = None
    with _ASYNC_LOCK:
        for ch_key, ch_data in job["chapters"].items():
            if ch_data.get("status") == "running":
                running_ch = ch_key
                break

    if job["status"] == "done":
        # Show toasts for any remaining un-toasted completions
        for cd in new_completions:
            if cd["status"] == "done":
                _report_persistence_status(
                    cd["ch_name"],
                    cd.get("fig_count", 0),
                    cd.get("json_ok", False),
                    cd.get("db_ok", False),
                )
        # Final cleanup — remove batch so the guard at the call-site
        # (``if st.session_state.get("fml_async_batch_id")``) stops
        # rendering this fragment on the next full-app rerun.
        st.session_state.pop("fml_async_batch_id", None)
        st.session_state.pop("fml_async_synced", None)
        st.session_state["fml_run_id"] = st.session_state.get("fml_run_id", 0) + 1
        with _ASYNC_LOCK:
            _ASYNC_JOBS.pop(batch_id, None)
        # Signal the main page to do the full rerun outside the fragment
        st.session_state["_fml_needs_rerun"] = True
    else:
        running_name = _CH_NAME_MAP.get(running_ch, running_ch) if running_ch else "…"
        pct = int(progress * 100)
        st.markdown(
            spinner_html(f"Running {running_name} — {pct}%"),
            unsafe_allow_html=True,
        )
        # Show toasts for chapters that just completed (no full rerun needed —
        # the fragment auto-refreshes every 3 s and tabs update when job finishes)
        for cd in new_completions:
            if cd["status"] == "done":
                _report_persistence_status(
                    cd["ch_name"],
                    cd.get("fig_count", 0),
                    cd.get("json_ok", False),
                    cd.get("db_ok", False),
                )


# ════════════════════════════════════════════════════════════════════════
# Analysis tabs
# ════════════════════════════════════════════════════════════════════════
def _render_analysis_tabs(tickers: List[str], date_start: str, date_end: str):
    tab_labels = [t[0] for t in ANALYSIS_TABS]
    tabs = st.tabs(tab_labels)

    # Peek at async job status for richer per-chapter indicators
    async_ch_status: Dict[str, str] = {}
    batch_id = st.session_state.get("fml_async_batch_id")
    if batch_id:
        with _ASYNC_LOCK:
            job = _ASYNC_JOBS.get(batch_id)
            if job:
                async_ch_status = {k: v.get("status", "") for k, v in job["chapters"].items()}

    for tab, (_, chapters) in zip(tabs, ANALYSIS_TABS):
        with tab:
            for ch_key, ch_name, ch_desc, needs_tickers in chapters:
                result = st.session_state.get(f"fml_result_{ch_key}")

                # Chapter heading + re-run button
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
                            "Re-run", key=f"fml_run_{ch_key}",
                            type="secondary", disabled=True,
                        )
                    else:
                        _render_chapter_rerun_button(
                            ch_key, ch_name, tickers,
                            date_start, date_end, needs_tickers,
                        )

                if needs_tickers and not tickers:
                    st.caption("⚠️ Select at least one ticker to enable.")

                # Show cached results
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
    btn_key = f"fml_run_{ch_key}"
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
            st.session_state[f"fml_result_{ch_key}"] = result
            run_id = _build_fml_run_id(tickers)
            fig_count = _save_fml_figures_to_minio(run_id, ch_key, ch_name, result)
            json_ok = _save_fml_text_to_minio(run_id, ch_key, ch_name, result)
            db_ok = _save_fml_to_database(ch_key, ch_name, tickers, date_start, date_end, result)
            rerun_spinner.empty()
            _report_persistence_status(ch_name, fig_count, json_ok, db_ok)
        except Exception as exc:
            logger.exception("Financial ML %s failed", ch_key)
            rerun_spinner.empty()
            st.error(f"Error: {exc}")
        st.session_state["_fml_needs_rerun"] = True


# ════════════════════════════════════════════════════════════════════════
# Chapter execution engine
# ════════════════════════════════════════════════════════════════════════

import re as _re

_DEMO_LINE_RE = _re.compile(r"(?mi)^.*\bdemos?\b.*\n?")
_DEMO_WORD_RE = _re.compile(r"\bdemos?\b", _re.IGNORECASE)


def _sanitize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Strip 'demo' references from text output and figure titles.

    Applied once right after chapter execution so every downstream
    consumer (session state, MinIO, PostgreSQL, UI) sees clean data.
    """
    # --- text output: drop entire lines containing 'demo' ----
    text = result.get("text", "")
    if text:
        text = _DEMO_LINE_RE.sub("", text)
        # collapse runs of blank lines left behind
        text = _re.sub(r"\n{3,}", "\n\n", text)
        result["text"] = text

    # --- figure titles embedded in matplotlib axes -----------
    for _idx, (title, fig) in enumerate(result.get("figures", [])):
        for ax in fig.get_axes():
            cur = ax.get_title()
            if cur and _DEMO_WORD_RE.search(cur):
                ax.set_title(_DEMO_WORD_RE.sub("", cur).strip(" |—-_"))

    # --- table titles ----------------------------------------
    cleaned_tables = []
    for title, df in result.get("tables", []):
        cleaned_tables.append((_DEMO_WORD_RE.sub("", title).strip(), df))
    result["tables"] = cleaned_tables

    return result


_CHAPTER_RUNNERS: Dict[str, Any] = {}


def _get_runner(ch_key: str):
    """Lazy-import and cache the chapter module."""
    if ch_key in _CHAPTER_RUNNERS:
        return _CHAPTER_RUNNERS[ch_key]

    module_map = {
        "ch02": "applied.ch02_financial_data_structures",
        "ch03": "applied.ch03_labeling",
        "ch04": "applied.ch04_sample_weights",
        "ch05": "applied.ch05_fractionally_differentiated_features",
        "ch06": "applied.ch06_ensemble_methods",
        "ch07": "applied.ch07_cross_validation_in_finance",
        "ch08": "applied.ch08_feature_importance",
        "ch09": "applied.ch09_hyper_parameter_tuning_with_cross_validation",
        "ch10": "applied.ch10_bet_sizing",
        "ch11": "applied.ch11_the_dangers_of_backtesting",
        "ch13": "applied.ch13_backtesting_on_synthetic_data",
        "ch14": "applied.ch14_backtest_statistics",
        "ch15": "applied.ch15_understanding_strategy_risk",
        "ch16": "applied.ch16_machine_learning_asset_allocation",
        "ch17": "applied.ch17_structural_breaks",
        "ch18": "applied.ch18_entropy_features",
        "ch19": "applied.ch19_microstructural_features",
        "ch20": "applied.ch20_multiprocessing_and_vectorization",
        "ch21": "applied.ch21_brute_force_and_quantum_computers",
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
        text   – captured stdout as string
        tables – list of (title, DataFrame) pairs
        figures – list of (title, matplotlib.Figure) pairs
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

    # Monkey-patch plt so scripts don't discard figures we want to capture
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

        # Close all existing figures so we can detect new ones
        _orig_close("all")
        pre_fignums = set(plt.get_fignums())

        # Capture printed output
        buf = StringIO()
        with redirect_stdout(buf):
            if hasattr(mod, "main") and callable(mod.main):
                mod.main()
            else:
                # Chapter runs in its `if __name__ == "__main__"` block
                script_path = str(Path(mod.__file__).resolve())
                runpy.run_path(script_path, run_name="__main__")

        # Collect any new matplotlib figures
        new_figs = [
            (f"Figure {num}", plt.figure(num))
            for num in plt.get_fignums()
            if num not in pre_fignums
        ]

        return _sanitize_result({
            "text": buf.getvalue(),
            "tables": [],
            "figures": new_figs,
        })
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

def _build_fml_run_id(tickers: List[str]) -> str:
    """Build a unique run_id for MinIO storage."""
    short_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"fml_{short_id}_{ts}"


def _save_fml_figures_to_minio(
    run_id: str,
    ch_key: str,
    ch_name: str,
    result: Dict[str, Any],
) -> int:
    """Save matplotlib figures from a chapter result to MinIO as PNGs."""
    if not MINIO_AVAILABLE:
        logger.warning("MinIO disabled — skipping figure save for %s", ch_key)
        return 0
    figures = result.get("figures", [])
    if not figures:
        return 0
    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            logger.warning("MinIO unreachable — skipping figure save for %s", ch_key)
            return 0
        minio_svc.ensure_bucket_ready()
    except Exception as e:
        logger.error("MinIO not available for FML figure save: %s", e)
        return 0

    saved = 0
    for idx, (title, fig) in enumerate(figures):
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            image_bytes = buf.getvalue()
            buf.close()

            tag = _ch_tag(ch_key)
            path = minio_svc.save_backtest_image(
                run_id=run_id,
                image_data=image_bytes,
                filename=f"{tag}_fig{idx}.png",
                strategy_name=f"fml_{tag}",
                chart_title=title or ch_name,
                chart_type="matplotlib",
                content_type="image/png",
            )
            if path:
                saved += 1
        except Exception as e:
            logger.error("Failed to save FML figure %s/%d to MinIO: %s", ch_key, idx, e)

    if saved:
        logger.info("Saved %d figure(s) to MinIO for %s", saved, ch_key)
    return saved


def _save_fml_text_to_minio(
    run_id: str,
    ch_key: str,
    ch_name: str,
    result: Dict[str, Any],
) -> bool:
    """Save text output and table data from a chapter result to MinIO as JSON.

    The JSON object stored has the structure::

        {
            "chapter": "ch08",
            "chapter_name": "Feature Importance",
            "run_id": "fml_abc12345_20260312_143000",
            "timestamp": "2026-03-12T14:30:00",
            "text_output": "<captured stdout>",
            "tables": [{"title": "...", "data": [{...}, ...]}, ...]
        }

    Returns True if the JSON was persisted, False otherwise.
    """
    text = result.get("text", "")
    tables = result.get("tables", [])
    if not text and not tables:
        return False
    if not MINIO_AVAILABLE:
        logger.warning("MinIO disabled — skipping text/table save for %s", ch_key)
        return False
    try:
        minio_svc = _get_minio()
        if not minio_svc.is_available:
            logger.warning("MinIO unreachable — skipping text/table save for %s", ch_key)
            return False
        minio_svc.ensure_bucket_ready()
    except Exception as e:
        logger.error("MinIO not available for FML text save: %s", e)
        return False

    # Build serialisable table list
    table_payload: List[Dict[str, Any]] = []
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
        # Escape HTML-special chars so the JSON renders safely inside <pre>
        safe_json = (
            json_str
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        # Wrap in styled HTML with CSS-based JSON syntax colouring
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
            ".s{color:#ce9178}"       # strings
            ".n{color:#b5cea8}"       # numbers
            ".k{color:#569cd6}"       # keys
            ".b{color:#569cd6}"       # booleans
            ".null{color:#808080}"    # null
            ".p{color:#808080}"       # punctuation
            "</style></head><body>"
            f"<h1>{ch_name}</h1>"
            f"<div class='meta'>Chapter {ch_key} &middot; {payload['run_id']} "
            f"&middot; {payload['timestamp']}</div>"
            f"<pre id='j'>{safe_json}</pre>"
            "<script>"
            # Lightweight regex-based JSON syntax highlighter
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
            strategy_name=f"fml_{tag}",
            chart_title=f"{ch_name} — text/tables",
            chart_type="json",
            content_type="text/html",
        )
        if path:
            logger.info("Saved text/table JSON to MinIO for %s: %s", ch_key, path)
            return True
        logger.warning("MinIO returned empty path for %s text save", ch_key)
        return False
    except Exception as e:
        logger.error("Failed to save FML text JSON %s to MinIO: %s", ch_key, e)
        return False


def _save_fml_to_database(
    ch_key: str,
    ch_name: str,
    tickers: List[str],
    date_start: str,
    date_end: str,
    result: Dict[str, Any],
) -> bool:
    """Save a Financial ML chapter result to the backtest_results table.

    The full text output and serialised tables are stored in the
    ``metrics`` JSONB column so they can be queried later via::

        SELECT id, strategy_name, created_at,
               metrics->'text_output'   AS text_output,
               metrics->'tables'        AS tables
        FROM   backtest_results
        WHERE  strategy_id LIKE 'fml_%'
        ORDER  BY created_at DESC;
    """
    if not DB_AVAILABLE:
        logger.warning("Database not configured — skipping save for %s", ch_key)
        return False
    try:
        db_service = _get_db_service()

        # Serialise table data for JSONB storage
        table_payload: List[Dict[str, Any]] = []
        for title, df in result.get("tables", []):
            try:
                table_payload.append({
                    "title": title,
                    "data": df.to_dict(orient="records") if hasattr(df, "to_dict") else [],
                })
            except Exception:
                table_payload.append({"title": title, "data": []})

        backtest_data = {
            "strategy_id": f"fml_{ch_key}",
            "strategy_name": f"FML: {ch_name}",
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
            logger.info("Saved FML %s result to database (with text & tables)", ch_key)
        else:
            logger.warning("Database service returned False for FML %s save", ch_key)
        return saved
    except Exception as e:
        logger.error("Failed to save FML %s to database: %s", ch_key, e)
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
        parts.append(f"{fig_count} figure(s) → MinIO")
    if json_ok:
        parts.append("text/tables → MinIO")
    if db_ok:
        parts.append("results → PostgreSQL")

    if parts:
        st.toast(f"{ch_name}: {', '.join(parts)}", icon="✅")
    else:
        warnings: List[str] = []
        if not fig_count:
            warnings.append("MinIO figures")
        if not json_ok:
            warnings.append("MinIO text/tables")
        if not db_ok:
            warnings.append("PostgreSQL")
        st.toast(
            f"⚠️ {ch_name}: persistence skipped — {', '.join(warnings)}",
            icon="⚠️",
        )


# ════════════════════════════════════════════════════════════════════════
# Result display
# ════════════════════════════════════════════════════════════════════════
def _display_result(result: Dict[str, Any], ch_key: str):
    """Render captured analysis output."""
    # Figures
    if result.get("figures"):
        for title, fig in result["figures"]:
            st.pyplot(fig, width="stretch")

    # Tables
    if result.get("tables"):
        for title, df in result["tables"]:
            st.markdown(f"**{title}**")
            st.dataframe(df, width="stretch")

    # Text output (already sanitised by _sanitize_result)
    text = result.get("text", "")
    if text.strip():
        st.code(text, language="text")


# ════════════════════════════════════════════════════════════════════════
# Compact CSS (mirrors Ind Stocks control panel styling)
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
