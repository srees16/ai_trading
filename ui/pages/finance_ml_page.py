"""
Financial ML Page Module for Centurion Capital LLC.

Interactive UI for *Advances in Financial Machine Learning* (AFML) chapter
analyses.  Users select stock tickers (or use defaults), then explore
results across tabbed categories that map to the ``applied/`` scripts.
"""

import io
import sys
import base64
import logging
import uuid
from datetime import datetime
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
         "Hierarchical Risk Parity (HRP) vs CLA vs IVP — requires 2+ tickers", True),
    ]),
    ("Computation", [
        ("ch20", "Multiprocessing",
         "Vectorisation benchmarks, barrier touch, parallel partitioning", False),
        ("ch21", "Brute Force & Quantum",
         "Combinatorial portfolio optimisation, dynamic vs static SR", False),
    ]),
]


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
                Financial Machine Learning Techniques
            </span><br>
            <span style="font-size:0.85rem; color:{Colors.TEXT_MUTED};">
                Based on the book by - Marcos López de Prado
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

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

    st.markdown("---")

    # ── Analysis tabs ───────────────────────────────────────────
    _render_analysis_tabs(tickers, date_start, date_end)

    render_footer()


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
    st.markdown("**Analysis Settings**")
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
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        run_all = st.button(
            "Run Analyses",
            type="primary",
            use_container_width=True,
            disabled=not tickers,
            help="Pre-compute all chapter analyses at once",
            key="fml_run_all",
        )

    if run_all and tickers:
        _precompute_all_chapters(tickers, date_start, date_end)


def _precompute_all_chapters(
    tickers: List[str], date_start: str, date_end: str,
):
    """Execute every chapter sequentially and cache results (like backtest precompute)."""
    _user = st.session_state.get("username", "unknown")

    # Flatten all chapters across all tabs
    all_chapters = [
        (ch_key, ch_name, ch_desc, needs_tickers)
        for _, chapters in ANALYSIS_TABS
        for ch_key, ch_name, ch_desc, needs_tickers in chapters
    ]
    total = len(all_chapters)
    succeeded = 0

    spinner_slot = st.empty()

    # Single run_id shared by all chapters in this batch
    run_id = _build_fml_run_id(tickers)

    for idx, (ch_key, ch_name, _, needs_tickers) in enumerate(all_chapters):
        logger.info(
            "[user=%s] Pre-computing %d/%d: %s", _user, idx + 1, total, ch_key,
        )
        spinner_slot.markdown(
            spinner_html(f"Running {ch_name} ({idx + 1}/{total})"),
            unsafe_allow_html=True,
        )

        if needs_tickers and not tickers:
            # Skip chapters that need tickers when none supplied
            continue

        try:
            result = _execute_chapter(ch_key, tickers, date_start, date_end, needs_tickers)
            st.session_state[f"fml_result_{ch_key}"] = result
            succeeded += 1
            # Persist to MinIO + DB
            _save_fml_figures_to_minio(run_id, ch_key, ch_name, result)
            _save_fml_to_database(ch_key, ch_name, tickers, date_start, date_end, result)
        except Exception as exc:
            logger.error("Pre-compute failed for %s: %s", ch_key, exc)
            st.session_state[f"fml_result_{ch_key}"] = {
                "text": f"Error: {exc}", "tables": [], "figures": [],
            }

    spinner_slot.markdown(
        spinner_html(f"All analyses computed ({succeeded}/{total} succeeded) ✓"),
        unsafe_allow_html=True,
    )
    import time as _time
    _time.sleep(0.8)
    spinner_slot.empty()

    # Bump a run counter so the tabs know results are fresh
    st.session_state["fml_run_id"] = st.session_state.get("fml_run_id", 0) + 1
    logger.info("[user=%s] Pre-computation complete: %d/%d succeeded", _user, succeeded, total)


# ════════════════════════════════════════════════════════════════════════
# Analysis tabs
# ════════════════════════════════════════════════════════════════════════
def _render_analysis_tabs(tickers: List[str], date_start: str, date_end: str):
    tab_labels = [t[0] for t in ANALYSIS_TABS]
    tabs = st.tabs(tab_labels)

    for tab, (_, chapters) in zip(tabs, ANALYSIS_TABS):
        with tab:
            for ch_key, ch_name, ch_desc, needs_tickers in chapters:
                result = st.session_state.get(f"fml_result_{ch_key}")

                # Chapter heading + re-run button
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    status = "✅" if result else "⬜"
                    st.markdown(f"{status} **{ch_name}** — {ch_desc}")
                with col_btn:
                    if needs_tickers and not tickers:
                        st.button(
                            "↻ Re-run", key=f"fml_run_{ch_key}",
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
                    with st.expander(f"📊 {ch_name} — Results", expanded=True):
                        _display_result(result, ch_key)

                st.markdown("---")


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
    if st.button("↻ Re-run", key=btn_key, type="secondary"):
        with st.spinner(f"Running {ch_name}…"):
            try:
                result = _execute_chapter(
                    ch_key, tickers, date_start, date_end, needs_tickers,
                )
                st.session_state[f"fml_result_{ch_key}"] = result
                run_id = _build_fml_run_id(tickers)
                _save_fml_figures_to_minio(run_id, ch_key, ch_name, result)
                _save_fml_to_database(ch_key, ch_name, tickers, date_start, date_end, result)
            except Exception as exc:
                logger.exception("Financial ML %s failed", ch_key)
                st.error(f"Error: {exc}")


# ════════════════════════════════════════════════════════════════════════
# Chapter execution engine
# ════════════════════════════════════════════════════════════════════════
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
                # Chapter has demo in `if __name__ == "__main__"` block
                script_path = str(Path(mod.__file__).resolve())
                runpy.run_path(script_path, run_name="__main__")

        # Collect any new matplotlib figures
        new_figs = [
            (f"Figure {num}", plt.figure(num))
            for num in plt.get_fignums()
            if num not in pre_fignums
        ]

        return {
            "text": buf.getvalue(),
            "tables": [],
            "figures": new_figs,
        }
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
        logger.error("MinIO not available for FML save: %s", e)
        return 0

    saved = 0
    for idx, (title, fig) in enumerate(figures):
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            image_bytes = buf.getvalue()
            buf.close()

            path = minio_svc.save_backtest_image(
                run_id=run_id,
                image_data=image_bytes,
                filename=f"{ch_key}_fig{idx}.png",
                strategy_name=f"fml_{ch_key}",
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


def _save_fml_to_database(
    ch_key: str,
    ch_name: str,
    tickers: List[str],
    date_start: str,
    date_end: str,
    result: Dict[str, Any],
) -> bool:
    """Save a Financial ML chapter result to the backtest_results table."""
    if not DB_AVAILABLE:
        return False
    try:
        db_service = _get_db_service()
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
            "total_trades": 0,
        }
        saved = db_service.save_backtest_result(backtest_data, market="US")
        if saved:
            logger.info("Saved FML %s result to database", ch_key)
        return saved
    except Exception as e:
        logger.error("Failed to save FML %s to database: %s", ch_key, e)
        return False


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

    # Text output (strip "demo" references from chapter output)
    import re
    text = result.get("text", "")
    text = re.sub(r"(?mi)^.*\bdemo\b.*\n?", "", text)
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
