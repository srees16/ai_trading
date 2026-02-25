"""
RAG Pipeline — Standalone Streamlit Page for Centurion Capital LLC.

Usage:
    streamlit run rag_page.py

Integration notes:
    The page delegates all rendering to reusable widgets in
    ``rag_pipeline.ui_components``.  To embed RAG into ANY Streamlit
    page in centurion_core, simply import the individual widget
    functions and call them.

    Example:
        from rag_pipeline.ui_components import (
            render_rag_toggle,
            render_query_input,
            render_rag_response,
        )
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `rag_pipeline` is importable
# regardless of how this script is launched (standalone or via main app).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import logging

from ui.components import load_logo_base64_small
from rag_pipeline.ui_components import (
    render_rag_toggle,
    render_pdf_uploader,
    render_query_input,
    render_rag_response,
    render_knowledge_base,
    _get_query_engine,
)

logger = logging.getLogger(__name__)


def render_rag_page() -> None:
    """
    Full-page RAG interface.

    Can be called from centurion_core's page router or run standalone.
    """
    _logo_html = load_logo_base64_small()

    st.markdown(
        """
        <style>
        /* ── Center content with max-width ── */
        .block-container > div {
            max-width: 720px;
            margin: 0 auto;
        }

        /* ── Multi-color spinner ── */
        @keyframes centurion-spin {
            0%   { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes centurion-color {
            0%   { border-top-color: #3498db; }
            25%  { border-top-color: #e74c3c; }
            50%  { border-top-color: #f1c40f; }
            75%  { border-top-color: #2ecc71; }
            100% { border-top-color: #3498db; }
        }
        .centurion-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0,0,0,0.1);
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: centurion-spin 0.8s linear infinite,
                       centurion-color 2.4s ease-in-out infinite;
            vertical-align: middle;
        }
        .spinner-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 0;
        }
        .spinner-text {
            font-size: 0.92rem;
            color: #555;
            font-weight: 500;
            font-style: italic;
        }

        /* ── Code block styling ── */
        pre, code {
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', 'Monaco', monospace !important;
        }
        div[data-testid="stCode"] {
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        div[data-testid="stCode"] pre {
            padding: 1rem !important;
            font-size: 0.88rem !important;
            line-height: 1.5 !important;
        }
        /* Inline code in markdown */
        .stMarkdown code {
            background-color: rgba(100, 100, 100, 0.15);
            padding: 0.15em 0.4em;
            border-radius: 4px;
            font-size: 0.88em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<h2 style="text-align:center; margin:0 0 0.3rem 0; padding:0;">{_logo_html} Centurion Capital LLC</h2>',
        unsafe_allow_html=True,
    )

    # ---- RAG toggle (always visible) ------------------------------------
    rag_on = render_rag_toggle()

    # ---- Knowledge-base source selector (radio buttons) -----------------
    selected_source = None
    if rag_on:
        from rag_pipeline.ui_components import render_kb_source_selector
        selected_source = render_kb_source_selector()

    # ---- Query section (always visible) ---------------------------------
    query_text = render_query_input()

    # Check for re-submit trigger from the response UI
    _resubmit_query = st.session_state.pop("rag_resubmit_query", None)
    _is_resubmit = _resubmit_query is not None
    if _is_resubmit:
        query_text = _resubmit_query

    if _is_resubmit or st.button("Submit Query", type="primary", key="rag_search_btn"):
        if not query_text:
            st.warning("Please enter a query first.")
        elif rag_on:
            # Show multi-color spinner with a creative line
            import random as _rnd
            _SPINNER_LINES = [
                "Diving deep into your documents…",
                "Summoning insights from the vault…",
                "Connecting the dots across pages…",
                "Mining your knowledge base for gold…",
                "Crunching context at light speed…",
                "Reading between the lines for you…",
                "Paging through the archives…",
                "Hunting for the perfect answer…",
                "Cross-referencing your strategy docs…",
                "Assembling intelligence from the KB…",
            ]
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown(
                '<div class="spinner-wrapper">'
                '  <div class="centurion-spinner"></div>'
                f'  <span class="spinner-text">{_rnd.choice(_SPINNER_LINES)}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            engine = _get_query_engine()
            source_kw = {"source_filter": selected_source} if selected_source else {}
            if _is_resubmit:
                source_kw["skip_cache"] = True
                st.info("🔄 Re-submitting query for a fresh response…")
            response = engine.query(query_text, **source_kw)
            spinner_placeholder.empty()
            render_rag_response(response)
        else:
            # RAG disabled — pass query directly to LLM without retrieval
            from rag_pipeline.query_engine import RAGResponse
            response = RAGResponse(
                query=query_text,
                answer=f"*(RAG disabled — no document retrieval performed)*\n\n**Your query:** {query_text}",
                chunks=[],
                rag_enabled=False,
            )
            render_rag_response(response)

    # ---- Upload & manage sections (only when RAG is on) -----------------
    if rag_on:
        render_pdf_uploader()

        with st.expander("Knowledge Base", expanded=False):
            render_knowledge_base()


# ---------------------------------------------------------------------------
# Standalone runner – ``streamlit run rag_pipeline/rag_page.py``
# ---------------------------------------------------------------------------
if __name__ == "__main__" or st.runtime.exists():
    # Minimal page config for standalone mode
    try:
        st.set_page_config(
            page_title="Centurion RAG Pipeline",
            layout="wide",
        )
    except st.errors.StreamlitAPIException:
        pass  # already set by main app

    from ui.styles import apply_custom_styles
    apply_custom_styles()
    render_rag_page()
