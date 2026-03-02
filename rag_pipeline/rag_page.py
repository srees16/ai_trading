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

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from ui.components import load_logo_base64_small, render_header_bar, render_footer

# RAG UI helpers are imported lazily inside render_rag_page() to avoid
# pulling in the heavy RAG dependency chain (chromadb, numpy, etc.) at
# module-import time.  Only lightweight stdlib imports live here.

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _rag_page_css() -> str:
    """Return RAG-page CSS — cached so st.markdown runs once per session."""
    return """
    <style>
    /* ── Center content with max-width ── */
    .block-container > div {
        max-width: 720px;
        margin: 0 auto;
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

    /* Nudge the RAG spinner upward so it sits closer to the Submit button */
    .spinner-wrapper {
        margin-top: -1.2rem;
    }
    </style>
    """


def render_rag_page() -> None:
    """
    Full-page RAG interface.

    Can be called from centurion_core's page router or run standalone.
    """
    # Ensure RAG session-state keys exist (deferred from global init)
    from services.session import ensure_rag_state
    ensure_rag_state()

    from rag_pipeline.ui_components import (
        render_rag_toggle,
        render_pdf_uploader,
        render_query_input,
        render_rag_response,
        render_knowledge_base,
        _get_query_engine,
    )

    _logo_html = load_logo_base64_small()

    st.markdown(_rag_page_css(), unsafe_allow_html=True)

    render_header_bar(subtitle="Knowledge Engine")
    _user = st.session_state.get('username', 'unknown')
    logger.info("[user=%s] RAG Engine: page view", _user)

    # ---- RAG toggle (always visible) ------------------------------------
    rag_on = render_rag_toggle()
    logger.info("[user=%s] RAG Engine: RAG toggle=%s", _user, rag_on)

    # ---- Upload & manage sections (only when RAG is on) -----------------
    #      Placed BEFORE the query section so users see ingestion status
    #      and can decide which source to query.
    if rag_on:
        logger.info("[user=%s] RAG Engine: PDF upload section visible", _user)
        render_pdf_uploader()

    # ---- Knowledge-base source selector (radio buttons) -----------------
    selected_source = None
    if rag_on:
        from rag_pipeline.ui_components import render_kb_source_selector
        selected_source = render_kb_source_selector()
        logger.info("[user=%s] RAG Engine: KB source=%s", _user, selected_source)

    # ---- Query section (always visible) ---------------------------------
    query_text = render_query_input()

    # Check for re-submit trigger from the response UI
    _resubmit_query = st.session_state.pop("rag_resubmit_query", None)
    _is_resubmit = _resubmit_query is not None
    if _is_resubmit:
        query_text = _resubmit_query

    if _is_resubmit or st.button("Submit Query", type="primary", key="rag_search_btn"):
        import time as _time
        _wall_t0 = _time.perf_counter()

        logger.info("[user=%s] RAG Engine: Submit Query clicked — rag_on=%s, resubmit=%s, query='%.80s'",
                    _user, rag_on, _is_resubmit, query_text or '')
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
            from ui.components import spinner_html as _spinner_html
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown(
                _spinner_html(_rnd.choice(_SPINNER_LINES)),
                unsafe_allow_html=True,
            )
            engine = _get_query_engine()
            source_kw = {"source_filter": selected_source} if selected_source else {}
            if _is_resubmit:
                source_kw["skip_cache"] = True
                st.info("🔄 Re-submitting query for a fresh response…")

            # ---- Stream the LLM answer token-by-token ----
            # query_stream() handles retrieval + LLM streaming internally.
            # Tokens arrive every 1–2 s; we render them incrementally so
            # the user sees output immediately instead of waiting for
            # the full 600 s hard timeout.
            _ttft: float | None = None  # time-to-first-token
            answer_placeholder = st.empty()
            collected_tokens: list[str] = []
            for token in engine.query_stream(query_text, **source_kw):
                # Clear spinner on first token
                if not collected_tokens:
                    spinner_placeholder.empty()
                    _ttft = _time.perf_counter() - _wall_t0
                collected_tokens.append(token)
                # Re-render the accumulated answer so far
                answer_placeholder.markdown(
                    "### 💡 Answer\n\n" + "".join(collected_tokens)
                )

            full_answer = "".join(collected_tokens)

            # Once streaming is done, replace the incremental render
            # with the full response widget (includes feedback buttons,
            # sources, code-apply, etc.).
            answer_placeholder.empty()
            spinner_placeholder.empty()

            # Build a RAGResponse for render_rag_response().  The chunks
            # were consumed inside query_stream() so we pass an empty
            # list; the sources expander will be hidden.
            from rag_pipeline.query_engine import RAGResponse
            response = RAGResponse(
                query=query_text,
                answer=full_answer,
                chunks=[],
                rag_enabled=True,
            )
            # --- Wall-clock runtime: terminal log + UI badge ---
            _wall_elapsed = _time.perf_counter() - _wall_t0
            _ttft_str = f"{_ttft:.1f}s" if _ttft is not None else "n/a"
            logger.info(
                "[user=%s] RAG query complete — wall_time=%.2fs, ttft=%s, query='%.80s'",
                _user, _wall_elapsed, _ttft_str, query_text or '',
            )
            if _wall_elapsed < 60:
                _elapsed_label = f"{_wall_elapsed:.1f}s"
            else:
                _m, _s = divmod(_wall_elapsed, 60)
                _elapsed_label = f"{int(_m)}m {_s:.0f}s"
            _runtime_label = f"⏱️ Total runtime: **{_elapsed_label}** · First token: **{_ttft_str}**"

            render_rag_response(response, runtime_label=_runtime_label)
        else:
            # RAG disabled — pass query directly to LLM without retrieval
            from rag_pipeline.query_engine import RAGResponse
            response = RAGResponse(
                query=query_text,
                answer=f"*(RAG disabled — no document retrieval performed)*\n\n**Your query:** {query_text}",
                chunks=[],
                rag_enabled=False,
            )
            _wall_elapsed = _time.perf_counter() - _wall_t0
            logger.info(
                "[user=%s] RAG-off query complete — wall_time=%.2fs, query='%.80s'",
                _user, _wall_elapsed, query_text or '',
            )
            _runtime_label = f"⏱️ Total runtime: **{_wall_elapsed:.1f}s**"
            render_rag_response(response, runtime_label=_runtime_label)

    # ---- Knowledge Base management (only when RAG is on) ----------------
    if rag_on:
        with st.expander("Knowledge Base", expanded=False):
            render_knowledge_base()

    render_footer()


# ---------------------------------------------------------------------------
# Standalone runner – ``streamlit run rag_pipeline/rag_page.py``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal page config for standalone mode only.
    # NOTE: Do NOT use ``st.runtime.exists()`` here – it is True whenever
    # Streamlit is running, which means the block would also execute when
    # app.py *imports* this module, causing ``render_rag_page()`` to be
    # called twice and triggering a duplicate widget-key error.
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
