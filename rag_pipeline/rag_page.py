"""
RAG Pipeline — Standalone Streamlit Page for Centurion Capital LLC.

This is a **placeholder page** that will later be replaced / integrated
into the main centurion_core UI. It demonstrates the full RAG workflow:
    1. Toggle RAG on/off
    2. Upload & ingest PDFs
    3. Query the knowledge base
    4. View / manage indexed documents

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
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #28a745;
            color: white;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #218838;
            color: white;
            border: none;
        }
        div.stButton > button:active,
        div.stButton > button:focus {
            background-color: #1e7e34;
            color: white;
            border: none;
        }
        </style>
        <h2 style="text-align:center; margin-bottom:0;">Strategy KB</h2>
        """,
        unsafe_allow_html=True,
    )

    # ---- RAG toggle (always visible) ------------------------------------
    rag_on = render_rag_toggle()

    # ---- Query section (always visible) ---------------------------------
    query_text = render_query_input()

    if st.button("Search", type="primary", key="rag_search_btn"):
        if not query_text:
            st.warning("Please enter a query first.")
        elif rag_on:
            with st.spinner("Searching knowledge base..."):
                engine = _get_query_engine()
                response = engine.query(query_text)
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
            #page_icon="📚",
            layout="wide",
        )
    except st.errors.StreamlitAPIException:
        pass  # already set by main app

    render_rag_page()
