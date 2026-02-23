"""
RAG Pipeline UI Components for Centurion Capital LLC.

Reusable Streamlit widgets that can be embedded in any page.
Designed to be replaced / composed into the main centurion_core UI.

Public API:
    - render_rag_toggle()      → bool  (RAG on/off state)
    - render_pdf_uploader()    → uploads & ingests PDFs
    - render_query_input()     → gets user query text
    - render_rag_response()    → displays RAG answer + sources
    - render_knowledge_base()  → shows collection stats & management
"""

import streamlit as st
import logging
from typing import Any, Dict, List, Optional

from rag_pipeline.config import RAGConfig
from rag_pipeline.vector_store import VectorStoreManager
from rag_pipeline.pdf_ingestion import PDFIngestionService
from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.query_engine import RAGQueryEngine, RAGResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session-state helpers (lazy singletons)
# ---------------------------------------------------------------------------

def _get_config() -> RAGConfig:
    if "rag_config" not in st.session_state:
        st.session_state["rag_config"] = RAGConfig()
    return st.session_state["rag_config"]


def _get_vector_store() -> VectorStoreManager:
    if "rag_vector_store" not in st.session_state:
        st.session_state["rag_vector_store"] = VectorStoreManager(_get_config())
    return st.session_state["rag_vector_store"]


def _get_embedding_service() -> EmbeddingService:
    if "rag_embedding_svc" not in st.session_state:
        st.session_state["rag_embedding_svc"] = EmbeddingService(_get_config())
    return st.session_state["rag_embedding_svc"]


def _get_ingestion_service() -> PDFIngestionService:
    if "rag_ingestion_svc" not in st.session_state:
        st.session_state["rag_ingestion_svc"] = PDFIngestionService(
            _get_vector_store(), _get_config(), _get_embedding_service()
        )
    return st.session_state["rag_ingestion_svc"]


def _get_query_engine() -> RAGQueryEngine:
    if "rag_query_engine" not in st.session_state:
        st.session_state["rag_query_engine"] = RAGQueryEngine(
            _get_vector_store(), _get_config(), _get_embedding_service()
        )
    return st.session_state["rag_query_engine"]


# ---------------------------------------------------------------------------
# 1. RAG Toggle
# ---------------------------------------------------------------------------

def render_rag_toggle() -> bool:
    """
    Render an on/off toggle for RAG-based responses.

    Returns the current toggle state (True = RAG enabled).
    """
    if "rag_enabled" not in st.session_state:
        st.session_state["rag_enabled"] = _get_config().rag_enabled

    st.session_state["rag_enabled"] = st.toggle(
        "RAG",
        value=st.session_state["rag_enabled"],
        help="When enabled, queries will retrieve context from uploaded strategy documents.",
    )
    return st.session_state["rag_enabled"]


# ---------------------------------------------------------------------------
# 2. PDF Uploader
# ---------------------------------------------------------------------------

def render_pdf_uploader() -> Optional[List[Dict[str, Any]]]:
    """
    Render a multi-file PDF uploader and ingest on submit.

    Returns ingestion stats list or None if nothing was uploaded.
    """
    col_upload, _ = st.columns([1, 2])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload strategy PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to build the knowledge base.",
            key="rag_pdf_uploader",
        )

        if not uploaded_files:
            return None

        if st.button("Ingest Documents", type="primary", key="rag_ingest_btn"):
            ingestion_svc = _get_ingestion_service()
            results: List[Dict[str, Any]] = []

            progress_bar = st.progress(0, text="Ingesting documents...")
            for i, file in enumerate(uploaded_files):
                progress_bar.progress(
                    (i) / len(uploaded_files),
                    text=f"Processing: {file.name}",
                )
                try:
                    stats = ingestion_svc.ingest_uploaded_bytes(
                        file_name=file.name,
                        file_bytes=file.read(),
                    )
                    results.append(stats)
                except Exception as e:
                    logger.error("Failed to ingest %s: %s", file.name, e)
                    results.append(
                        {"status": "error", "source": file.name, "error": str(e)}
                    )

            progress_bar.progress(1.0, text="Done!")

            # Display results
            for r in results:
                if r["status"] == "success":
                    st.success(
                        f"✅ **{r['source']}** — {r['chunks']} chunks from {r['pages']} pages"
                    )
                elif r["status"] == "skipped":
                    st.warning(f"⚠️ **{r['source']}** — skipped ({r.get('reason', '')})")
                else:
                    st.error(f"❌ **{r['source']}** — {r.get('error', 'unknown error')}")

            return results

    return None


# ---------------------------------------------------------------------------
# 3. Query Input
# ---------------------------------------------------------------------------

def render_query_input() -> Optional[str]:
    """
    Render a text input for RAG queries.

    Returns the query string or None if empty.
    """
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the entry and exit rules for the momentum strategy?",
        key="rag_query_input",
    )
    return query.strip() if query else None


# ---------------------------------------------------------------------------
# 4. RAG Response display
# ---------------------------------------------------------------------------

def render_rag_response(response: RAGResponse) -> None:
    """Render a RAGResponse with LLM-generated answer and collapsible sources."""
    st.markdown("### 💡 Answer")
    st.markdown(response.answer)

    if response.chunks:
        with st.expander(
            f"📚 Retrieved Sources ({len(response.chunks)} chunks)",
            expanded=False,
        ):
            for i, chunk in enumerate(response.chunks, 1):
                similarity = 1.0 - chunk.distance
                st.markdown(
                    f"**Chunk {i}** | Source: `{chunk.source}` | "
                    f"Similarity: `{similarity:.2%}`"
                )
                st.caption(
                    chunk.text[:500] + ("…" if len(chunk.text) > 500 else "")
                )
                if i < len(response.chunks):
                    st.divider()


# ---------------------------------------------------------------------------
# 5. Knowledge Base management
# ---------------------------------------------------------------------------

def render_knowledge_base() -> None:
    """Render collection stats and source management."""
    vs = _get_vector_store()
    stats = vs.get_collection_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", stats["total_documents"])
    col2.metric("PDF Sources", stats["total_sources"])
    col3.metric("Collection", stats["collection_name"])

    if stats["sources"]:
        st.markdown("**Indexed Sources:**")
        for src in stats["sources"]:
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"📄 {src}")
            if col_b.button("🗑️", key=f"del_{src}", help=f"Remove {src}"):
                ingestion_svc = _get_ingestion_service()
                removed = ingestion_svc.delete_source(src)
                st.success(f"Removed {removed} chunks from **{src}**")
                st.rerun()
    else:
        st.info("No documents indexed yet. Upload PDFs above to get started.")

    st.divider()
    if st.button("⚠️ Reset Entire Knowledge Base", type="secondary"):
        vs.reset_collection()
        st.warning("Knowledge base has been reset.")
        st.rerun()
