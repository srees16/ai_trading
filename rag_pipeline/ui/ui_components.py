"""
RAG Pipeline UI Components for Centurion Capital LLC.

Reusable Streamlit widgets that can be embedded in any page.
Designed to be replaced / composed into the main centurion_core UI.

Public API:
    - render_rag_toggle()         → bool  (RAG on/off state)
    - render_pdf_uploader()       → uploads & ingests PDFs
    - render_query_input()        → gets user query text
    - render_rag_response()       → displays RAG answer + sources
    - render_kb_source_selector() → radio-button KB source picker
    - render_knowledge_base()     → shows collection stats & management
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ui.components import spinner_html

import streamlit as st

# Heavy RAG imports are deferred to first use so that importing this
# module (which happens eagerly at routing time) stays near-instant.
# Only lightweight stdlib / typing imports live at module level.
if TYPE_CHECKING:
    from rag_pipeline.config import RAGConfig
    from rag_pipeline.storage.vector_store import VectorStoreManager
    from rag_pipeline.ingestion.pdf_ingestion import PDFIngestionService
    from rag_pipeline.storage.embeddings import EmbeddingService
    from rag_pipeline.core.query_engine import RAGQueryEngine, RAGResponse
    from rag_pipeline.llm.code_applier import (
        extract_code_blocks,
        list_strategy_files,
        generate_patch,
        apply_patch,
        revert_last_patch,
        CodeBlock,
        StrategyFileInfo,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feedback Logger
# ---------------------------------------------------------------------------

_FEEDBACK_LOG_PATH = Path(__file__).resolve().parent / "data" / "feedback_log.jsonl"


def _log_feedback(
    query: str,
    answer: str,
    feedback: str,
    sources: List[str],
    comment: str = "",
) -> None:
    """Log user feedback (thumbs up/down) to a JSONL file."""
    _FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer_preview": answer[:500],
        "feedback": feedback,  # "positive" or "negative"
        "sources": sources,
        "comment": comment,
    }
    with _FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(
        "Feedback logged: %s for query='%.60s'",
        feedback, query,
    )


# ---------------------------------------------------------------------------
# Session-state helpers (lazy singletons)
#
# All heavy RAG imports are deferred to the first call of each helper
# so the module can be imported near-instantly at routing time.
# ---------------------------------------------------------------------------

def _get_config():
    if "rag_config" not in st.session_state:
        from rag_pipeline.config import RAGConfig
        st.session_state["rag_config"] = RAGConfig()
    return st.session_state["rag_config"]


def _get_vector_store():
    if "rag_vector_store" not in st.session_state:
        from rag_pipeline.storage.vector_store import VectorStoreManager
        st.session_state["rag_vector_store"] = VectorStoreManager(_get_config())
    return st.session_state["rag_vector_store"]


def _get_embedding_service():
    if "rag_embedding_svc" not in st.session_state:
        from rag_pipeline.storage.embeddings import EmbeddingService
        st.session_state["rag_embedding_svc"] = EmbeddingService(_get_config())
    return st.session_state["rag_embedding_svc"]


def _get_ingestion_service():
    if "rag_ingestion_svc" not in st.session_state:
        from rag_pipeline.ingestion.pdf_ingestion import PDFIngestionService
        # Wire cache invalidation: when docs change, the ingestion service
        # notifies the query engine to clear stale cache entries.
        engine = _get_query_engine()
        st.session_state["rag_ingestion_svc"] = PDFIngestionService(
            _get_vector_store(),
            _get_config(),
            _get_embedding_service(),
            on_change_callback=engine.invalidate_cache,
        )
    return st.session_state["rag_ingestion_svc"]


def _get_query_engine():
    if "rag_query_engine" not in st.session_state:
        from rag_pipeline.core.query_engine import RAGQueryEngine
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
        key="rag_pipeline_toggle",
    )
    return st.session_state["rag_enabled"]


# ---------------------------------------------------------------------------
# 2. PDF Uploader
# ---------------------------------------------------------------------------

def render_pdf_uploader() -> Optional[List[Dict[str, Any]]]:
    """
    Render a multi-file PDF uploader with **background ingestion**.

    Files are submitted to ``BackgroundIngestionManager`` and processed
    asynchronously so the user can continue submitting queries while
    ingestion runs.  A live-updating status panel shows progress.

    Returns ingestion stats list for newly completed tasks, or None.
    """
    from rag_pipeline.ingestion.background_ingest import get_ingestion_manager, TaskStatus

    mgr = get_ingestion_manager()

    # ---- Show live status for active / recently-completed tasks --------
    _render_ingestion_status(mgr)

    # ---- File upload + submit button ------------------------------------
    col_upload, col_hint = st.columns([1, 2])
    with col_upload:
        uploaded_files = st.file_uploader(
            "📄 Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to build the knowledge base.",
            label_visibility="collapsed",
            key="rag_pdf_uploader",
        )

        if not uploaded_files:
            return None

        if st.button("📥 Ingest Documents", type="primary", key="rag_ingest_btn"):
            submitted_count = 0
            for file in uploaded_files:
                file_bytes = file.read()
                mgr.submit(file.name, file_bytes)
                submitted_count += 1
                logger.info("Queued background ingestion for %s", file.name)

            st.toast(
                f"{submitted_count} file(s) submitted for background ingestion. "
                "You can continue querying while they process.",
                icon="📥",
            )
            st.rerun()  # rerun to show the status panel immediately

    with col_hint:
        if mgr.has_active_tasks():
            st.info(
                "**Ingestion in progress** — you can submit queries "
                "for previously ingested documents while new files are processing."
            )

    # ---- Consume completed tasks and invalidate caches -----------------
    # Display is handled by _render_ingestion_status via get_recently_completed();
    # here we only invalidate the query cache and collect results for callers.
    completed = mgr.pop_completed()
    results: List[Dict[str, Any]] = []
    if completed:
        engine = _get_query_engine()
        for task in completed:
            if task.status == TaskStatus.COMPLETED and task.result:
                results.append(task.result)
                r = task.result
                if r["status"] == "success":
                    try:
                        engine.invalidate_cache(r["source"], "ingested")
                    except Exception:
                        pass
            elif task.status == TaskStatus.FAILED:
                results.append({
                    "status": "error",
                    "source": task.file_name,
                    "error": task.error,
                })

    # ---- Auto-refresh while ingestion is running -----------------------
    # Polls every 2 s so the progress bar stays in sync with the backend.
    # Calls st.rerun() which stops further execution in this render cycle.
    if mgr.has_active_tasks():
        import time
        time.sleep(2)
        st.rerun()

    return results if results else None


def _render_ingestion_status(mgr) -> None:
    """Render a live status panel for active background ingestion tasks.

    Active tasks auto-refresh every 2 s (via the polling loop in
    ``render_pdf_uploader``).  Recently-completed tasks are shown here
    until they are consumed by ``pop_completed()``.
    """
    from rag_pipeline.ingestion.background_ingest import TaskStatus

    active = mgr.get_active_tasks()
    recently_done = mgr.get_recently_completed(max_age_s=60)

    if not active and not recently_done:
        return

    with st.container():
        st.markdown("#### 📥 Ingestion Status")

        # Active tasks — progress is kept in sync by the auto-refresh loop
        for task in active:
            col_name, col_stage, col_pct = st.columns([2, 3, 1])
            with col_name:
                status_icon = "⏳" if task.status == TaskStatus.PENDING else "⚙️"
                st.markdown(f"{status_icon} **{task.file_name}**")
            with col_stage:
                st.caption(task.stage or "Queued…")
            with col_pct:
                st.progress(task.stage_pct, text=f"{task.stage_pct:.0%}")

        # Recently completed — shown until pop_completed() consumes them
        for task in recently_done:
            if task.status == TaskStatus.COMPLETED:
                r = task.result or {}
                if r.get("status") == "success":
                    st.success(
                        f"**{task.file_name}** — {r.get('chunks', '?')} chunks "
                        f"from {r.get('pages', '?')} pages"
                    )
                elif r.get("reason") == "already_ingested":
                    st.info(
                        f"**{task.file_name}** — already ingested "
                        f"({r.get('chunks', '?')} chunks persisted). Skipped."
                    )
                elif r.get("status") == "skipped":
                    st.warning(
                        f"**{task.file_name}** — skipped ({r.get('reason', '')})"
                    )
                else:
                    st.info(f"**{task.file_name}** — {r.get('status', 'done')}")
            elif task.status == TaskStatus.FAILED:
                st.error(f"**{task.file_name}** — {task.error or 'unknown error'}")

        if active:
            st.caption("*Updating automatically — submit queries below while ingestion runs.*")


# ---------------------------------------------------------------------------
# 3. Query Input
# ---------------------------------------------------------------------------

def render_query_input() -> Optional[str]:
    """
    Render a text input for RAG queries.

    Returns the query string or None if empty.
    """
    query = st.text_input(
        "🔍 Ask a question",
        placeholder="e.g. What are the entry and exit rules for the momentum strategy?",
        key="rag_query_input",
    )
    return query.strip() if query else None


# ---------------------------------------------------------------------------
# 4. RAG Response display
# ---------------------------------------------------------------------------

def _render_answer_content(text: str) -> None:
    """Render answer text with proper handling of code blocks and inline code.

    Splits the answer on fenced code blocks (```...```) and renders each
    segment with the appropriate Streamlit widget:
      - Fenced code blocks  → ``st.code()`` with syntax highlighting
      - Everything else      → ``st.markdown()`` (handles inline `code` too)
    """
    import re as _re

    # Pattern captures: optional language tag (supports c++, c#, etc.), then the code body
    _CODE_BLOCK = _re.compile(
        r"```([\w+#.-]*)\n?(.*?)```", _re.DOTALL
    )

    parts = _CODE_BLOCK.split(text)
    # split produces: [before, lang, code, between, lang, code, ..., after]
    idx = 0
    while idx < len(parts):
        if idx % 3 == 0:
            # Plain markdown segment
            segment = parts[idx].strip()
            if segment:
                st.markdown(segment, unsafe_allow_html=False)
        elif idx % 3 == 1:
            # Language tag (may be empty)
            lang = parts[idx].strip() or None
        elif idx % 3 == 2:
            # Code body
            code = parts[idx].rstrip("\n")
            if code:
                st.code(code, language=lang)
        idx += 1


def render_rag_response(response, *, runtime_label: str | None = None) -> None:
    """Render a RAGResponse with LLM-generated answer, feedback buttons,
    and collapsible sources.

    Supports both plain text and fenced code blocks in the answer so that
    code snippets returned by the LLM are syntax-highlighted.

    Parameters
    ----------
    runtime_label : str | None
        Pre-formatted runtime string (e.g. "Total runtime: …").
        Displayed right below the HITL feedback buttons when provided.
    """
    from rag_pipeline.llm.code_applier import extract_code_blocks

    st.markdown("### Answer")
    _render_answer_content(response.answer)

    # ---- Feedback buttons (thumbs up / thumbs down) ----
    if response.rag_enabled and response.answer:
        feedback_key = f"fb_{hash(response.query + response.answer[:100])}"
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])

        with fb_col1:
            if st.button("👍", key=f"{feedback_key}_up", help="Good answer"):
                sources = [c.source for c in response.chunks] if response.chunks else []
                _log_feedback(
                    query=response.query,
                    answer=response.answer,
                    feedback="positive",
                    sources=sources,
                )
                st.toast("Thanks for the feedback!", icon="✅")

        with fb_col2:
            if st.button("👎", key=f"{feedback_key}_down", help="Poor answer"):
                sources = [c.source for c in response.chunks] if response.chunks else []
                _log_feedback(
                    query=response.query,
                    answer=response.answer,
                    feedback="negative",
                    sources=sources,
                )
                st.toast("Feedback recorded — we'll improve!", icon="📝")

    # ---- Runtime badge (right below feedback emojis) ----
    if runtime_label:
        st.caption(runtime_label)

    # ---- Re-submit query button ----
    if response.rag_enabled and response.answer:
        resubmit_key = f"resub_{hash(response.query + response.answer[:80])}"
        st.divider()
        rs_col1, rs_col2 = st.columns([2, 5])
        with rs_col1:
            if st.button(
                "🔄 Re-submit",
                key=resubmit_key,
                help="Not satisfied with the answer? Re-run the query to get a fresh response.",
            ):
                st.session_state["rag_resubmit_query"] = response.query
                st.rerun()
        with rs_col2:
            st.caption("💡 Not happy with this answer? Re-submit to get a fresh response.")

    # ---- Apply Code Suggestion section ----
    if response.rag_enabled and response.answer:
        code_blocks = extract_code_blocks(response.answer)
        if code_blocks:
            _render_code_apply_section(code_blocks, response.query, response.answer)

    if response.chunks:
        with st.expander(
            f"Retrieved Sources ({len(response.chunks)} chunks)",
            expanded=False,
        ):
            for i, chunk in enumerate(response.chunks, 1):
                similarity = 1.0 - chunk.distance
                meta = getattr(chunk, "metadata", {}) or {}
                page = meta.get("page_number", "")
                section = meta.get("section", "")
                rerank_score = meta.get("rerank_score")

                info_parts = [
                    f"**Chunk {i}**",
                    f"Source: `{chunk.source}`",
                    f"Similarity: `{similarity:.2%}`",
                ]
                if rerank_score is not None:
                    info_parts.append(f"Rerank: `{rerank_score:.4f}`")
                if page:
                    info_parts.append(f"Page: `{page}`")
                if section:
                    info_parts.append(f"Section: `{section}`")
                st.markdown(" | ".join(info_parts))

                # Render chunk preview with code-block awareness
                preview = chunk.text[:800] + ("…" if len(chunk.text) > 800 else "")
                _render_answer_content(preview)

                if i < len(response.chunks):
                    st.divider()


# ---------------------------------------------------------------------------
# 5. Apply Code Suggestion
# ---------------------------------------------------------------------------

def _render_code_apply_section(
    code_blocks,
    query: str,
    answer: str,
) -> None:
    """Render the 'Apply Code to Strategy' UI panel.

    Shows extracted code snippets and lets the user pick a target strategy
    file.  On confirmation the LLM merges the snippets into the file with
    automatic backup and syntax validation.
    """
    from rag_pipeline.llm.code_applier import (
        list_strategy_files,
        generate_patch,
        apply_patch,
        revert_last_patch,
    )
    apply_key = f"apply_{hash(query + answer[:80])}"

    with st.expander(
        f"Apply Code Suggestion ({len(code_blocks)} snippet"
        f"{'s' if len(code_blocks) != 1 else ''})",
        expanded=False,
    ):
        # --- Show extracted snippets ---
        st.markdown("**Extracted code snippets from the answer:**")
        for blk in code_blocks:
            lang_label = blk.language or "python"
            st.caption(f"Snippet {blk.index + 1}  ·  {lang_label}")
            st.code(blk.code, language=lang_label)

        st.divider()

        # --- Strategy file picker ---
        strategy_files = list_strategy_files()
        if not strategy_files:
            st.warning("No strategy files found in the workspace.")
            return

        file_options = {
            f"{sf.category}/{sf.name}  ({sf.rel_path})": sf
            for sf in strategy_files
        }
        selected_label = st.selectbox(
            "Target strategy file",
            options=list(file_options.keys()),
            index=0,
            key=f"{apply_key}_target",
            help="Select the strategy file where the code should be applied.",
        )
        selected_file: StrategyFileInfo = file_options[selected_label]

        # --- Preview / Apply / Revert columns ---
        col_preview, col_apply, col_revert = st.columns([1, 1, 1])

        # ---- Preview button ----
        with col_preview:
            if st.button(
                "Preview",
                key=f"{apply_key}_preview",
                help="Generate a preview of the merged code without writing anything.",
            ):
                _preview_sp = st.empty()
                _preview_sp.markdown(spinner_html("Generating preview with LLM…"), unsafe_allow_html=True)
                modified_source, summary = generate_patch(
                    target_file=selected_file.path,
                    code_blocks=code_blocks,
                    query=query,
                )
                _preview_sp.empty()
                st.session_state[f"{apply_key}_preview_src"] = modified_source
                st.session_state[f"{apply_key}_preview_summary"] = summary

        # Show preview if available
        preview_src = st.session_state.get(f"{apply_key}_preview_src")
        preview_summary = st.session_state.get(f"{apply_key}_preview_summary")
        if preview_src:
            st.info(f"**Preview summary:** {preview_summary}")
            with st.expander("Preview — merged file", expanded=False):
                st.code(preview_src, language="python")

        # ---- Apply button ----
        with col_apply:
            if st.button(
                "Apply",
                key=f"{apply_key}_apply",
                help="Apply the code snippet to the selected strategy file (backup created automatically).",
            ):
                # Use preview if available, otherwise generate fresh
                source_to_apply = st.session_state.get(f"{apply_key}_preview_src")
                if not source_to_apply:
                    _merge_sp = st.empty()
                    _merge_sp.markdown(spinner_html("Generating merged code with LLM…"), unsafe_allow_html=True)
                    source_to_apply, summary = generate_patch(
                        target_file=selected_file.path,
                        code_blocks=code_blocks,
                        query=query,
                    )
                    _merge_sp.empty()
                    st.session_state[f"{apply_key}_preview_summary"] = summary

                result = apply_patch(
                    target_file=selected_file.path,
                    modified_source=source_to_apply,
                    query=query,
                )
                if result.success:
                    st.success(
                        f"**Code applied** to `{selected_file.rel_path}`\n\n"
                        f"{result.diff_summary}\n\n"
                        f"Backup: `{Path(result.backup_file).name}`"
                    )
                    # Clear preview cache
                    st.session_state.pop(f"{apply_key}_preview_src", None)
                    st.session_state.pop(f"{apply_key}_preview_summary", None)
                else:
                    st.error(f"{result.message}")

        # ---- Revert button ----
        with col_revert:
            if st.button(
                "↩Revert",
                key=f"{apply_key}_revert",
                help="Undo the last applied change and restore the backup.",
            ):
                result = revert_last_patch(selected_file.path)
                if result.success:
                    st.success(f"↩{result.message}")
                else:
                    st.warning(f"{result.message}")


# ---------------------------------------------------------------------------
# 6. Knowledge-Base Source Selector (Dropdown)
# ---------------------------------------------------------------------------

def render_kb_source_selector() -> Optional[str]:
    """Render a dropdown so the user can pick which uploaded PDF to query.

    Returns the selected source filename, or *None* for "All Sources".
    """
    vs = _get_vector_store()
    sources = vs.list_sources()

    if not sources:
        return None  # nothing uploaded yet

    options = ["All"] + sources
    choice = st.selectbox(
        "Source",
        options,
        index=0,
        key="rag_kb_source_select",
        help="Select a specific knowledge base (PDF) to search, or query all.",
    )
    return None if choice == "All" else choice


# ---------------------------------------------------------------------------
# 7. Knowledge Base management
# ---------------------------------------------------------------------------

def render_knowledge_base() -> None:
    """Render collection stats and source management."""
    vs = _get_vector_store()
    stats = vs.get_collection_stats()

    col1, col2, col3 = st.columns(3)
    st.markdown(
        "<style>[data-testid='stMetric'] {font-size: 0.85rem;} "
        "[data-testid='stMetricValue'] {font-size: 1.1rem;}</style>",
        unsafe_allow_html=True,
    )
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
                st.success(f"✅ Removed {removed} chunks from **{src}**")
                st.rerun()
    else:
        st.info("ℹ️ No documents indexed yet. Upload PDFs above to get started.")

    st.divider()
    col_reingest, col_reset = st.columns(2)
    with col_reingest:
        if st.button(
            "🔄 Re-ingest All Documents",
            type="secondary",
            help="Delete existing chunks and re-process all PDFs with the "
                 "latest chunking pipeline. Use after pipeline upgrades.",
        ):
            ingestion_svc = _get_ingestion_service()
            _reingest_sp = st.empty()
            _reingest_sp.markdown(spinner_html("Re-ingesting documents…"), unsafe_allow_html=True)
            results = ingestion_svc.reingest_all()
            _reingest_sp.empty()
            for r in results:
                if r["status"] == "success":
                    st.success(
                        f"**{r['source']}** — {r['chunks']} chunks from {r['pages']} pages"
                    )
                else:
                    st.warning(f"**{r.get('source', '?')}** — {r.get('status', 'unknown')}")
            st.rerun()
    with col_reset:
        if st.button("🗑️ Reset Knowledge Base", type="secondary"):
            vs.reset_collection()
            st.warning("⚠️ Knowledge base has been reset.")
            st.rerun()
