"""
RAG Query Engine for Centurion Capital LLC.

Orchestrates the full latency-optimised RAG pipeline:

    1. **Semantic cache** — skip pipeline if a near-duplicate response
       is already cached (< 5 ms).
    2. **FAQ fast-path** — check a small, high-confidence FAQ collection
       before running the full pipeline (< 50 ms).
    3. **Query rewriting** — expand the user query into multiple
       paraphrases for broader recall.
    4. **Concurrent embedding** — embed rewritten queries in parallel
       with ``concurrent.futures``.
    5. **Hybrid search** — BM25 + vector retrieval with RRF merging.
    6. **Threshold filter & dedup** — remove low-score / duplicate chunks.
    7. **Cross-encoder re-rank** — score the top candidates with a
       lightweight cross-encoder model.
    8. **Token-budget context builder** — dynamically trim the context
       to fit within a token budget before sending to the LLM.
    9. **LLM generation** — blocking or streaming answer generation.
   10. **Performance tracing** — per-stage latency logged and surfaced
       in the ``RAGResponse``.

The engine is LLM-agnostic: plug in Ollama, Claude, OpenAI, or any
object that satisfies ``generate(query, context) -> str``.
"""

import logging
import re as _re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol

from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.evaluation import RetrievalLogger
from rag_pipeline.hybrid_search import HybridSearcher
from rag_pipeline.llm_service import create_llm_backend
from rag_pipeline.perf_trace import PipelineTrace
from rag_pipeline.query_rewriter import QueryRewriter
from rag_pipeline.reranker import CrossEncoderReranker
from rag_pipeline.semantic_cache import SemanticCache
from rag_pipeline.tiered_retrieval import TieredRetriever
from rag_pipeline.token_counter import budget_chunks, count_tokens
from rag_pipeline.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata boost — exact match for snippet / section references in query
# ---------------------------------------------------------------------------

_QUERY_SNIPPET_RE = _re.compile(
    r"(?:snippet|code\s*(?:listing|example)?|listing|algorithm|example)"
    r"\s*([\d.]+)",
    _re.IGNORECASE,
)
_QUERY_SECTION_RE = _re.compile(
    r"(?:section|chapter)\s*([\d.]+)", _re.IGNORECASE,
)
_QUERY_NUMBERED_REF_RE = _re.compile(
    r"\b(\d+\.\d+)\b",  # any N.N that might be a section/snippet ref
)

_BOOST_STOPWORDS = frozenset(
    "a an and are as at be but by for from give gives has have how i if in "
    "into is it me my no not of on or our so such that the their them then "
    "there these they this to us using was we what when which who will with".split()
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata."""
    text: str
    source: str
    chunk_index: int
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete response from the RAG query engine."""
    query: str
    answer: str
    chunks: List[RetrievedChunk]
    rag_enabled: bool = True
    cached: bool = False
    faq_hit: bool = False
    trace: Optional[PipelineTrace] = None


# ---------------------------------------------------------------------------
# LLM callback protocol (plug-in any model)
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """
    Any callable that takes (query, context_str) → answer_str.

    Implement this to plug in OpenAI, Anthropic, local LLM, etc.
    """

    def generate(self, query: str, context: str) -> str:
        ...


class DefaultLLMBackend:
    """
    Fallback – returns the retrieved context verbatim.

    Used only when Ollama is not configured / available.
    """

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return (
                "No relevant documents found in the knowledge base. "
                "Please upload strategy PDFs first."
            )
        return (
            f"**Retrieved context for:** *{query}*\n\n"
            "---\n\n"
            f"{context}"
        )


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class RAGQueryEngine:
    """
    High-level RAG query orchestrator with latency optimisations.

    Pipeline flow:
        cache check → FAQ fast-path → query rewrite → concurrent embed
        → hybrid search → threshold + dedup → re-rank → token-budget
        context build → LLM generate → cache store

    Usage:
        engine = RAGQueryEngine(vector_store)
        response = engine.query("What momentum indicators are used?")
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: Optional[RAGConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_backend: Optional[LLMBackend] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self._vs = vector_store
        self._config = config or RAGConfig()
        self._embedder = embedding_service or EmbeddingService(self._config)
        self._reranker = reranker
        if self._reranker is None and self._config.reranker_enabled:
            self._reranker = CrossEncoderReranker(self._config)

        # Query rewriter for multi-query expansion
        self._rewriter = query_rewriter
        if self._rewriter is None and self._config.query_rewrite_enabled:
            self._rewriter = QueryRewriter(self._config)

        # Hybrid BM25+vector search
        self._hybrid = None
        if self._config.hybrid_search_enabled:
            self._hybrid = HybridSearcher(
                self._vs, self._embedder, self._config
            )

        # LLM backend: use provided or auto-create from config
        if llm_backend is not None:
            self._llm = llm_backend
        else:
            self._llm = create_llm_backend(self._config)

        # Semantic cache
        self._cache: Optional[SemanticCache] = None
        cache_enabled = getattr(self._config, "cache_enabled", False)
        if cache_enabled:
            self._cache = SemanticCache(
                embedding_fn=self._embedder.embed_query,
                similarity_threshold=getattr(
                    self._config, "cache_similarity_threshold", 0.95
                ),
                ttl_seconds=getattr(self._config, "cache_ttl_seconds", 3600),
                max_entries=getattr(self._config, "cache_max_entries", 256),
            )

        # Tiered retrieval (FAQ fast-path)
        self._tiered: Optional[TieredRetriever] = None
        faq_enabled = getattr(self._config, "faq_enabled", False)
        if faq_enabled:
            self._tiered = TieredRetriever(
                vector_store=self._vs,
                embedding_service=self._embedder,
                config=self._config,
                faq_similarity_threshold=getattr(
                    self._config, "faq_similarity_threshold", 0.90
                ),
            )

        # Token budget for context
        self._context_token_budget: int = getattr(
            self._config, "context_token_budget", 2000
        )

        # Performance tracing toggle
        self._perf_logging: bool = getattr(
            self._config, "perf_logging_enabled", True
        )

        # Streaming toggle
        self._streaming_enabled: bool = getattr(
            self._config, "streaming_enabled", False
        )

        # Retrieval logger for offline analysis
        self._retrieval_logger = RetrievalLogger()

    # ------------------------------------------------------------------ #
    # Cache invalidation callback (wire into ingestion service)
    # ------------------------------------------------------------------ #

    def invalidate_cache(self, source_name: str, action: str = "changed") -> None:
        """
        Invalidate semantic cache entries related to a source document.

        This should be called whenever documents are ingested or deleted
        so that stale cached answers are not served.

        Args:
            source_name: The PDF filename whose cache entries to invalidate.
            action: Descriptive action string (\"ingested\", \"deleted\").
        """
        if self._cache is not None:
            removed = self._cache.invalidate(source=source_name)
            logger.info(
                "Cache invalidated for '%s' (action=%s): %d entries removed",
                source_name, action, removed,
            )
        # Also force BM25 index rebuild on next search
        if self._hybrid is not None:
            self._hybrid._bm25_count = -1
            logger.debug("Forced BM25 index rebuild after %s of '%s'", action, source_name)

    # ------------------------------------------------------------------ #
    # Main query entry point
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
        skip_cache: bool = False,
    ) -> RAGResponse:
        """
        Run the full latency-optimised RAG pipeline.

        Steps:
            1. Semantic cache check
            2. FAQ fast-path
            3. Query rewrite + concurrent embed
            4. Hybrid/vector search
            5. Threshold filter + dedup
            6. Cross-encoder re-rank
            7. Token-budget context trim
            8. LLM generate
            9. Cache store + trace summary
        """
        trace = PipelineTrace()
        trace.start()
        top_k = top_k or self._config.top_k
        space_id = self._config.default_space_id

        # Include source_filter in cache key so switching sources
        # does not return stale cached answers from a different PDF.
        cache_space = f"{space_id}::src={source_filter}" if source_filter else space_id

        # --- Step 1: Semantic cache check ---
        if self._cache is not None and not skip_cache:
            with trace.span("cache_lookup"):
                hit = self._cache.lookup(query_text, space_id=cache_space)
            if hit is not None:
                trace.stop()
                if self._perf_logging:
                    trace.summary()
                return RAGResponse(
                    query=query_text,
                    answer=hit.answer,
                    chunks=[],
                    rag_enabled=True,
                    cached=True,
                    trace=trace,
                )

        # --- Step 2: FAQ fast-path ---
        if self._tiered is not None:
            with trace.span("faq_lookup"):
                faq_hit = self._tiered.check_faq(query_text)
            if faq_hit is not None:
                trace.stop()
                if self._perf_logging:
                    trace.summary()
                return RAGResponse(
                    query=query_text,
                    answer=faq_hit.answer,
                    chunks=[],
                    rag_enabled=True,
                    faq_hit=True,
                    trace=trace,
                )

        # Build metadata filter
        effective_where = self._build_where_filter(where, source_filter)

        # --- Step 3: Query rewriting ---
        queries = [query_text]
        if self._rewriter:
            with trace.span("query_rewrite"):
                queries = self._rewriter.rewrite(query_text)
            logger.info("Multi-query expansion: %d variants", len(queries))

        # --- Step 4: Concurrent embed + retrieve ---
        all_chunks: List[RetrievedChunk] = []
        seen_texts: set = set()

        with trace.span("embed_and_retrieve", n_queries=len(queries)):
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
                futures = {
                    pool.submit(
                        self._embed_and_search, q, top_k, effective_where
                    ): q
                    for q in queries
                }
                for future in as_completed(futures):
                    try:
                        for chunk in future.result():
                            chunk_key = chunk.text[:200]
                            if chunk_key not in seen_texts:
                                seen_texts.add(chunk_key)
                                all_chunks.append(chunk)
                    except Exception as exc:
                        logger.warning(
                            "Query variant failed: %s", exc, exc_info=True
                        )

        chunks = all_chunks
        # Sort by distance (lower = better)
        chunks.sort(key=lambda c: c.distance)
        logger.info(
            "Retrieved %d chunks (distances: %s)",
            len(chunks),
            [f"{c.distance:.4f}" for c in chunks[:5]],
        )

        # --- Step 5: Threshold filter + dedup ---
        with trace.span("filter_and_dedup", before=len(chunks)):
            threshold = self._config.similarity_threshold
            chunks = [c for c in chunks if c.distance <= threshold]
            chunks = self._deduplicate_chunks(chunks)
        logger.info(
            "%d chunks after threshold+dedup (threshold=%.2f)",
            len(chunks), threshold,
        )

        # --- Step 5b: Metadata boost (exact match on snippet/section) ---
        if chunks:
            with trace.span("metadata_boost", n_candidates=len(chunks)):
                chunks = self._apply_metadata_boost(query_text, chunks)

        # --- Step 6: Re-rank ---
        if self._reranker and chunks:
            with trace.span("rerank", n_candidates=len(chunks)):
                chunks = self._reranker.rerank(
                    query_text, chunks, top_n=self._config.rerank_top_n
                )
            if not chunks:
                logger.warning(
                    "All chunks filtered out by reranker score threshold "
                    "(threshold=%.2f). No relevant context for LLM.",
                    getattr(self._config, "rerank_score_threshold", 0.25),
                )

        # --- Relevance gate: refuse to hallucinate if no relevant chunks ---
        if not chunks:
            logger.warning(
                "RELEVANCE GATE: No relevant chunks survived for query: %.80s",
                query_text,
            )
            trace.stop()
            if self._perf_logging:
                trace.summary()
            return RAGResponse(
                query=query_text,
                answer=(
                    "I could not find relevant information in the uploaded "
                    "documents to answer this question.\n\n"
                    "**Possible reasons:**\n"
                    "- The relevant document may not have been uploaded yet\n"
                    "- The query may need to be rephrased to match the "
                    "document content\n"
                    "- The uploaded documents may not cover this topic\n\n"
                    "Please upload the relevant PDF or try rephrasing your query."
                ),
                chunks=[],
                rag_enabled=True,
                trace=trace,
            )

        # --- Step 7: Token-budget context build ---
        max_ctx = self._config.max_context_chunks
        if len(chunks) > max_ctx:
            logger.info(
                "Trimming %d chunks to max_context_chunks=%d",
                len(chunks), max_ctx,
            )
            chunks = chunks[:max_ctx]

        with trace.span("build_context", n_chunks=len(chunks)):
            context = self._build_context(chunks, self._context_token_budget)

        # --- Step 8: LLM generate ---
        with trace.span("llm_generate"):
            answer = self._llm.generate(query_text, context)

        # --- Step 9: Cache store + trace ---
        if self._cache is not None:
            chunks_summary = [
                {"source": c.source, "chunk_index": c.chunk_index}
                for c in chunks
            ]
            self._cache.store(
                query_text, answer, chunks_summary, space_id=cache_space
            )

        # --- Step 10: Log retrieval for offline analysis ---
        try:
            self._retrieval_logger.log(
                query=query_text,
                retrieved_ids=[
                    c.metadata.get("chunk_hash", f"{c.source}_{c.chunk_index}")
                    for c in chunks
                ],
                retrieved_sources=[c.source for c in chunks],
                distances=[c.distance for c in chunks],
                rerank_scores=[
                    c.metadata.get("rerank_score", 0.0) for c in chunks
                ],
                extra={
                    "space_id": cache_space,
                    "latency_ms": round(trace.total_ms, 1),
                },
            )
        except Exception as exc:
            logger.debug("Retrieval logging failed: %s", exc)

        trace.stop()
        if self._perf_logging:
            trace.summary()

        return RAGResponse(
            query=query_text,
            answer=answer,
            chunks=chunks,
            rag_enabled=True,
            trace=trace,
        )

    # ------------------------------------------------------------------ #
    # Streaming query
    # ------------------------------------------------------------------ #

    def query_stream(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
        skip_cache: bool = False,
    ) -> Generator[str, None, None]:
        """
        Run the retrieval pipeline, then stream the LLM answer
        token-by-token.

        Everything up to the LLM call is identical to ``query()``.
        The LLM step uses ``generate_stream()`` to yield tokens.
        Includes TTFT (time-to-first-token) tracing.
        """
        import time as _time

        trace = PipelineTrace()
        trace.start()
        top_k = top_k or self._config.top_k
        space_id = self._config.default_space_id

        # Include source_filter in cache key so switching sources
        # does not return stale cached answers from a different PDF.
        cache_space = f"{space_id}::src={source_filter}" if source_filter else space_id

        # Cache check
        if self._cache is not None and not skip_cache:
            with trace.span("cache_lookup"):
                hit = self._cache.lookup(query_text, space_id=cache_space)
            if hit is not None:
                trace.stop()
                if self._perf_logging:
                    trace.summary()
                yield hit.answer
                return

        # FAQ fast-path
        if self._tiered is not None:
            with trace.span("faq_lookup"):
                faq_hit = self._tiered.check_faq(query_text)
            if faq_hit is not None:
                trace.stop()
                if self._perf_logging:
                    trace.summary()
                yield faq_hit.answer
                return

        effective_where = self._build_where_filter(where, source_filter)

        # Query rewrite
        queries = [query_text]
        if self._rewriter:
            with trace.span("query_rewrite"):
                queries = self._rewriter.rewrite(query_text)

        # Concurrent embed + retrieve
        all_chunks: List[RetrievedChunk] = []
        seen_texts: set = set()
        with trace.span("embed_and_retrieve", n_queries=len(queries)):
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
                futures = {
                    pool.submit(
                        self._embed_and_search, q, top_k, effective_where
                    ): q
                    for q in queries
                }
                for future in as_completed(futures):
                    try:
                        for chunk in future.result():
                            chunk_key = chunk.text[:200]
                            if chunk_key not in seen_texts:
                                seen_texts.add(chunk_key)
                                all_chunks.append(chunk)
                    except Exception:
                        pass

        chunks = all_chunks
        chunks.sort(key=lambda c: c.distance)

        with trace.span("filter_and_dedup", before=len(chunks)):
            threshold = self._config.similarity_threshold
            chunks = [c for c in chunks if c.distance <= threshold]
            chunks = self._deduplicate_chunks(chunks)

        # Metadata boost (exact match on snippet/section)
        if chunks:
            with trace.span("metadata_boost", n_candidates=len(chunks)):
                chunks = self._apply_metadata_boost(query_text, chunks)

        if self._reranker and chunks:
            with trace.span("rerank", n_candidates=len(chunks)):
                chunks = self._reranker.rerank(
                    query_text, chunks, top_n=self._config.rerank_top_n
                )
            if not chunks:
                logger.warning(
                    "Streaming: all chunks filtered by reranker threshold."
                )

        # --- Relevance gate (streaming) ---
        if not chunks:
            logger.warning(
                "RELEVANCE GATE (stream): No relevant chunks for query: %.80s",
                query_text,
            )
            trace.stop()
            if self._perf_logging:
                trace.summary()
            yield (
                "I could not find relevant information in the uploaded "
                "documents to answer this question.\n\n"
                "Please upload the relevant PDF or try rephrasing your query."
            )
            return

        max_ctx = self._config.max_context_chunks
        if len(chunks) > max_ctx:
            chunks = chunks[:max_ctx]

        with trace.span("build_context", n_chunks=len(chunks)):
            context = self._build_context(chunks, self._context_token_budget)

        # Stream LLM tokens with TTFT tracking
        if hasattr(self._llm, "generate_stream"):
            collected: List[str] = []
            llm_start = _time.perf_counter()
            ttft_recorded = False
            for token in self._llm.generate_stream(query_text, context):
                if not ttft_recorded:
                    ttft_ms = (_time.perf_counter() - llm_start) * 1000.0
                    # Record TTFT as a span with metadata
                    with trace.span("llm_ttft", ttft_ms=round(ttft_ms, 1)):
                        pass
                    ttft_recorded = True
                collected.append(token)
                yield token
            # Record total LLM streaming time
            llm_total_ms = (_time.perf_counter() - llm_start) * 1000.0
            with trace.span("llm_stream_total", total_ms=round(llm_total_ms, 1)):
                pass
            # Cache the full answer
            if self._cache is not None:
                full_answer = "".join(collected)
                chunks_summary = [
                    {"source": c.source, "chunk_index": c.chunk_index}
                    for c in chunks
                ]
                self._cache.store(
                    query_text, full_answer, chunks_summary,
                    space_id=cache_space,
                )
        else:
            with trace.span("llm_generate"):
                answer = self._llm.generate(query_text, context)
            yield answer

        trace.stop()
        if self._perf_logging:
            trace.summary()

    # ------------------------------------------------------------------ #
    # Embed + search helper (used by ThreadPoolExecutor)
    # ------------------------------------------------------------------ #

    def _embed_and_search(
        self,
        query_text: str,
        top_k: int,
        where: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        """Embed a single query variant and retrieve chunks."""
        query_embedding = self._embedder.embed_query(query_text)

        if self._hybrid:
            results = self._hybrid.search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                where=where,
                bm25_weight=self._config.hybrid_bm25_weight,
                vector_weight=self._config.hybrid_vector_weight,
            )
        else:
            results = self._vs.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        return self._parse_results(results)

    def _build_where_filter(
        self,
        where: Optional[Dict[str, Any]],
        source_filter: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Merge caller-supplied filter with convenience parameters.

        Enforces multi-tenant space isolation when enabled in config.
        Returns a ChromaDB ``where`` dict or None.
        """
        filters: List[Dict[str, Any]] = []

        if where:
            filters.append(where)

        if source_filter:
            filters.append({"source": source_filter})

        # Multi-tenant space isolation
        if self._config.enforce_space_isolation:
            filters.append({"space_id": self._config.default_space_id})

        # Filter out very short chunks (noise) at query time
        if self._config.enable_metadata_filtering:
            filters.append({
                "word_count": {"$gte": self._config.min_chunk_word_count}
            })

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate_chunks(
        self, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Remove near-duplicate chunks based on chunk_hash or text overlap.

        Uses ``chunk_hash`` metadata when available (fast, O(n)).
        Falls back to simple text-prefix dedup otherwise.
        """
        if not chunks:
            return chunks

        seen_hashes: set = set()
        deduped: List[RetrievedChunk] = []

        for chunk in chunks:
            h = chunk.metadata.get("chunk_hash", "")
            if h and h in seen_hashes:
                continue
            if h:
                seen_hashes.add(h)
            deduped.append(chunk)

        removed = len(chunks) - len(deduped)
        if removed:
            logger.info("Dedup removed %d near-duplicate chunks", removed)
        return deduped

    @staticmethod
    def _parse_results(results: Dict[str, Any]) -> List[RetrievedChunk]:
        """Convert raw ChromaDB results into ``RetrievedChunk`` objects."""
        chunks: List[RetrievedChunk] = []
        if not results.get("ids") or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            chunks.append(
                RetrievedChunk(
                    text=docs[i] if i < len(docs) else "",
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    distance=dists[i] if i < len(dists) else 1.0,
                    metadata=meta,
                )
            )

        # Sort by similarity (lower distance = more similar for cosine)
        chunks.sort(key=lambda c: c.distance)
        return chunks

    # ------------------------------------------------------------------ #
    # Metadata boost — re-score chunks matching snippet / section refs
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_metadata_boost(
        query: str,
        chunks: List[RetrievedChunk],
        snippet_boost: float = 0.15,
        section_boost: float = 0.10,
        keyword_boost: float = 0.08,
    ) -> List[RetrievedChunk]:
        """Reduce distance (= boost) for chunks whose metadata matches
        explicit snippet / section references found in the query.

        Boost magnitudes (applied as distance reduction):
            - Snippet ID match:  ``snippet_boost``  (strongest)
            - Section ID match:  ``section_boost``
            - Title keyword overlap:  ``keyword_boost`` × overlap count
            - Code chunk when query asks for code:  0.05

        The resulting chunk list is re-sorted by boosted distance.
        """
        if not chunks:
            return chunks

        query_lower = query.lower()

        # Extract explicit references from the query
        snippet_refs = set(
            m.group(1) for m in _QUERY_SNIPPET_RE.finditer(query)
        )
        section_refs = set(
            m.group(1) for m in _QUERY_SECTION_RE.finditer(query)
        )
        numbered_refs = set(
            m.group(1) for m in _QUERY_NUMBERED_REF_RE.finditer(query)
        )

        # Content keywords (minus stopwords)
        query_words = set(query_lower.split()) - _BOOST_STOPWORDS

        # Does the query ask for code?
        wants_code = any(
            kw in query_lower
            for kw in (
                "code", "snippet", "function", "method",
                "implementation", "algorithm", "listing",
            )
        )

        boosted = False
        for chunk in chunks:
            meta = chunk.metadata
            boost = 0.0

            # --- Snippet ID match (strongest) ---
            chunk_snippet = str(meta.get("snippet_id", ""))
            if chunk_snippet and (
                chunk_snippet in snippet_refs
                or chunk_snippet in numbered_refs
            ):
                boost = max(boost, snippet_boost)

            # --- Section ID match ---
            chunk_section = str(meta.get("section_id", ""))
            if chunk_section and (
                chunk_section in section_refs
                or chunk_section in numbered_refs
            ):
                boost = max(boost, section_boost)

            # --- Title keyword overlap ---
            titles = " ".join([
                str(meta.get("snippet_title", "")),
                str(meta.get("section_title", "")),
                str(meta.get("section", "")),
            ]).lower()
            if titles.strip() and query_words:
                title_words = set(titles.split()) - _BOOST_STOPWORDS
                overlap = len(query_words & title_words)
                if overlap >= 2:
                    boost = max(boost, keyword_boost * min(overlap, 4))

            # --- Code chunk boost when query asks for code ---
            if wants_code and meta.get("contains_code"):
                boost = max(boost, 0.05)

            if boost > 0:
                chunk.distance = max(0.0, chunk.distance - boost)
                boosted = True

        if boosted:
            chunks.sort(key=lambda c: c.distance)
            logger.info(
                "Metadata boost applied: snippet_refs=%s, section_refs=%s, "
                "numbered_refs=%s",
                snippet_refs, section_refs, numbered_refs,
            )

        return chunks

    @staticmethod
    def _build_context(
        chunks: List[RetrievedChunk],
        token_budget: int = 4000,
    ) -> str:
        """Format retrieved chunks into a structured context string for the LLM.

        Each chunk is wrapped in clear delimiters with metadata headers so
        the LLM can:
          - Distinguish chunk boundaries unambiguously
          - Cite sources accurately using the provided metadata
          - Prioritise high-relevance chunks via the relevance/rerank scores

        Enforces a **token budget** — chunks are added in order of
        relevance until the budget is exhausted, ensuring the LLM
        prompt stays within a predictable size.
        """
        if not chunks:
            return ""

        # Build each chunk's formatted text, then apply budget
        formatted: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            page = meta.get("page_number", "?")
            section = meta.get("section", "")
            similarity = 1.0 - chunk.distance
            rerank_score = meta.get("rerank_score")
            title = meta.get("title", chunk.source)
            word_count = meta.get("word_count", len(chunk.text.split()))
            contains_code = meta.get("contains_code", False)

            # Structured header with all available metadata
            header_parts = [
                f"=== CONTEXT CHUNK {i}/{len(chunks)} ===",
                f"Source: {chunk.source}",
                f"Title: {title}",
                f"Page: {page}",
            ]
            if section:
                header_parts.append(f"Section: {section}")

            # Structural metadata (chapter / section / snippet)
            chapter = meta.get("chapter", "")
            chapter_title = meta.get("chapter_title", "")
            if chapter:
                ch_label = f"Chapter: {chapter}"
                if chapter_title:
                    ch_label += f" — {chapter_title}"
                header_parts.append(ch_label)
            section_id = meta.get("section_id", "")
            section_title = meta.get("section_title", "")
            if section_id:
                sec_label = f"Section ID: {section_id}"
                if section_title:
                    sec_label += f" — {section_title}"
                header_parts.append(sec_label)
            snippet_id = meta.get("snippet_id", "")
            snippet_title = meta.get("snippet_title", "")
            if snippet_id:
                snip_label = f"Snippet: {snippet_id}"
                if snippet_title:
                    snip_label += f" — {snippet_title}"
                header_parts.append(snip_label)

            header_parts.append(f"Vector Similarity: {similarity:.1%}")
            if rerank_score is not None:
                header_parts.append(f"Reranker Relevance: {rerank_score:.4f}")
            header_parts.append(f"Word Count: {word_count}")
            if contains_code:
                header_parts.append(
                    "⚠ THIS CHUNK CONTAINS SOURCE CODE — reproduce it "
                    "EXACTLY as written. Do NOT rewrite, paraphrase, or "
                    "generate alternative code."
                )
            header_parts.append(f"--- Content Start ---")
            header_parts.append(chunk.text)
            header_parts.append(f"--- Content End ---")

            formatted.append("\n".join(header_parts))

        # Apply token budget — include as many chunks as fit
        kept_indices = budget_chunks(
            formatted, max_total_tokens=token_budget, separator_tokens=15
        )
        if len(kept_indices) < len(formatted):
            logger.info(
                "Token budget trimmed context from %d to %d chunks "
                "(budget=%d tokens)",
                len(formatted), len(kept_indices), token_budget,
            )

        parts = [formatted[i] for i in kept_indices]
        return "\n\n".join(parts)
