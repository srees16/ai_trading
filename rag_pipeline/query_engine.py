"""
RAG Query Engine for Centurion Capital LLC.

Orchestrates:
    1. Embed the user query
    2. Retrieve top-k similar chunks from ChromaDB
    3. Format a context-augmented prompt
    4. (Optional) Call an LLM for a synthesised answer

The engine is intentionally LLM-agnostic so you can plug in OpenAI,
Anthropic, a local model, or simply return raw retrieved chunks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.llm_service import OllamaLLMBackend
from rag_pipeline.reranker import CrossEncoderReranker
from rag_pipeline.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


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
    High-level RAG query orchestrator.

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
    ) -> None:
        self._vs = vector_store
        self._config = config or RAGConfig()
        self._embedder = embedding_service or EmbeddingService(self._config)
        self._reranker = reranker
        if self._reranker is None and self._config.reranker_enabled:
            self._reranker = CrossEncoderReranker(self._config)

        # LLM backend: use Ollama if configured, else fallback
        if llm_backend is not None:
            self._llm = llm_backend
        elif self._config.llm_provider == "ollama":
            self._llm = OllamaLLMBackend(
                model=self._config.llm_model,
                base_url=self._config.llm_base_url,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
                timeout=self._config.llm_timeout,
            )
        else:
            self._llm = DefaultLLMBackend()

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a user query.

        1. Embed the query
        2. Retrieve similar chunks from ChromaDB
        3. Build context string
        4. Generate answer via LLM backend
        """
        top_k = top_k or self._config.top_k

        # Step 1: Embed
        query_embedding = self._embedder.embed_query(query_text)

        # Step 2: Retrieve
        results = self._vs.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Step 3: Parse results
        chunks = self._parse_results(results)
        logger.info(
            "Retrieved %d chunks (distances: %s)",
            len(chunks),
            [f"{c.distance:.4f}" for c in chunks[:5]],
        )

        # Filter by similarity threshold (cosine distance 0-2; lower = better)
        threshold = self._config.similarity_threshold
        chunks = [c for c in chunks if c.distance <= threshold]
        logger.info(
            "%d chunks passed threshold filter (threshold=%.2f)",
            len(chunks), threshold,
        )

        # Step 4: Re-rank with cross-encoder (if enabled)
        if self._reranker and chunks:
            chunks = self._reranker.rerank(
                query_text, chunks, top_n=self._config.rerank_top_n
            )

        # Step 5: Build context
        context = self._build_context(chunks)

        # Step 6: Generate answer
        answer = self._llm.generate(query_text, context)

        return RAGResponse(
            query=query_text,
            answer=answer,
            chunks=chunks,
            rag_enabled=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _build_context(chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not chunks:
            return ""

        parts: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source: {chunk.source} | Chunk {chunk.chunk_index}]"
            parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(parts)
