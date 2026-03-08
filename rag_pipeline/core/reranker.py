"""
Cross-Encoder Re-Ranker for Centurion Capital LLC RAG Pipeline.

Two-stage retrieval strategy:
    Stage 1 (bi-encoder):  Fast approximate retrieval of top-N candidates
                           using cosine similarity on dense embeddings.
    Stage 2 (cross-encoder): Precise re-ranking of candidates using a
                             cross-encoder that scores (query, passage) pairs
                             jointly for much higher accuracy.

The cross-encoder model ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is:
    - Lightweight (~80 MB)
    - Purpose-built for passage re-ranking (trained on MS MARCO)
    - Runs locally, no API key required

Usage:
    from rag_pipeline.core.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, chunks, top_n=5)
"""

import logging
from typing import List, Optional, Protocol

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol – swap in any re-ranking backend
# ---------------------------------------------------------------------------

class RerankerBackend(Protocol):
    """Interface every re-ranker backend must satisfy."""

    def rerank(
        self, query: str, texts: List[str], top_n: int = 5
    ) -> List[tuple]:
        """
        Re-rank texts by relevance to query.

        Returns list of (index, score) tuples ordered by descending
        relevance, truncated to ``top_n``.
        """
        ...


# ---------------------------------------------------------------------------
# Default: sentence-transformers CrossEncoder (local, free)
# ---------------------------------------------------------------------------

class CrossEncoderBackend:
    """
    Local cross-encoder re-ranker using sentence-transformers.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
        - Trained on MS MARCO passage ranking
        - Returns a relevance score per (query, passage) pair
        - Higher score = more relevant
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("Loaded re-ranker model: %s", self._model_name)
        return self._model

    def rerank(
        self, query: str, texts: List[str], top_n: int = 5
    ) -> List[tuple]:
        """
        Score each (query, text) pair and return top_n (index, score)
        tuples sorted by descending relevance score.
        """
        if not texts:
            return []

        # Cross-encoder expects list of [query, passage] pairs
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Pair each index with its score, sort descending, take top_n
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in scored[:top_n]]


# ---------------------------------------------------------------------------
# High-level re-ranker service
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Re-ranking service used by the RAG query engine.

    Wraps a configurable backend and provides typed chunk re-ranking
    with a **relevance score threshold** to filter out irrelevant chunks.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        backend: Optional[RerankerBackend] = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._backend = backend or CrossEncoderBackend(
            model_name=self._config.reranker_model
        )
        self._score_threshold: float = getattr(
            self._config, "rerank_score_threshold", 0.25
        )

    def rerank(
        self,
        query: str,
        chunks: list,
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list:
        """
        Re-rank a list of RetrievedChunk objects and filter by relevance.

        Args:
            query:  The user query string.
            chunks: List of RetrievedChunk from the initial retrieval.
            top_n:  Number of top results to return after re-ranking.
            score_threshold: Minimum cross-encoder score to keep a chunk.
                If None, uses the configured default.

        Returns:
            Re-ranked list of RetrievedChunk, filtered by score threshold
            and truncated to top_n.  Each chunk's metadata is enriched
            with ``rerank_score``.
        """
        if not chunks:
            return []

        top_n = top_n or self._config.rerank_top_n
        threshold = score_threshold if score_threshold is not None else self._score_threshold

        texts = [chunk.text for chunk in chunks]
        # Get all scored results (not truncated) so we can filter first
        ranked = self._backend.rerank(query, texts, top_n=len(texts))

        reranked = []
        filtered_out = 0
        for idx, score in ranked:
            # Apply relevance score threshold — drop chunks the
            # cross-encoder considers irrelevant to the query
            if score < threshold:
                filtered_out += 1
                continue
            chunk = chunks[idx]
            chunk.metadata["rerank_score"] = round(score, 4)
            reranked.append(chunk)
            if len(reranked) >= top_n:
                break

        logger.info(
            "Re-ranked %d %d chunks (threshold=%.2f, filtered_out=%d) "
            "for query: %.60s…",
            len(chunks), len(reranked), threshold, filtered_out, query,
        )
        if reranked:
            scores = [c.metadata.get('rerank_score', 0.0) for c in reranked]
            logger.info(
                "Rerank scores: min=%.4f, max=%.4f, mean=%.4f",
                min(scores), max(scores), sum(scores) / len(scores),
            )
        return reranked
