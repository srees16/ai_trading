"""
Hybrid BM25 + Vector Search for Centurion Capital LLC RAG Pipeline.

Combines lexical (BM25) keyword matching with dense vector similarity
to improve recall for both exact-match and semantic queries.

Architecture:
    1. BM25 retrieves top-N candidates by keyword relevance
    2. Vector search retrieves top-N candidates by embedding similarity
    3. Reciprocal Rank Fusion (RRF) merges both ranked lists
    4. Merged list is sent to the cross-encoder re-ranker

Usage:
    from rag_pipeline.core.hybrid_search import HybridSearcher

    searcher = HybridSearcher(vector_store, embedding_service, config)
    chunks = searcher.search("RSI momentum strategy", top_k=40)
"""

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline.config import RAGConfig
from rag_pipeline.storage.embeddings import EmbeddingService
from rag_pipeline.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight BM25 (no external dependency)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or "
    "such that the their then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenisation with stop-word removal."""
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


class BM25Index:
    """
    In-memory BM25 index over a corpus of text documents.

    Parameters follow the standard BM25 formulation:
        k1 = 1.5  (term-frequency saturation)
        b  = 0.75 (length normalisation)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._df: Dict[str, int] = defaultdict(int)  # doc frequency
        self._n_docs: int = 0
        self._built = False

    def build(self, documents: List[str]) -> None:
        """Index a list of raw text documents."""
        self._corpus = [_tokenize(doc) for doc in documents]
        self._doc_lens = [len(tokens) for tokens in self._corpus]
        self._n_docs = len(documents)
        self._avgdl = (
            sum(self._doc_lens) / self._n_docs if self._n_docs else 1.0
        )
        self._df = defaultdict(int)
        for tokens in self._corpus:
            seen = set(tokens)
            for t in seen:
                self._df[t] += 1
        self._built = True
        logger.info("BM25 index built: %d documents", self._n_docs)

    def query(self, query_text: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """
        Score all documents against *query_text*.

        Returns list of ``(doc_index, bm25_score)`` sorted descending,
        truncated to *top_n*.
        """
        if not self._built or not self._n_docs:
            return []

        q_tokens = _tokenize(query_text)
        if not q_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._corpus):
            score = self._score_doc(q_tokens, doc_tokens, self._doc_lens[idx])
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def _score_doc(
        self, q_tokens: List[str], doc_tokens: List[str], doc_len: int
    ) -> float:
        """BM25 score for a single document."""
        tf_map: Dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1

        score = 0.0
        for qt in q_tokens:
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            df = self._df.get(qt, 0)
            idf = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1.0
            )
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self._avgdl
            )
            score += idf * numerator / denominator
        return score


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each list is ``[(item_key, score), ...]`` in descending order.
    Returns a merged list sorted by RRF score (descending).

    ``k`` is the RRF constant (default 60, per the original paper).
    """
    fused: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (key, _score) in enumerate(ranked):
            fused[key] += 1.0 / (k + rank + 1)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Combines BM25 keyword search with dense vector similarity.

    The BM25 index is built lazily from all documents in the ChromaDB
    collection and cached in memory.  It is rebuilt when the collection
    size changes (new ingestions / deletions).

    Supports **adaptive weight tuning**: tracks which retrieval source
    (BM25 vs vector) contributes more to the final fused results and
    logs the effectiveness for offline analysis.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_service: EmbeddingService,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedding_service
        self._config = config or RAGConfig()
        self._bm25 = BM25Index()
        self._bm25_doc_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[Dict[str, Any]] = []
        self._bm25_count: int = -1  # force first build
        # Adaptive weight tracking
        self._query_count: int = 0
        self._vector_contribution: float = 0.0
        self._bm25_contribution: float = 0.0

    def _ensure_bm25(self) -> None:
        """Rebuild the BM25 index if the collection has changed."""
        current_count = self._vs.count()
        if current_count == self._bm25_count:
            return  # no change
        logger.info("Rebuilding BM25 index (collection size: %d)", current_count)
        # Fetch all documents from ChromaDB
        all_data = self._vs.collection.get(
            include=["documents", "metadatas"],
        )
        self._bm25_doc_ids = all_data.get("ids") or []
        self._bm25_docs = all_data.get("documents") or []
        self._bm25_metas = all_data.get("metadatas") or []
        self._bm25.build(self._bm25_docs)
        self._bm25_count = current_count

    def search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Perform hybrid BM25 + vector search.

        Returns results in the same format as VectorStoreManager.query()
        for drop-in compatibility.
        """
        top_k = top_k or self._config.top_k

        # --- Vector search ---
        if query_embedding is None:
            query_embedding = self._embedder.embed_query(query_text)

        vector_results = self._vs.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Build vector ranked list: [(doc_id, similarity)]
        v_ids = vector_results.get("ids", [[]])[0]
        v_dists = vector_results.get("distances", [[]])[0]
        vector_ranked: List[Tuple[str, float]] = [
            (doc_id, 1.0 - dist)  # convert cosine distance to similarity
            for doc_id, dist in zip(v_ids, v_dists)
        ]

        # --- BM25 search ---
        self._ensure_bm25()
        bm25_hits = self._bm25.query(query_text, top_n=top_k * 2)  # over-fetch
        # Filter BM25 hits by the same metadata constraints used for vector
        # search to prevent cross-tenant leakage
        bm25_ranked: List[Tuple[str, float]] = []
        for idx, score in bm25_hits:
            if idx >= len(self._bm25_doc_ids):
                continue
            if where and not self._matches_where(
                self._bm25_metas[idx] if idx < len(self._bm25_metas) else {},
                where,
            ):
                continue
            bm25_ranked.append((self._bm25_doc_ids[idx], score))
            if len(bm25_ranked) >= top_k:
                break

        # --- Reciprocal Rank Fusion ---
        fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        # Take top_k fused results
        top_ids = [doc_id for doc_id, _score in fused[:top_k]]

        if not top_ids:
            return vector_results  # fallback to vector-only

        # Build merged output — look up from vector results + BM25 cache
        v_lookup: Dict[str, int] = {
            doc_id: i for i, doc_id in enumerate(v_ids)
        }
        bm25_lookup: Dict[str, int] = {
            doc_id: i for i, doc_id in enumerate(self._bm25_doc_ids)
        }

        out_ids, out_docs, out_metas, out_dists = [], [], [], []
        v_docs = vector_results.get("documents", [[]])[0]
        v_metas = vector_results.get("metadatas", [[]])[0]

        for doc_id in top_ids:
            if doc_id in v_lookup:
                i = v_lookup[doc_id]
                out_ids.append(doc_id)
                out_docs.append(v_docs[i])
                out_metas.append(v_metas[i])
                out_dists.append(v_dists[i])
            elif doc_id in bm25_lookup:
                i = bm25_lookup[doc_id]
                out_ids.append(doc_id)
                out_docs.append(
                    self._bm25_docs[i] if i < len(self._bm25_docs) else ""
                )
                out_metas.append(
                    self._bm25_metas[i] if i < len(self._bm25_metas) else {}
                )
                out_dists.append(0.5)  # neutral distance for BM25-only hits

        logger.info(
            "Hybrid search: %d vector + %d BM25 → %d fused results",
            len(vector_ranked), len(bm25_ranked), len(out_ids),
        )

        # Track contribution of each retrieval source
        self._query_count += 1
        vector_ids = {doc_id for doc_id, _ in vector_ranked}
        bm25_ids = {doc_id for doc_id, _ in bm25_ranked}
        fused_ids = set(out_ids)
        vector_only = len(fused_ids & vector_ids - bm25_ids)
        bm25_only = len(fused_ids & bm25_ids - vector_ids)
        both = len(fused_ids & vector_ids & bm25_ids)
        total_fused = len(fused_ids) or 1

        self._vector_contribution += (vector_only + both) / total_fused
        self._bm25_contribution += (bm25_only + both) / total_fused

        logger.info(
            "Hybrid source breakdown: vector_only=%d, bm25_only=%d, "
            "both=%d | Cumulative avg vector=%.2f, bm25=%.2f (over %d queries)",
            vector_only, bm25_only, both,
            self._vector_contribution / self._query_count,
            self._bm25_contribution / self._query_count,
            self._query_count,
        )

        return {
            "ids": [out_ids],
            "documents": [out_docs],
            "metadatas": [out_metas],
            "distances": [out_dists],
        }

    # ------------------------------------------------------------------ #
    # Metadata filter compliance for BM25 hits
    # ------------------------------------------------------------------ #

    @staticmethod
    def _matches_where(
        meta: Dict[str, Any], where: Dict[str, Any]
    ) -> bool:
        """
        Check if *meta* satisfies a ChromaDB-style *where* filter.

        Supports:
            - Simple equality: {"key": "value"}
            - Comparison operators: {"key": {"$gte": 10}}
            - ``$and`` / ``$or`` combinators (recursive)

        This is a best-effort client-side filter to prune BM25 hits
        that would be excluded by the vector search's server-side filter,
        thereby preventing cross-tenant leakage.
        """
        if not where:
            return True

        if "$and" in where:
            return all(
                HybridSearcher._matches_where(meta, sub)
                for sub in where["$and"]
            )
        if "$or" in where:
            return any(
                HybridSearcher._matches_where(meta, sub)
                for sub in where["$or"]
            )

        for key, condition in where.items():
            if key.startswith("$"):
                continue
            val = meta.get(key)
            if isinstance(condition, dict):
                # Operator filter: {"$gte": 10}, {"$lte": 5}, etc.
                for op, target in condition.items():
                    if op == "$gte" and (val is None or val < target):
                        return False
                    if op == "$lte" and (val is None or val > target):
                        return False
                    if op == "$gt" and (val is None or val <= target):
                        return False
                    if op == "$lt" and (val is None or val >= target):
                        return False
                    if op == "$ne" and val == target:
                        return False
                    if op == "$eq" and val != target:
                        return False
            else:
                # Simple equality
                if val != condition:
                    return False
        return True
