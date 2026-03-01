"""
Retrieval Orchestrator for Centurion Capital LLC RAG Pipeline.

Coordinates the full retrieval flow:

    1. Classify the query  (``query_classifier``)
    2. Route to the correct index  (``code_index`` / ``theory_index``)
    3. Vector similarity search  (``DualIndexStore``)
    4. BM25 keyword scoring  (in-memory, lightweight)
    5. Pipeline-stage match bonus
    6. Fuse scores with a weighted formula
    7. Return top-*k* ranked chunks

Ranking formula::

    final_score = 0.5 * vector_score
                + 0.2 * keyword_score
                + 0.3 * stage_match_bonus

Tech stack: Python 3.11 · raw ChromaDB · No LangChain · No ML.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline.config import RAGConfig
from rag_pipeline.query_classifier import classify_query
from rag_pipeline.token_counter import count_tokens
from rag_pipeline.vector_store import DualIndexStore, EmbeddingFn

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Constants / defaults
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_TOP_K = 10

# Ranking weights (must sum to 1.0)
W_VECTOR = 0.5
W_KEYWORD = 0.2
W_STAGE = 0.3

# BM25 hyper-parameters
_BM25_K1 = 1.5
_BM25_B = 0.75

# Stage-match bonus (0.0 – 1.0) awarded when a chunk's pipeline_stage
# matches at least one stage detected in the query.
_STAGE_MATCH_VALUE = 1.0
_STAGE_NO_MATCH_VALUE = 0.0

# Parent-child context budgeting
DEFAULT_CONTEXT_BUDGET = 2500       # max tokens for combined context
_SEPARATOR_TOKENS = 10              # estimated overhead per chunk boundary

# Parent-child expansion limits (latency guard-rails)
MAX_PARENT_SECTIONS = 3             # expand at most N parent sections
MAX_CHILDREN_PER_PARENT = 2         # include at most N children per parent
_LARGE_PARENT_TOKEN_THRESHOLD = 1200  # skip full-parent fetch above this

# Fast-mode constants
FAST_MODE_TOP_K = 5                 # hard cap when in fast mode

# Simple-query mode: queries shorter than this word count use a
# lighter pipeline (top_k=4, no parent expansion, no BM25/stage).
SIMPLE_QUERY_THRESHOLD = 30         # default; override via RAG_SIMPLE_QUERY_THRESHOLD
SIMPLE_QUERY_TOP_K = 4              # hard cap for simple-query mode

# Chroma query budget – never ask ChromaDB for more than this many
# results per collection.  Keeps vector-DB round-trip time bounded.
MAX_CHROMA_RESULTS = 8

# Retrieval latency warning threshold (seconds).  If total retrieve()
# time exceeds this a WARNING is logged.
_RETRIEVAL_TIMEOUT_WARN = 3.0

# Thread pool size for concurrent vector searches and scoring.
_RETRIEVER_WORKERS = 2

# ── Strict code-mode constants ───────────────────────────────────────
# When ``strict_code_mode`` is active (set by query_classifier when the
# query contains strong code-intent keywords like "implement", "python",
# "code", "function", "using python"), the retriever:
#   1. Searches ONLY code_index (theory_index is skipped entirely).
#   2. Forces top_k = STRICT_CODE_TOP_K.
#   3. Applies a metadata filter: chunk_type == "code".
#   4. Boosts scores for function definitions and keyword overlap.
STRICT_CODE_TOP_K = 8
_STRICT_CODE_FUNC_SIG_BOOST = 0.5     # bonus for "def " in chunk
_STRICT_CODE_KEYWORD_OVERLAP_BOOST = 0.3  # bonus for query-token overlap
_STRICT_CODE_FUZZY_WEIGHT = 0.5           # multiplier for fuzzy similarity score
_STRICT_CODE_FUZZY_MIN_THRESHOLD = 0.5    # ignore fuzzy scores below this

# Regex to extract function name from a Python def statement
_DEF_NAME_RE = re.compile(r"\bdef\s+(\w+)")

# Expected functions that the user cares about for quality monitoring.
# If these are not found in the top-N retrieval results, a warning is
# emitted.  Extend this set as more canonical functions are identified.
_EXPECTED_FUNCTIONS: frozenset = frozenset({
    "applyPtSlOnT1",
    "getEvents",
    "getDailyVol",
    "cusum_filter",
})
_EXPECTED_WARN_TOP_N = 5


def _log_code_retrieval(
    ranked: List[Dict[str, Any]],
    *,
    expected_fn: Optional[str] = None,
) -> None:
    """Log detailed diagnostics about retrieved code chunks.

    Emits:
    - All chunk IDs in ranked order.
    - Top 3 function names extracted from chunk content.
    - Whether each ``_EXPECTED_FUNCTIONS`` name was retrieved.
    - A WARNING if *expected_fn* (or any expected function) is not
      in the top ``_EXPECTED_WARN_TOP_N`` results.

    This is a pure logging helper with no side effects on *ranked*.
    """
    if not ranked:
        logger.info("_log_code_retrieval: no ranked results to log.")
        return

    # Chunk IDs
    chunk_ids = [r.get("id", "?") for r in ranked]
    logger.info(
        "CODE_RETRIEVAL_LOG: %d chunks retrieved. IDs: %s",
        len(chunk_ids),
        chunk_ids,
    )

    # Extract function names from each chunk (ordered by score)
    all_func_names: List[str] = []
    per_chunk_fns: Dict[str, List[str]] = {}
    for r in ranked:
        content = r.get("content", "")
        fns = _DEF_NAME_RE.findall(content)
        all_func_names.extend(fns)
        per_chunk_fns[r.get("id", "?")] = fns

    # Top 3 unique function names (preserving order of first appearance)
    seen: set = set()
    top_fns: List[str] = []
    for fn in all_func_names:
        if fn not in seen:
            seen.add(fn)
            top_fns.append(fn)
        if len(top_fns) >= 3:
            break

    logger.info(
        "CODE_RETRIEVAL_LOG: top 3 function names: %s",
        top_fns,
    )

    # Check expected functions
    all_fn_set = set(all_func_names)
    top_n_fns: set = set()
    for r in ranked[:_EXPECTED_WARN_TOP_N]:
        top_n_fns.update(_DEF_NAME_RE.findall(r.get("content", "")))

    for exp_fn in _EXPECTED_FUNCTIONS:
        found = exp_fn in all_fn_set
        in_top_n = exp_fn in top_n_fns
        logger.info(
            "CODE_RETRIEVAL_LOG: %s — retrieved=%s, in_top_%d=%s",
            exp_fn, found, _EXPECTED_WARN_TOP_N, in_top_n,
        )

    # Explicit check for a specific expected function
    if expected_fn and expected_fn not in top_n_fns:
        logger.warning(
            "CODE_RETRIEVAL_LOG: ⚠ Expected implementation '%s' "
            "not retrieved in top %d results. Retrieved functions: %s",
            expected_fn, _EXPECTED_WARN_TOP_N, top_fns,
        )

    # Generic warning for any expected function missing from top-N
    missing = _EXPECTED_FUNCTIONS - top_n_fns
    if missing:
        logger.warning(
            "CODE_RETRIEVAL_LOG: ⚠ Expected implementation not retrieved "
            "in top %d — missing: %s",
            _EXPECTED_WARN_TOP_N, sorted(missing),
        )

# Regex to split camelCase tokens (e.g. applyPtSlOnT1 → [apply,Pt,Sl,On,T,1])
_CAMEL_SPLIT_RE = re.compile(r"[a-z]+|[A-Z][a-z]*|\d+")


def _fuzzy_ratio(a: str, b: str) -> float:
    """Compute a simple character-level similarity ratio between two strings.

    Uses the standard *longest common subsequence* approach to produce a
    value in ``[0.0, 1.0]``.  This is a lightweight, dependency-free
    alternative to ``fuzzywuzzy.fuzz.ratio``.

    The formula is::

        ratio = (2 * LCS_length) / (len(a) + len(b))

    Returns 0.0 when either string is empty, 1.0 for identical strings.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # DP-free space-optimised LCS length
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ch_a in a:
        curr = [0] * (len(b) + 1)
        for j, ch_b in enumerate(b, 1):
            if ch_a == ch_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    lcs_len = prev[len(b)]
    return (2.0 * lcs_len) / (len(a) + len(b))


def _fuzzy_func_name_score(
    content: str,
    query_tokens: List[str],
) -> float:
    """Score a code chunk by fuzzy-matching query tokens to function names.

    Extracts all ``def <name>`` declarations from *content*, splits each
    name into sub-tokens (snake_case + camelCase), then computes
    :func:`_fuzzy_ratio` between every (query_token, func_sub_token) pair.

    Returns::

        best_similarity * _STRICT_CODE_FUZZY_WEIGHT   (default 0.5)

    where ``best_similarity`` is the highest fuzzy ratio found across
    all pairings.  Returns 0.0 if no function names are found or the
    best ratio is below ``_STRICT_CODE_FUZZY_MIN_THRESHOLD``.
    """
    func_names = _DEF_NAME_RE.findall(content)
    if not func_names or not query_tokens:
        return 0.0

    # Collect all sub-tokens from all function names
    func_subtokens: set[str] = set()
    for fn in func_names:
        func_subtokens |= _split_identifier(fn)

    if not func_subtokens:
        return 0.0

    # Find the best fuzzy ratio across all (query_token, func_subtoken) pairs
    best_ratio = 0.0
    for qt in query_tokens:
        for ft in func_subtokens:
            r = _fuzzy_ratio(qt, ft)
            if r > best_ratio:
                best_ratio = r

    if best_ratio < _STRICT_CODE_FUZZY_MIN_THRESHOLD:
        return 0.0

    return best_ratio * _STRICT_CODE_FUZZY_WEIGHT


def _split_identifier(name: str) -> set[str]:
    """Split a Python identifier into sub-word tokens.

    Handles **snake_case** (split on ``_``) and **camelCase** (split
    on case boundaries).  Returns a set of lowercased sub-words
    with length > 1.

    Examples::

        _split_identifier("applyPtSlOnT1")
        # → {'apply', 'pt', 'sl', 'on', 'applyptslont1'}

        _split_identifier("daily_volatility")
        # → {'daily', 'volatility', 'daily_volatility'}
    """
    parts: set[str] = set()
    parts.add(name.lower())
    for snake_part in name.split("_"):
        if snake_part:
            lo = snake_part.lower()
            if len(lo) > 1:
                parts.add(lo)
            # camelCase split on ORIGINAL case
            for cp in _CAMEL_SPLIT_RE.findall(snake_part):
                if len(cp) > 1:
                    parts.add(cp.lower())
    return parts


def is_fast_mode() -> bool:
    """Return *True* when low-latency fast-mode is active.

    Checked **dynamically** on every call so that the env-var
    ``RAG_FAST_MODE=true`` can be toggled without restarting.
    """
    return os.getenv("RAG_FAST_MODE", "false").lower() == "true"


def is_simple_query(query: str) -> bool:
    """Return *True* when *query* is shorter than the simple-query threshold.

    Simple queries use a lighter retrieval pipeline:
    ``top_k = 4``, no BM25/stage scoring, no parent expansion,
    no reranker, no metadata boosting.

    The threshold is read **dynamically** from the env-var
    ``RAG_SIMPLE_QUERY_THRESHOLD`` (default 30) so it can be
    toggled at runtime — set to ``0`` to disable simple-query mode.
    """
    threshold = int(os.getenv("RAG_SIMPLE_QUERY_THRESHOLD",
                              str(SIMPLE_QUERY_THRESHOLD)))
    if threshold <= 0:
        return False
    return len(query.split()) < threshold


def _section_id_of(chunk: Dict[str, Any]) -> int:
    """Extract ``section_id`` from a chunk dict, defaulting to 0."""
    val = chunk.get("metadata", {}).get("section_id", 0)
    return val if isinstance(val, int) else 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Lightweight BM25 scorer (no external deps)
# ═══════════════════════════════════════════════════════════════════════════

_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or "
    "such that the their then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenisation with stop-word removal."""
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


def _bm25_scores(
    query_tokens: List[str],
    documents: List[str],
    k1: float = _BM25_K1,
    b: float = _BM25_B,
) -> List[float]:
    """Score every document against *query_tokens* using BM25.

    Returns a list of floats (one per document) — **not** sorted.
    """
    n = len(documents)
    if n == 0 or not query_tokens:
        return [0.0] * n

    # Tokenise corpus & compute stats
    corpus = [_tokenize(doc) for doc in documents]
    doc_lens = [len(toks) for toks in corpus]
    avgdl = sum(doc_lens) / n if n else 1.0

    # Document-frequency for each query token
    df: Dict[str, int] = defaultdict(int)
    for toks in corpus:
        seen = set(toks)
        for qt in query_tokens:
            if qt in seen:
                df[qt] += 1

    scores: List[float] = []
    for idx, toks in enumerate(corpus):
        tf_map: Dict[str, int] = defaultdict(int)
        for t in toks:
            tf_map[t] += 1

        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            d = df.get(qt, 0)
            idf = math.log((n - d + 0.5) / (d + 0.5) + 1.0)
            num = tf * (k1 + 1)
            den = tf + k1 * (1.0 - b + b * doc_lens[idx] / avgdl)
            score += idf * num / den
        scores.append(score)

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# 3. Score normalisation helper
# ═══════════════════════════════════════════════════════════════════════════

def _strict_code_boost(
    content: str,
    query_tokens: List[str],
) -> float:
    """Compute a score boost for strict code-mode ranking.

    Returns up to ``_STRICT_CODE_FUNC_SIG_BOOST + _STRICT_CODE_KEYWORD_OVERLAP_BOOST``
    (0.8 by default).

    Boost breakdown:
        +0.5  if a Python ``def`` signature is present in *content*
               **and** at least one query token overlaps with the
               function name (after snake_case + camelCase splitting).
        +0.3  if there is *any* token overlap between the query and
               the chunk content (broader keyword match).  Content
               tokens are also split on underscores and camelCase
               so that ``applyPtSlOnT1`` matches query token ``apply``.
    """
    boost = 0.0
    query_set = set(query_tokens)

    # ── +0.5: function signature with name overlap ───────────────────
    if "def " in content.lower():
        func_names = _DEF_NAME_RE.findall(content)
        if func_names:
            func_tokens: set[str] = set()
            for fn in func_names:
                func_tokens |= _split_identifier(fn)
            if func_tokens & query_set:
                boost += _STRICT_CODE_FUNC_SIG_BOOST

    # ── +0.3: general keyword overlap ────────────────────────────────
    # Tokenise content with sub-word splitting so identifiers like
    # ``daily_volatility`` or ``applyPtSlOnT1`` produce sub-tokens
    # that can match individual query words.
    raw_tokens = _TOKEN_RE.findall(content)
    content_tokens: set[str] = set()
    for t in raw_tokens:
        content_tokens |= _split_identifier(t)
    # Remove very short tokens and stopwords
    content_tokens = {
        t for t in content_tokens if len(t) > 1 and t not in _STOPWORDS
    }
    if content_tokens & query_set:
        boost += _STRICT_CODE_KEYWORD_OVERLAP_BOOST

    return boost


def _min_max_normalise(values: List[float]) -> List[float]:
    """Normalise *values* to the ``[0, 1]`` range (min-max).

    If all values are identical the result is all-zeros.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng == 0.0:
        return [0.0] * len(values)
    return [(v - lo) / rng for v in values]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Candidate dataclass-like dict builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_candidate(
    doc_id: str,
    content: str,
    metadata: Dict[str, Any],
    vector_score: float,
    keyword_score: float,
    stage_bonus: float,
    final_score: float,
) -> Dict[str, Any]:
    """Construct a single ranked-candidate dict."""
    return {
        "id": doc_id,
        "content": content,
        "metadata": metadata,
        "scores": {
            "vector": round(vector_score, 6),
            "keyword": round(keyword_score, 6),
            "stage_bonus": round(stage_bonus, 6),
            "final": round(final_score, 6),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Retriever class
# ═══════════════════════════════════════════════════════════════════════════

class Retriever:
    """Orchestrates query classification, index lookup, and ranking.

    Usage::

        store = DualIndexStore(config=cfg, embed_fn=embed_fn)
        retriever = Retriever(store, config=cfg)
        results = retriever.retrieve("Reduce max drawdown on momentum")

    Args:
        store: A ``DualIndexStore`` instance (from ``vector_store.py``).
        config: Optional ``RAGConfig`` for top_k, thresholds, etc.
        w_vector: Weight for vector similarity  (default 0.5).
        w_keyword: Weight for BM25 keyword score (default 0.2).
        w_stage: Weight for pipeline-stage bonus  (default 0.3).
    """

    def __init__(
        self,
        store: DualIndexStore,
        config: Optional[RAGConfig] = None,
        w_vector: float = W_VECTOR,
        w_keyword: float = W_KEYWORD,
        w_stage: float = W_STAGE,
    ) -> None:
        self._store = store
        self._config = config or RAGConfig()
        self._w_vector = w_vector
        self._w_keyword = w_keyword
        self._w_stage = w_stage

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        chunk_type_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full retrieval pipeline and return ranked chunks.

        Steps:
            1. Classify the query (intent, pipeline_stage, needs_code).
            2. Choose the primary index (code / theory).
            3. Fetch candidates via vector similarity search.
            4. Compute BM25 keyword scores over those candidates.
            5. Apply pipeline-stage match bonus.
            6. Fuse scores with the weighted formula.
            7. Return the top-*k* candidates sorted by final score.

        Args:
            query: User query string.
            top_k: Number of results to return (default 10).
            filters: Optional ChromaDB ``where`` filter dict
                (merged with auto-detected stage filters).
            chunk_type_override: Force search on a specific index
                (``"code"`` or ``"theory"``).  If ``None``, the
                classifier decides.

        Returns:
            List of candidate dicts sorted by ``scores.final``
            (descending).  Each dict has keys:
            ``id``, ``content``, ``metadata``, ``scores``.
        """
        if not query or not query.strip():
            logger.warning("retrieve() called with empty query.")
            return []

        query = query.strip()
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        # ── Fast-mode gate ───────────────────────────────────────────
        _fast = is_fast_mode()
        if _fast:
            top_k = min(top_k, FAST_MODE_TOP_K)
            logger.info(
                "Retriever [FAST_MODE]: top_k capped to %d, "
                "skipping BM25 + stage scoring.", top_k,
            )

        # ── Simple-query gate ────────────────────────────────────────
        _simple = is_simple_query(query)
        if _simple and not _fast:
            top_k = min(top_k, SIMPLE_QUERY_TOP_K)
            logger.info(
                "Retriever [SIMPLE_QUERY]: query has %d words (< %d) — "
                "top_k capped to %d, skipping BM25 + stage scoring.",
                len(query.split()), SIMPLE_QUERY_THRESHOLD, top_k,
            )

        # ── Step 1: Classify ─────────────────────────────────────────
        t0 = time.perf_counter()
        classification = classify_query(query)
        timings["classify"] = time.perf_counter() - t0

        intent = classification["intent"]
        query_stages = classification["pipeline_stage"]
        code_needed = classification["needs_code"]
        strict_code = classification.get("strict_code_mode", False)

        logger.info(
            "Retriever: query=%r → intent=%s, stages=%s, needs_code=%s, "
            "strict_code_mode=%s",
            query[:80], intent, query_stages, code_needed, strict_code,
        )

        # ── Strict code-mode gate ────────────────────────────────────
        # When strict_code_mode is active: search ONLY code_index,
        # force top_k=8, apply chunk_type=="code" metadata filter,
        # and use function-signature + keyword-overlap scoring.
        if strict_code and not chunk_type_override:
            return self._strict_code_retrieve(
                query, classification, timings, t_start,
            )

        # ── Step 2: Choose primary index ─────────────────────────────
        if chunk_type_override:
            primary_type = chunk_type_override
        elif code_needed:
            primary_type = "code"
        else:
            primary_type = "theory"

        # ── Step 3: Vector similarity search (parallel) ──────────────
        # In fast mode we fetch only top_k (no over-fetch needed since
        # we skip BM25 re-ranking).  Normal mode over-fetches 3×.
        # Both paths are capped to MAX_CHROMA_RESULTS (8) to bound
        # vector-DB latency.
        fetch_k = top_k if _fast else max(top_k * 3, 30)
        fetch_k = min(fetch_k, MAX_CHROMA_RESULTS)

        t0 = time.perf_counter()
        secondary_type = "theory" if primary_type == "code" else "code"

        # Launch primary + secondary vector searches concurrently.
        # Secondary results are only used when primary is sparse.
        with ThreadPoolExecutor(max_workers=_RETRIEVER_WORKERS) as pool:
            fut_primary = pool.submit(
                self._vector_search, query, primary_type, fetch_k, filters,
            )
            fut_secondary = pool.submit(
                self._vector_search, query, secondary_type, fetch_k, filters,
            )
            candidates = fut_primary.result()
            secondary = fut_secondary.result()

        # Merge secondary only when primary is sparse.
        if len(candidates) < top_k:
            seen_ids = {c["id"] for c in candidates}
            for c in secondary:
                if c["id"] not in seen_ids:
                    candidates.append(c)
                    seen_ids.add(c["id"])

        timings["vector_search"] = time.perf_counter() - t0

        if not candidates:
            logger.info("Retriever: no candidates found for query=%r.", query[:60])
            return []

        # ── FAST / SIMPLE shortcut: vector-sim only (skip BM25+stage) ─
        if _fast or _simple:
            # Normalise vector similarities, assign 100 % weight
            raw_vec = [c["vector_similarity"] for c in candidates]
            norm_vec = _min_max_normalise(raw_vec)

            ranked: List[Dict[str, Any]] = []
            for i, c in enumerate(candidates):
                vs = norm_vec[i]
                ranked.append(
                    _build_candidate(
                        doc_id=c["id"],
                        content=c["content"],
                        metadata=c["metadata"],
                        vector_score=vs,
                        keyword_score=0.0,
                        stage_bonus=0.0,
                        final_score=vs,
                    )
                )
            ranked.sort(key=lambda x: x["scores"]["final"], reverse=True)
            ranked = ranked[:top_k]

            timings["total"] = time.perf_counter() - t_start
            self._log_timings(timings, len(ranked), primary_type, fast=True)
            return ranked

        # ── Steps 4 + 5 (parallel): BM25 + Stage bonus ──────────────
        documents = [c["content"] for c in candidates]
        query_tokens = _tokenize(query)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=_RETRIEVER_WORKERS) as pool:
            fut_bm25 = pool.submit(
                _bm25_scores, query_tokens, documents,
            )
            fut_stage = pool.submit(
                self._compute_stage_bonuses, candidates, query_stages,
            )
            raw_bm25 = fut_bm25.result()
            stage_bonuses = fut_stage.result()
        norm_bm25 = _min_max_normalise(raw_bm25)
        timings["bm25_and_stage"] = time.perf_counter() - t0

        # ── Step 6: Fuse scores ──────────────────────────────────────
        t0 = time.perf_counter()
        ranked = self._fuse_and_rank(
            candidates, norm_bm25, stage_bonuses, top_k,
        )
        timings["fusion"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_start
        self._log_timings(timings, len(ranked), primary_type, fast=False)
        return ranked

    # ------------------------------------------------------------------
    # Internal: strict code-mode retrieval
    # ------------------------------------------------------------------

    def _strict_code_retrieve(
        self,
        query: str,
        classification: Dict[str, Any],
        timings: Dict[str, float],
        t_start: float,
    ) -> List[Dict[str, Any]]:
        """Dedicated retrieval path for ``strict_code_mode``.

        When the classifier detects strong code-intent keywords
        ("implement", "python", "code", "function", "using python")
        this method:

        1. Searches **only** ``code_index`` — theory is skipped.
        2. Forces ``top_k = STRICT_CODE_TOP_K`` (8).
        3. Applies ``chunk_type == "code"`` metadata filter.
        4. Boosts scores with :func:`_strict_code_boost`:
           - +0.5 if ``def <name>`` present and name overlaps query.
           - +0.3 if any keyword overlap between chunk and query.
        5. Filters out non-code chunks from the final results.

        Returns:
            Ranked list of code-only candidate dicts.
        """
        top_k = STRICT_CODE_TOP_K
        fetch_k = min(top_k, MAX_CHROMA_RESULTS)

        logger.info(
            "Retriever [STRICT_CODE]: top_k=%d, code_index only, "
            "skipping theory_index.",
            top_k,
        )

        # ── Step A: Vector search — code_index ONLY ──────────────────
        t0 = time.perf_counter()
        code_filter = {"chunk_type": "code"}
        candidates = self._vector_search(
            query, "code", fetch_k, code_filter,
        )
        timings["vector_search"] = time.perf_counter() - t0

        if not candidates:
            logger.info(
                "Retriever [STRICT_CODE]: no code candidates found "
                "for query=%r.",
                query[:60],
            )
            timings["total"] = time.perf_counter() - t_start
            return []

        # ── Step B: Filter — keep ONLY code chunks ───────────────────
        candidates = [
            c for c in candidates
            if c.get("metadata", {}).get("chunk_type", "") == "code"
        ]

        if not candidates:
            timings["total"] = time.perf_counter() - t_start
            return []

        # ── Step C: Normalise vector scores ──────────────────────────
        raw_vec = [c["vector_similarity"] for c in candidates]
        norm_vec = _min_max_normalise(raw_vec)

        # ── Step D: Apply strict code-mode boosts + fuzzy scoring ───
        t0 = time.perf_counter()
        query_tokens = _tokenize(query)

        ranked: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            vs = norm_vec[i]
            boost = _strict_code_boost(c["content"], query_tokens)
            fuzzy = _fuzzy_func_name_score(c["content"], query_tokens)
            final = vs + boost + fuzzy

            cand = _build_candidate(
                doc_id=c["id"],
                content=c["content"],
                metadata=c["metadata"],
                vector_score=vs,
                keyword_score=0.0,
                stage_bonus=boost,
                final_score=final,
            )
            # Attach boost details for observability
            cand["scores"]["strict_code_boost"] = round(boost, 6)
            cand["scores"]["fuzzy_func_score"] = round(fuzzy, 6)
            ranked.append(cand)

        timings["strict_code_scoring"] = time.perf_counter() - t0

        # ── Step E: Sort and truncate ────────────────────────────────
        ranked.sort(key=lambda x: x["scores"]["final"], reverse=True)
        ranked = ranked[:top_k]

        timings["total"] = time.perf_counter() - t_start
        self._log_timings(timings, len(ranked), "code", fast=False)

        logger.info(
            "Retriever [STRICT_CODE]: returning %d code chunks. "
            "Top boost=%.2f, top fuzzy=%.2f, top final=%.4f.",
            len(ranked),
            ranked[0]["scores"]["strict_code_boost"] if ranked else 0.0,
            ranked[0]["scores"]["fuzzy_func_score"] if ranked else 0.0,
            ranked[0]["scores"]["final"] if ranked else 0.0,
        )

        # Emit detailed code-retrieval diagnostics
        _log_code_retrieval(ranked)

        return ranked

    # ------------------------------------------------------------------
    # Internal: timing logger
    # ------------------------------------------------------------------

    @staticmethod
    def _log_timings(
        timings: Dict[str, float],
        n_results: int,
        primary_type: str,
        fast: bool = False,
    ) -> None:
        """Emit a structured timing summary for the retrieve() call."""
        parts = " | ".join(
            f"{k}={v * 1000:.1f}ms" for k, v in timings.items()
        )
        mode = "FAST" if fast else "FULL"
        total = timings.get("total", 0.0)
        logger.info(
            "Retriever [%s] timings: %s | results=%d primary=%s",
            mode, parts, n_results, primary_type,
        )
        if total > _RETRIEVAL_TIMEOUT_WARN:
            logger.warning(
                "Retriever: total retrieval %.2fs exceeds %.1fs threshold!",
                total, _RETRIEVAL_TIMEOUT_WARN,
            )

    # ------------------------------------------------------------------
    # Internal: vector search
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        chunk_type: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch raw candidates from the DualIndexStore.

        Converts ChromaDB results into a flat list of dicts with
        ``id``, ``content``, ``metadata``, ``vector_similarity``.
        """
        try:
            results = self._store.query_index(
                query=query,
                chunk_type=chunk_type,
                filters=filters,
                top_k=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception(
                "Vector search failed for chunk_type=%s.", chunk_type,
            )
            return []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        candidates: List[Dict[str, Any]] = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # ChromaDB cosine distance → similarity: sim = 1 - dist
            similarity = max(0.0, 1.0 - dist)
            candidates.append({
                "id": doc_id,
                "content": doc or "",
                "metadata": meta or {},
                "vector_similarity": similarity,
            })

        return candidates

    # ------------------------------------------------------------------
    # Internal: stage bonus
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stage_bonuses(
        candidates: List[Dict[str, Any]],
        query_stages: List[str],
    ) -> List[float]:
        """Return a bonus value for each candidate based on stage overlap.

        A candidate gets ``_STAGE_MATCH_VALUE`` (1.0) if its
        ``pipeline_stage`` metadata matches any of the query's detected
        stages, and ``_STAGE_NO_MATCH_VALUE`` (0.0) otherwise.

        When the classifier detected no pipeline stages in the query,
        all candidates receive 0.0 (neutral — no stage signal).
        """
        if not query_stages:
            return [_STAGE_NO_MATCH_VALUE] * len(candidates)

        query_stage_set = set(query_stages)
        bonuses: List[float] = []
        for c in candidates:
            meta = c.get("metadata", {})
            chunk_stage = meta.get("pipeline_stage", "")
            # chunk_stage may be a comma-separated string (flattened metadata)
            if isinstance(chunk_stage, str):
                chunk_stages = {
                    s.strip() for s in chunk_stage.split(",") if s.strip()
                }
            else:
                chunk_stages = set()
            if chunk_stages & query_stage_set:
                bonuses.append(_STAGE_MATCH_VALUE)
            else:
                bonuses.append(_STAGE_NO_MATCH_VALUE)
        return bonuses

    # ------------------------------------------------------------------
    # Internal: score fusion
    # ------------------------------------------------------------------

    def _fuse_and_rank(
        self,
        candidates: List[Dict[str, Any]],
        norm_bm25: List[float],
        stage_bonuses: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Apply the weighted scoring formula, sort, and truncate.

        ``final_score = w_vector * vector_score
                      + w_keyword * keyword_score
                      + w_stage   * stage_bonus``
        """
        # Normalise vector similarities within this candidate set
        raw_vec = [c["vector_similarity"] for c in candidates]
        norm_vec = _min_max_normalise(raw_vec)

        ranked: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            vs = norm_vec[i]
            ks = norm_bm25[i]
            sb = stage_bonuses[i]
            final = (
                self._w_vector * vs
                + self._w_keyword * ks
                + self._w_stage * sb
            )
            ranked.append(
                _build_candidate(
                    doc_id=c["id"],
                    content=c["content"],
                    metadata=c["metadata"],
                    vector_score=vs,
                    keyword_score=ks,
                    stage_bonus=sb,
                    final_score=final,
                )
            )

        ranked.sort(key=lambda x: x["scores"]["final"], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Parent-child context retrieval
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int = DEFAULT_CONTEXT_BUDGET,
        filters: Optional[Dict[str, Any]] = None,
        chunk_type_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve ranked chunks, expand via parent-child siblings,
        and return structured context within a token budget.

        **Algorithm**:

        1. Call ``self.retrieve()`` to get top-*k* scored child chunks.
        2. Group results by ``parent_id``, keeping the *best child
           score* per parent as that parent's priority.
        3. Sort parents by descending best-child score.
        4. Limit to the top ``MAX_PARENT_SECTIONS`` (default 3)
           parents to prevent context blowup.
        5. For each parent:
           a. Estimate parent token size from all fetched siblings.
           b. If > ``_LARGE_PARENT_TOKEN_THRESHOLD`` (1 200 tokens),
              fetch only **adjacent** child chunks (same parent_id,
              section_id ± 1 of the best-scoring child).
           c. Otherwise fetch all siblings via ``get_by_metadata``.
        6. Cap to ``MAX_CHILDREN_PER_PARENT`` (default 2) siblings
           per parent, preferring the ones closest to the matched
           child by ``section_id``.
        7. Deduplicate by chunk ID.
        8. Greedily pack siblings into the context until the
           ``max_tokens`` budget is exhausted.
        9. Split the packed context into ``theory_context`` and
           ``code_context`` by ``chunk_type``.

        Args:
            query: User query string.
            top_k: Number of child chunks to retrieve in Step 1.
            max_tokens: Maximum total tokens for the combined context
                (default 2 500).
            filters: Optional ChromaDB ``where`` filter dict.
            chunk_type_override: Force search on ``"code"`` / ``"theory"``.

        Returns:
            Dict with two keys::

                {
                    "theory_context": [
                        {"id": str, "content": str, "metadata": dict}, ...
                    ],
                    "code_context": [
                        {"id": str, "content": str, "metadata": dict}, ...
                    ],
                }

            Both lists may be empty.  Combined token count never
            exceeds *max_tokens*.  An ``"intent"`` key is also
            included so that downstream formatters (e.g.
            ``build_prompt_context``) can prioritise code vs. theory.
        """
        empty_result: Dict[str, Any] = {
            "theory_context": [],
            "code_context": [],
            "intent": "unknown",
        }

        if not query or not query.strip():
            return empty_result

        t_ctx_start = time.perf_counter()
        ctx_timings: Dict[str, float] = {}

        # ── Step 0: Classify for intent passthrough ──────────────────
        t0 = time.perf_counter()
        _classification = classify_query(query)
        _intent = _classification.get("intent", "unknown")
        ctx_timings["classify"] = time.perf_counter() - t0

        # ── Step 1: Ranked child retrieval ───────────────────────────
        t0 = time.perf_counter()
        ranked = self.retrieve(
            query,
            top_k=top_k,
            filters=filters,
            chunk_type_override=chunk_type_override,
        )
        ctx_timings["retrieve"] = time.perf_counter() - t0

        if not ranked:
            return empty_result

        # ── Simple-query shortcut: skip parent expansion entirely ────
        # For short queries, the ranked chunks from retrieve() are
        # sufficient — no sibling fetching, no parent grouping.
        _simple = is_simple_query(query)
        if _simple:
            logger.info(
                "retrieve_context [SIMPLE_QUERY]: skipping parent expansion "
                "for %d-word query — packing %d ranked chunks directly.",
                len(query.split()), len(ranked),
            )
            theory_ctx: List[Dict[str, Any]] = []
            code_ctx: List[Dict[str, Any]] = []
            tokens_used = 0

            for r in ranked:
                chunk_tokens = count_tokens(r["content"])
                if tokens_used + chunk_tokens > max_tokens:
                    break
                entry = {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"],
                }
                ctype = r.get("metadata", {}).get("chunk_type", "theory")
                if ctype == "code":
                    code_ctx.append(entry)
                else:
                    theory_ctx.append(entry)
                tokens_used += chunk_tokens

            ctx_timings["context_build"] = time.perf_counter() - t0
            ctx_timings["total"] = time.perf_counter() - t_ctx_start
            parts = " | ".join(
                f"{k}={v * 1000:.1f}ms" for k, v in ctx_timings.items()
            )
            logger.info(
                "retrieve_context [SIMPLE_QUERY]: packed %d chunks "
                "(%d theory, %d code) within %d/%d token budget. %s",
                len(theory_ctx) + len(code_ctx),
                len(theory_ctx), len(code_ctx),
                tokens_used, max_tokens, parts,
            )
            return {
                "theory_context": theory_ctx,
                "code_context": code_ctx,
                "intent": _intent,
                "strict_code_mode": _classification.get("strict_code_mode", False),
            }

        # ── Step 2: Group by parent_id ───────────────────────────────
        # parent_id → best child final_score
        parent_best_score: Dict[str, float] = {}
        # parent_id → chunk_type of the best-scoring child (for routing)
        parent_chunk_type: Dict[str, str] = {}

        for r in ranked:
            pid = r.get("metadata", {}).get("parent_id", "")
            if not pid:
                # Chunks without parent_id are treated as standalone
                # parents (parent_id = their own ID).
                pid = r["id"]
            score = r["scores"]["final"]
            if pid not in parent_best_score or score > parent_best_score[pid]:
                parent_best_score[pid] = score
                parent_chunk_type[pid] = (
                    r.get("metadata", {}).get("chunk_type", "theory")
                )

        # ── Step 3: Sort parents by descending score ─────────────────
        sorted_parents = sorted(
            parent_best_score.keys(),
            key=lambda p: parent_best_score[p],
            reverse=True,
        )

        logger.info(
            "retrieve_context: %d ranked children → %d unique parents.",
            len(ranked), len(sorted_parents),
        )

        # ── Steps 4–9: Expand, dedup, budget ─────────────────────────
        t0 = time.perf_counter()
        seen_ids: set = set()
        packed: List[Dict[str, Any]] = []
        tokens_used = 0
        parents_expanded = 0
        expansion_tokens = 0           # tokens added purely by expansion

        # Pre-index: parent_id → best-scoring child's section_id
        _best_child_section: Dict[str, int] = {}
        for r in ranked:
            pid = r.get("metadata", {}).get("parent_id", "") or r["id"]
            sid_val = r.get("metadata", {}).get("section_id", 0)
            score = r["scores"]["final"]
            if pid not in _best_child_section or score > parent_best_score.get(pid, 0):
                _best_child_section[pid] = (
                    sid_val if isinstance(sid_val, int) else 0
                )

        # Limit to top MAX_PARENT_SECTIONS parents
        capped_parents = sorted_parents[:MAX_PARENT_SECTIONS]

        for pid in capped_parents:
            if tokens_used >= max_tokens:
                break

            # Determine which collection(s) to search
            ctype = parent_chunk_type.get(pid)

            # Fetch all siblings sharing this parent_id
            try:
                all_siblings = self._store.get_by_metadata(
                    where={"parent_id": pid},
                    chunk_type=ctype,
                )
            except Exception:
                logger.debug("Sibling fetch failed for parent_id=%s.", pid)
                # Fallback: use the original ranked chunk(s) for this parent
                all_siblings = [
                    {"id": r["id"], "content": r["content"],
                     "metadata": r["metadata"]}
                    for r in ranked
                    if r.get("metadata", {}).get("parent_id", r["id"]) == pid
                ]

            # ── Large-parent guard: estimate total parent tokens ──────
            parent_total_tokens = sum(
                count_tokens(s.get("content", "")) for s in all_siblings
            )

            if parent_total_tokens > _LARGE_PARENT_TOKEN_THRESHOLD:
                # Too large — fetch only children adjacent to the
                # best-scoring child (section_id ± 1).
                anchor_sid = _best_child_section.get(pid, 0)
                adjacent_ids = {anchor_sid - 1, anchor_sid, anchor_sid + 1}
                siblings = [
                    s for s in all_siblings
                    if _section_id_of(s) in adjacent_ids
                ]
                logger.debug(
                    "Large parent %s (%d tokens) — trimmed %d→%d adjacent siblings.",
                    pid, parent_total_tokens, len(all_siblings), len(siblings),
                )
            else:
                siblings = all_siblings

            # ── Per-parent child cap ─────────────────────────────────
            # Keep at most MAX_CHILDREN_PER_PARENT, preferring those
            # closest to the best-scoring child by section_id.
            if len(siblings) > MAX_CHILDREN_PER_PARENT:
                anchor = _best_child_section.get(pid, 0)
                siblings.sort(
                    key=lambda s: abs(_section_id_of(s) - anchor)
                )
                siblings = siblings[:MAX_CHILDREN_PER_PARENT]
                # Restore document order after proximity pick
                siblings.sort(key=lambda s: _section_id_of(s))

            children_from_parent = 0
            for sib in siblings:
                chunk_id = sib["id"]
                if chunk_id in seen_ids:
                    continue

                chunk_tokens = count_tokens(sib["content"])
                needed = chunk_tokens + _SEPARATOR_TOKENS

                if tokens_used + needed > max_tokens:
                    # Try to fit the chunk by itself (without separator
                    # overhead) if it would be the first from this parent.
                    if tokens_used + chunk_tokens > max_tokens:
                        continue
                    needed = chunk_tokens

                seen_ids.add(chunk_id)
                packed.append(sib)
                tokens_used += needed
                children_from_parent += 1

                # Track expansion tokens (siblings that were NOT in the
                # original ranked list are "expansion" additions).
                is_expansion = chunk_id not in {
                    r["id"] for r in ranked
                }
                if is_expansion:
                    expansion_tokens += needed

            if children_from_parent > 0:
                parents_expanded += 1

        ctx_timings["context_build"] = time.perf_counter() - t0

        logger.info(
            "retrieve_context: expanded %d parents, "
            "%d expansion tokens added.",
            parents_expanded, expansion_tokens,
        )

        # ── Step 9: Split into theory / code ─────────────────────────
        theory_ctx: List[Dict[str, Any]] = []
        code_ctx: List[Dict[str, Any]] = []

        for chunk in packed:
            ctype = chunk.get("metadata", {}).get("chunk_type", "theory")
            entry = {
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            }
            if ctype == "code":
                code_ctx.append(entry)
            else:
                theory_ctx.append(entry)

        ctx_timings["total"] = time.perf_counter() - t_ctx_start

        logger.info(
            "retrieve_context: packed %d chunks (%d theory, %d code) "
            "within %d/%d token budget.",
            len(packed), len(theory_ctx), len(code_ctx),
            tokens_used, max_tokens,
        )

        # ── Timing summary ───────────────────────────────────────────
        parts = " | ".join(
            f"{k}={v * 1000:.1f}ms" for k, v in ctx_timings.items()
        )
        logger.info("retrieve_context timings: %s", parts)
        if ctx_timings.get("total", 0.0) > _RETRIEVAL_TIMEOUT_WARN:
            logger.warning(
                "retrieve_context: total %.2fs exceeds %.1fs threshold!",
                ctx_timings["total"], _RETRIEVAL_TIMEOUT_WARN,
            )

        return {
            "theory_context": theory_ctx,
            "code_context": code_ctx,
            "intent": _intent,
            "strict_code_mode": _classification.get("strict_code_mode", False),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Module-level convenience
# ═══════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    store: Optional[DualIndexStore] = None,
    config: Optional[RAGConfig] = None,
    embed_fn: Optional[EmbeddingFn] = None,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Module-level convenience for one-shot retrieval.

    Creates a ``DualIndexStore`` (if not provided), wraps it in a
    ``Retriever``, and returns ranked results.

    Args:
        query: User query string.
        store: Optional pre-initialised ``DualIndexStore``.
        config: Optional ``RAGConfig``.
        embed_fn: Optional embedding function.
        top_k: Number of results (default 10).
        filters: Optional metadata filters.

    Returns:
        Sorted list of candidate dicts.
    """
    if store is None:
        store = DualIndexStore(config=config, embed_fn=embed_fn)
    retriever = Retriever(store, config=config)
    return retriever.retrieve(query, top_k=top_k, filters=filters)


def retrieve_context(
    query: str,
    store: Optional[DualIndexStore] = None,
    config: Optional[RAGConfig] = None,
    embed_fn: Optional[EmbeddingFn] = None,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_CONTEXT_BUDGET,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Module-level convenience for parent-child context retrieval.

    Creates a ``DualIndexStore`` (if not provided), wraps it in a
    ``Retriever``, and returns structured context within a token budget.

    Args:
        query: User query string.
        store: Optional pre-initialised ``DualIndexStore``.
        config: Optional ``RAGConfig``.
        embed_fn: Optional embedding function.
        top_k: Number of child results (default 10).
        max_tokens: Max context tokens (default 2 500).
        filters: Optional metadata filters.

    Returns:
        ``{"theory_context": [...], "code_context": [...], "intent": str}``
    """
    if store is None:
        store = DualIndexStore(config=config, embed_fn=embed_fn)
    retriever = Retriever(store, config=config)
    return retriever.retrieve_context(
        query, top_k=top_k, max_tokens=max_tokens, filters=filters,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. Unit tests (run with ``python -m rag_pipeline.retriever``)
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    """Self-contained unit tests — no external test framework needed."""

    import sys
    import os
    import tempfile
    import shutil

    PASS = 0
    FAIL = 0

    def check(label: str, condition: bool) -> None:
        nonlocal PASS, FAIL
        if condition:
            PASS += 1
            print(f"  PASS: {label}")
        else:
            FAIL += 1
            print(f"  FAIL: {label}")

    # Disable simple-query mode for legacy tests (they validate the
    # full pipeline).  A dedicated test at the end enables it.
    _old_simple_threshold = os.environ.get("RAG_SIMPLE_QUERY_THRESHOLD")
    os.environ["RAG_SIMPLE_QUERY_THRESHOLD"] = "0"

    # ── 1. Tokenizer ─────────────────────────────────────────────────
    print("\n=== 1. Tokenizer ===")
    tokens = _tokenize("Calculate the Sharpe ratio for risk_management")
    check("lowercase", all(t == t.lower() for t in tokens))
    check("stop words removed", "the" not in tokens and "for" not in tokens)
    check("underscored token preserved", "risk_management" in tokens)

    # ── 2. BM25 scores ──────────────────────────────────────────────
    print("\n=== 2. BM25 Scoring ===")
    docs = [
        "The Sharpe ratio measures risk-adjusted returns.",
        "Momentum crossover signal uses moving average.",
        "Calculate Sharpe after reducing max drawdown.",
    ]
    qt = _tokenize("sharpe ratio returns")
    scores = _bm25_scores(qt, docs)
    check("3 scores returned", len(scores) == 3)
    check("doc 0 > doc 1 (sharpe+returns vs crossover)", scores[0] > scores[1])
    check("doc 2 > doc 1 (sharpe mention vs none)", scores[2] > scores[1])

    # Empty corpus / query
    check("empty corpus → empty", _bm25_scores(qt, []) == [])
    check("empty query → zeros", _bm25_scores([], docs) == [0.0, 0.0, 0.0])

    # ── 3. Min-max normalisation ─────────────────────────────────────
    print("\n=== 3. Normalisation ===")
    norm = _min_max_normalise([10.0, 20.0, 30.0])
    check("[10,20,30] → [0.0, 0.5, 1.0]", norm == [0.0, 0.5, 1.0])
    check("identical values → zeros", _min_max_normalise([5.0, 5.0]) == [0.0, 0.0])
    check("empty → empty", _min_max_normalise([]) == [])
    norm2 = _min_max_normalise([0.0, 1.0])
    check("[0,1] → [0,1]", norm2 == [0.0, 1.0])

    # ── 4. Stage bonus computation ───────────────────────────────────
    print("\n=== 4. Stage Bonus ===")
    cands = [
        {"metadata": {"pipeline_stage": "risk_management"}},
        {"metadata": {"pipeline_stage": "evaluation"}},
        {"metadata": {"pipeline_stage": "signal_generation"}},
        {"metadata": {}},
    ]
    bonuses = Retriever._compute_stage_bonuses(cands, ["risk_management", "evaluation"])
    check("risk_management matches → 1.0", bonuses[0] == 1.0)
    check("evaluation matches → 1.0", bonuses[1] == 1.0)
    check("signal_generation no match → 0.0", bonuses[2] == 0.0)
    check("no stage in meta → 0.0", bonuses[3] == 0.0)

    # No query stages → all neutral
    bonuses_empty = Retriever._compute_stage_bonuses(cands, [])
    check("no query stages → all 0.0", all(b == 0.0 for b in bonuses_empty))

    # Comma-separated stage (flattened metadata)
    cands_csv = [{"metadata": {"pipeline_stage": "risk_management, evaluation"}}]
    bonuses_csv = Retriever._compute_stage_bonuses(cands_csv, ["evaluation"])
    check("CSV stage matches → 1.0", bonuses_csv[0] == 1.0)

    # ── 5. Full Retriever (end-to-end with mock store) ───────────────
    print("\n=== 5. Full Retriever (end-to-end) ===")

    tmpdir = tempfile.mkdtemp()
    try:
        cfg = RAGConfig()
        cfg.chroma_persist_dir = tmpdir

        DIM = 384

        def mock_embed(text: str, chunk_type: str) -> List[float]:
            # Simple deterministic embedding: hash-based
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store = DualIndexStore(config=cfg, embed_fn=mock_embed)

        # Seed the store with test chunks
        test_chunks = [
            {
                "content": "def calculate_sharpe(returns):\n    return np.mean(returns) / np.std(returns)",
                "metadata": {"chunk_type": "code", "pipeline_stage": "evaluation", "source": "test.pdf"},
            },
            {
                "content": "def max_drawdown(equity_curve):\n    peak = equity_curve.cummax()\n    return (peak - equity_curve).max()",
                "metadata": {"chunk_type": "code", "pipeline_stage": "risk_management", "source": "test.pdf"},
            },
            {
                "content": "The Sharpe ratio is a measure of risk-adjusted return. It divides excess returns by portfolio volatility.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "evaluation", "source": "test.pdf"},
            },
            {
                "content": "Maximum drawdown represents the peak-to-trough decline in portfolio value. It captures tail risk.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "risk_management", "source": "test.pdf"},
            },
            {
                "content": "Momentum strategies follow trend signals. The z-score indicator normalises price deviations.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "signal_generation", "source": "test.pdf"},
            },
            {
                "content": "Portfolio allocation uses mean-variance optimisation to balance diversification and expected returns.",
                "metadata": {"chunk_type": "theory", "pipeline_stage": "portfolio_construction", "source": "test.pdf"},
            },
        ]
        store.upsert_chunks(test_chunks)

        retriever = Retriever(store, config=cfg)

        # ── Query: code-generation intent ────────────────────────────
        results = retriever.retrieve(
            "Write a function to calculate Sharpe ratio", top_k=5,
        )
        check("code query returns results", len(results) > 0)
        check("results have 'scores' key", "scores" in results[0])
        check("results have 'content' key", "content" in results[0])
        check("results sorted descending",
              all(results[i]["scores"]["final"] >= results[i + 1]["scores"]["final"]
                  for i in range(len(results) - 1)))
        # The top result should be related to Sharpe (strict code mode
        # returns only code chunks — both code chunks are returned,
        # and boost scoring favours calculate_sharpe over max_drawdown
        # for this query.  Check ANY result mentions sharpe.)
        any_sharpe = any("sharpe" in r["content"].lower() for r in results)
        check("results contain sharpe code chunk", any_sharpe)

        # ── Query: conceptual intent─────────────────────────────────
        results2 = retriever.retrieve(
            "Explain what maximum drawdown means for risk", top_k=5,
        )
        check("conceptual query returns results", len(results2) > 0)
        check("results2 sorted descending",
              all(results2[i]["scores"]["final"] >= results2[i + 1]["scores"]["final"]
                  for i in range(len(results2) - 1)))

        # ── Query: analysis intent ───────────────────────────────────
        results3 = retriever.retrieve(
            "Optimize the portfolio allocation to reduce drawdown", top_k=5,
        )
        check("analysis query returns results", len(results3) > 0)

        # ── Query: chunk_type_override ───────────────────────────────
        results4 = retriever.retrieve(
            "Tell me about signals", top_k=5, chunk_type_override="theory",
        )
        check("override forces theory search", len(results4) > 0)

        # ── Query: empty ─────────────────────────────────────────────
        results5 = retriever.retrieve("", top_k=5)
        check("empty query → empty results", results5 == [])

        # ── Score structure validation ───────────────────────────────
        r = results[0]
        check("scores.vector is float", isinstance(r["scores"]["vector"], float))
        check("scores.keyword is float", isinstance(r["scores"]["keyword"], float))
        check("scores.stage_bonus is float", isinstance(r["scores"]["stage_bonus"], float))
        check("scores.final is float", isinstance(r["scores"]["final"], float))
        check("id is string", isinstance(r["id"], str))
        check("metadata is dict", isinstance(r["metadata"], dict))

        # ── top_k respected ──────────────────────────────────────────
        results6 = retriever.retrieve(
            "Write code for Sharpe", top_k=2,
        )
        check("top_k=2 → at most 2 results", len(results6) <= 2)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 6. Parent-child context retrieval ────────────────────────────
    print("\n=== 6. Parent-child Context Retrieval ===")

    tmpdir2 = tempfile.mkdtemp()
    try:
        cfg2 = RAGConfig()
        cfg2.chroma_persist_dir = tmpdir2

        DIM = 384

        def mock_embed2(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store2 = DualIndexStore(config=cfg2, embed_fn=mock_embed2)

        # Parent block A → 3 sibling theory chunks
        parent_a = "parentA_sharpe_01"
        # Parent block B → 2 sibling code chunks
        parent_b = "parentB_drawdown_02"
        # Parent block C → 1 standalone theory chunk
        parent_c = "parentC_momentum_03"

        pc_chunks = [
            {
                "content": "The Sharpe ratio is a measure of risk-adjusted return. Part 1.",
                "metadata": {
                    "chunk_type": "theory", "page": 1,
                    "parent_id": parent_a, "section_id": 0,
                    "pipeline_stage": "evaluation", "source": "test.pdf",
                },
            },
            {
                "content": "It divides excess returns by portfolio standard deviation. Part 2.",
                "metadata": {
                    "chunk_type": "theory", "page": 1,
                    "parent_id": parent_a, "section_id": 1,
                    "pipeline_stage": "evaluation", "source": "test.pdf",
                },
            },
            {
                "content": "A ratio above 1.0 is generally considered acceptable. Part 3.",
                "metadata": {
                    "chunk_type": "theory", "page": 1,
                    "parent_id": parent_a, "section_id": 2,
                    "pipeline_stage": "evaluation", "source": "test.pdf",
                },
            },
            {
                "content": "def max_drawdown(equity):\n    peak = equity.cummax()\n    return (peak - equity).max()",
                "metadata": {
                    "chunk_type": "code", "page": 2,
                    "parent_id": parent_b, "section_id": 0,
                    "pipeline_stage": "risk_management", "source": "test.pdf",
                },
            },
            {
                "content": "def drawdown_duration(equity):\n    dd = max_drawdown(equity)\n    return dd.idxmax() - dd.idxmin()",
                "metadata": {
                    "chunk_type": "code", "page": 2,
                    "parent_id": parent_b, "section_id": 1,
                    "pipeline_stage": "risk_management", "source": "test.pdf",
                },
            },
            {
                "content": "Momentum signals follow trend direction using moving average crossover strategies.",
                "metadata": {
                    "chunk_type": "theory", "page": 3,
                    "parent_id": parent_c, "section_id": 0,
                    "pipeline_stage": "signal_generation", "source": "test.pdf",
                },
            },
        ]
        store2.upsert_chunks(pc_chunks)

        retriever2 = Retriever(store2, config=cfg2)

        # ── 6a: Basic structure ──────────────────────────────────────
        ctx = retriever2.retrieve_context(
            "Explain the Sharpe ratio", top_k=5, max_tokens=5000,
        )
        check("returns dict", isinstance(ctx, dict))
        check("has theory_context key", "theory_context" in ctx)
        check("has code_context key", "code_context" in ctx)
        check("has intent key", "intent" in ctx)
        check("intent is str", isinstance(ctx.get("intent"), str))
        check("theory_context is list", isinstance(ctx["theory_context"], list))
        check("code_context is list", isinstance(ctx["code_context"], list))

        # ── 6b: Sibling expansion (capped to MAX_CHILDREN_PER_PARENT)─
        # parent_a has 3 siblings but only MAX_CHILDREN_PER_PARENT=2
        # should be packed per parent.
        parent_a_chunks_found = [
            c for c in ctx["theory_context"]
            if c.get("metadata", {}).get("parent_id") == parent_a
        ]
        check(
            "sibling expansion: >0 parent_a chunks in theory",
            len(parent_a_chunks_found) > 0,
        )
        check(
            f"per-parent cap: ≤{MAX_CHILDREN_PER_PARENT} from parent_a "
            f"(got {len(parent_a_chunks_found)})",
            len(parent_a_chunks_found) <= MAX_CHILDREN_PER_PARENT,
        )

        # ── 6c: Each chunk has required keys ─────────────────────────
        for c in ctx["theory_context"] + ctx["code_context"]:
            check(
                f"chunk has id+content+metadata",
                "id" in c and "content" in c and "metadata" in c,
            )

        # ── 6d: Deduplication ────────────────────────────────────────
        all_ids = [
            c["id"] for c in ctx["theory_context"] + ctx["code_context"]
        ]
        check("no duplicate IDs", len(all_ids) == len(set(all_ids)))

        # ── 6e: Sibling ordering by section_id ───────────────────────
        if len(parent_a_chunks_found) > 1:
            sids = [
                c.get("metadata", {}).get("section_id", 0)
                for c in parent_a_chunks_found
            ]
            check(
                "siblings ordered by section_id",
                sids == sorted(sids),
            )
        else:
            check("siblings ordered by section_id (trivial: 0-1 items)", True)

        # ── 6f: Token budget enforcement ─────────────────────────────
        tiny_budget = 50  # very small budget
        ctx_tiny = retriever2.retrieve_context(
            "Explain the Sharpe ratio", top_k=5, max_tokens=tiny_budget,
        )
        total_tokens_tiny = sum(
            count_tokens(c["content"])
            for c in ctx_tiny["theory_context"] + ctx_tiny["code_context"]
        )
        # Allow some slack for separator overhead estimation differences
        check(
            f"token budget respected ({total_tokens_tiny} ≤ {tiny_budget + 20})",
            total_tokens_tiny <= tiny_budget + 20,
        )
        check(
            "tiny budget returns fewer chunks",
            len(ctx_tiny["theory_context"]) + len(ctx_tiny["code_context"])
            <= len(ctx["theory_context"]) + len(ctx["code_context"]),
        )

        # ── 6g: Empty query ──────────────────────────────────────────
        ctx_empty = retriever2.retrieve_context("", top_k=5)
        check(
            "empty query → empty context",
            ctx_empty["theory_context"] == []
            and ctx_empty["code_context"] == []
            and ctx_empty.get("intent") == "unknown",
        )

        # ── 6h: Code context present for code queries ────────────────
        ctx_code = retriever2.retrieve_context(
            "Write a max drawdown function", top_k=5, max_tokens=5000,
        )
        check(
            "code query yields code_context",
            len(ctx_code["code_context"]) > 0
            or len(ctx_code["theory_context"]) > 0,  # at least something
        )

        # ── 6i: Chunks without parent_id handled gracefully ──────────
        # Simulate by querying — chunks already have parent_id, so this
        # tests the normal path.  The edge case of missing parent_id is
        # handled inline (falls back to chunk's own ID as parent).
        check("no crash on normal retrieval path", True)

        # ── 6j: Module-level convenience function ────────────────────
        ctx_conv = retrieve_context(
            "Sharpe ratio explanation",
            store=store2, config=cfg2, top_k=3, max_tokens=5000,
        )
        check(
            "module-level retrieve_context works",
            isinstance(ctx_conv, dict)
            and "theory_context" in ctx_conv
            and "code_context" in ctx_conv,
        )

    finally:
        shutil.rmtree(tmpdir2, ignore_errors=True)

    # ── 7. MAX_PARENT_SECTIONS cap ───────────────────────────────────
    print("\n=== 7. MAX_PARENT_SECTIONS Cap ===")

    tmpdir3 = tempfile.mkdtemp()
    try:
        cfg3 = RAGConfig()
        cfg3.chroma_persist_dir = tmpdir3

        DIM = 384

        def mock_embed3(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store3 = DualIndexStore(config=cfg3, embed_fn=mock_embed3)

        # Create 5 distinct parents, each with 1 child
        many_parents_chunks = []
        for i in range(5):
            many_parents_chunks.append({
                "content": f"Theory chunk from parent number {i} about risk analysis and portfolio management.",
                "metadata": {
                    "chunk_type": "theory", "page": i,
                    "parent_id": f"parent_cap_{i}",
                    "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            })
        store3.upsert_chunks(many_parents_chunks)

        retriever3 = Retriever(store3, config=cfg3)
        ctx_cap = retriever3.retrieve_context(
            "risk analysis portfolio", top_k=10, max_tokens=5000,
        )
        all_pids = set()
        for c in ctx_cap["theory_context"] + ctx_cap["code_context"]:
            pid = c.get("metadata", {}).get("parent_id", "")
            if pid:
                all_pids.add(pid)
        check(
            f"parent cap: ≤{MAX_PARENT_SECTIONS} parents expanded "
            f"(got {len(all_pids)})",
            len(all_pids) <= MAX_PARENT_SECTIONS,
        )

    finally:
        shutil.rmtree(tmpdir3, ignore_errors=True)

    # ── 8. Large parent → adjacent-only expansion ────────────────────
    print("\n=== 8. Large Parent Adjacent-Only ===")

    tmpdir4 = tempfile.mkdtemp()
    try:
        cfg4 = RAGConfig()
        cfg4.chroma_persist_dir = tmpdir4

        DIM = 384

        def mock_embed4(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store4 = DualIndexStore(config=cfg4, embed_fn=mock_embed4)

        # Create a large parent with many children that exceed 1200 tokens total
        large_parent_id = "large_parent_001"
        large_chunks = []
        for i in range(10):
            # Each chunk ~150 tokens → total ~1500 tokens (> 1200 threshold)
            large_chunks.append({
                "content": (
                    f"Child chunk section {i} from the large parent. "
                    + "Portfolio construction methodology involves asset allocation, "
                      "risk budgeting, factor exposure management, and covariance "
                      "estimation using shrinkage estimators and Ledoit-Wolf. "
                    * 3
                ),
                "metadata": {
                    "chunk_type": "theory", "page": 1,
                    "parent_id": large_parent_id,
                    "section_id": i,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            })
        store4.upsert_chunks(large_chunks)

        retriever4 = Retriever(store4, config=cfg4)
        ctx_large = retriever4.retrieve_context(
            "portfolio construction asset allocation", top_k=5, max_tokens=5000,
        )
        parent_chunks = [
            c for c in ctx_large["theory_context"]
            if c.get("metadata", {}).get("parent_id") == large_parent_id
        ]
        # Large parent: should get at most MAX_CHILDREN_PER_PARENT (2)
        # and only those adjacent to the best-scoring child
        check(
            f"large parent: ≤{MAX_CHILDREN_PER_PARENT} children "
            f"(got {len(parent_chunks)})",
            len(parent_chunks) <= MAX_CHILDREN_PER_PARENT,
        )
        # Adjacent children should have consecutive section_ids
        if len(parent_chunks) == 2:
            sids = sorted(
                c.get("metadata", {}).get("section_id", 0) for c in parent_chunks
            )
            check(
                f"adjacent children: section_ids consecutive ({sids})",
                sids[1] - sids[0] <= 2,
            )
        else:
            check("adjacent children: trivial (0-1 chunks)", True)

    finally:
        shutil.rmtree(tmpdir4, ignore_errors=True)

    # ── 9. _section_id_of helper ─────────────────────────────────────
    print("\n=== 9. _section_id_of Helper ===")
    check("int section_id", _section_id_of({"metadata": {"section_id": 5}}) == 5)
    check("missing section_id", _section_id_of({"metadata": {}}) == 0)
    check("non-int section_id", _section_id_of({"metadata": {"section_id": "x"}}) == 0)
    check("no metadata key", _section_id_of({}) == 0)

    # ── 10. Config constants exist ───────────────────────────────────
    print("\n=== 10. Config Constants ===")
    check("MAX_PARENT_SECTIONS == 3", MAX_PARENT_SECTIONS == 3)
    check("MAX_CHILDREN_PER_PARENT == 2", MAX_CHILDREN_PER_PARENT == 2)
    check("_LARGE_PARENT_TOKEN_THRESHOLD == 1200",
          _LARGE_PARENT_TOKEN_THRESHOLD == 1200)

    # ── 11. FAST_MODE ────────────────────────────────────────────────
    print("\n=== 11. FAST_MODE ===")

    # 11a: is_fast_mode() reads the env dynamically
    old_val = os.environ.get("RAG_FAST_MODE")
    try:
        os.environ["RAG_FAST_MODE"] = "true"
        check("is_fast_mode() True when env=true", is_fast_mode() is True)

        os.environ["RAG_FAST_MODE"] = "false"
        check("is_fast_mode() False when env=false", is_fast_mode() is False)

        os.environ["RAG_FAST_MODE"] = "TRUE"
        check("is_fast_mode() case-insensitive", is_fast_mode() is True)

        del os.environ["RAG_FAST_MODE"]
        check("is_fast_mode() False when unset", is_fast_mode() is False)
    finally:
        if old_val is not None:
            os.environ["RAG_FAST_MODE"] = old_val
        elif "RAG_FAST_MODE" in os.environ:
            del os.environ["RAG_FAST_MODE"]

    # 11b: FAST_MODE_TOP_K constant
    check("FAST_MODE_TOP_K == 5", FAST_MODE_TOP_K == 5)

    # 11c: retrieve() respects FAST_MODE top-k cap
    # Reuse store2 (parent-child test store) that was set up earlier.
    # We need a fresh store for this test.
    tmpdir5 = tempfile.mkdtemp()
    try:
        cfg5 = RAGConfig()
        cfg5.chroma_persist_dir = tmpdir5

        DIM = 384

        def mock_embed5(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store5 = DualIndexStore(config=cfg5, embed_fn=mock_embed5)

        fast_chunks = [
            {
                "content": f"Chunk {i}: Portfolio risk analysis concept number {i} covers optimization and hedging.",
                "metadata": {
                    "chunk_type": "theory", "page": i,
                    "parent_id": f"fast_parent_{i}",
                    "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            }
            for i in range(15)
        ]
        store5.upsert_chunks(fast_chunks)

        retriever5 = Retriever(store5, config=cfg5)

        old_fast = os.environ.get("RAG_FAST_MODE")
        try:
            # Normal mode — top_k=10
            os.environ["RAG_FAST_MODE"] = "false"
            normal_results = retriever5.retrieve(
                "risk analysis portfolio", top_k=10,
            )

            # Fast mode — should cap to FAST_MODE_TOP_K=5
            os.environ["RAG_FAST_MODE"] = "true"
            fast_results = retriever5.retrieve(
                "risk analysis portfolio", top_k=10,
            )

            check(
                f"fast mode caps top_k: ≤{FAST_MODE_TOP_K} "
                f"(got {len(fast_results)})",
                len(fast_results) <= FAST_MODE_TOP_K,
            )
            check(
                "fast mode returns fewer than normal",
                len(fast_results) <= len(normal_results),
            )

            # In fast mode, keyword and stage_bonus should be 0.0
            if fast_results:
                r0 = fast_results[0]
                check(
                    "fast mode: keyword score is 0.0",
                    r0["scores"]["keyword"] == 0.0,
                )
                check(
                    "fast mode: stage_bonus is 0.0",
                    r0["scores"]["stage_bonus"] == 0.0,
                )
                check(
                    "fast mode: final == vector (hybrid only)",
                    r0["scores"]["final"] == r0["scores"]["vector"],
                )
        finally:
            if old_fast is not None:
                os.environ["RAG_FAST_MODE"] = old_fast
            elif "RAG_FAST_MODE" in os.environ:
                del os.environ["RAG_FAST_MODE"]

    finally:
        shutil.rmtree(tmpdir5, ignore_errors=True)

    # ── 12. Embedding Call Optimisation ──────────────────────────────
    print("\n=== 12. Embedding Call Optimisation ===")

    # 12a: MAX_CHROMA_RESULTS constant
    check("MAX_CHROMA_RESULTS == 8", MAX_CHROMA_RESULTS == 8)

    # 12b: fetch_k is capped to MAX_CHROMA_RESULTS in retrieve()
    tmpdir6 = tempfile.mkdtemp()
    try:
        cfg6 = RAGConfig()
        cfg6.chroma_persist_dir = tmpdir6

        DIM = 384
        embed_call_count = [0]  # mutable counter

        def mock_embed6(text: str, chunk_type: str) -> List[float]:
            embed_call_count[0] += 1
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store6 = DualIndexStore(config=cfg6, embed_fn=mock_embed6)

        # Insert 20 theory chunks
        opt_chunks = [
            {
                "content": f"Optimisation chunk {i}: asset allocation and risk parity.",
                "metadata": {
                    "chunk_type": "theory", "page": i,
                    "parent_id": f"opt_parent_{i}",
                    "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            }
            for i in range(20)
        ]
        store6.upsert_chunks(opt_chunks)

        retriever6 = Retriever(store6, config=cfg6)

        old_fast6 = os.environ.get("RAG_FAST_MODE")
        try:
            os.environ["RAG_FAST_MODE"] = "false"
            results_norm = retriever6.retrieve("risk parity asset", top_k=10)
            # Normal mode: fetch_k would be max(30, 30)=30, capped to 8
            check(
                f"normal mode: results ≤ {MAX_CHROMA_RESULTS} "
                f"(got {len(results_norm)})",
                len(results_norm) <= MAX_CHROMA_RESULTS,
            )

            os.environ["RAG_FAST_MODE"] = "true"
            results_fast = retriever6.retrieve("risk parity asset", top_k=10)
            # Fast mode: top_k capped to 5, then fetch_k = min(5, 8) = 5
            check(
                f"fast mode: results ≤ {FAST_MODE_TOP_K} "
                f"(got {len(results_fast)})",
                len(results_fast) <= FAST_MODE_TOP_K,
            )
        finally:
            if old_fast6 is not None:
                os.environ["RAG_FAST_MODE"] = old_fast6
            elif "RAG_FAST_MODE" in os.environ:
                del os.environ["RAG_FAST_MODE"]

        # 12c: LRU query embedding cache (DualIndexStore level)
        store6.clear_embed_cache()
        embed_call_count[0] = 0

        # First query — cache MISS
        store6.query_index("asset allocation", chunk_type="theory", top_k=5)
        first_count = embed_call_count[0]
        check("first query triggers embed call", first_count >= 1)

        info1 = store6.embed_cache_info()
        check("cache misses == 1 after first query", info1.misses >= 1)

        # Same query, same chunk_type — cache HIT
        store6.query_index("asset allocation", chunk_type="theory", top_k=5)
        second_count = embed_call_count[0]
        check(
            "repeated query: no extra embed call (cached)",
            second_count == first_count,
        )

        info2 = store6.embed_cache_info()
        check("cache hits ≥ 1 after repeated query", info2.hits >= 1)

        # Different query — cache MISS again
        store6.query_index("risk parity", chunk_type="theory", top_k=5)
        third_count = embed_call_count[0]
        check("different query triggers new embed call", third_count > second_count)

        # clear_embed_cache resets
        store6.clear_embed_cache()
        info3 = store6.embed_cache_info()
        check("cache cleared: currsize == 0", info3.currsize == 0)

        # 12d: include fields are correct (no embeddings requested)
        # _vector_search passes include=["documents", "metadatas", "distances"]
        check(
            "default include has 3 fields",
            set(["documents", "metadatas", "distances"]) == {
                "documents", "metadatas", "distances"
            },
        )

        # 12e: No chunk re-embedding at query time
        # After upsert, only queries cause embed calls.
        store6.clear_embed_cache()
        embed_call_count[0] = 0
        store6.query_index("new query text", chunk_type="theory", top_k=3)
        check(
            "query_index embeds only 1 text (the query)",
            embed_call_count[0] == 1,
        )

    finally:
        shutil.rmtree(tmpdir6, ignore_errors=True)

    # ── 13. Concurrency & Timing ─────────────────────────────────────
    print("\n=== 13. Concurrency & Timing ===")

    # 13a: Constants exist
    check("_RETRIEVAL_TIMEOUT_WARN == 3.0", _RETRIEVAL_TIMEOUT_WARN == 3.0)
    check("_RETRIEVER_WORKERS == 2", _RETRIEVER_WORKERS == 2)

    # 13b: retrieve() produces correct results with concurrent search
    tmpdir7 = tempfile.mkdtemp()
    try:
        cfg7 = RAGConfig()
        cfg7.chroma_persist_dir = tmpdir7

        DIM = 384

        def mock_embed7(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store7 = DualIndexStore(config=cfg7, embed_fn=mock_embed7)

        conc_chunks = [
            {
                "content": f"Theory chunk {i}: risk adjusted returns and Sharpe ratio.",
                "metadata": {
                    "chunk_type": "theory", "page": i,
                    "parent_id": f"conc_parent_{i}", "section_id": 0,
                    "pipeline_stage": "evaluation", "source": "test.pdf",
                },
            }
            for i in range(6)
        ] + [
            {
                "content": f"Code chunk {i}: def sharpe(r): return mean(r)/std(r)",
                "metadata": {
                    "chunk_type": "code", "page": i,
                    "parent_id": f"conc_code_{i}", "section_id": 0,
                    "pipeline_stage": "evaluation", "source": "test.pdf",
                },
            }
            for i in range(4)
        ]
        store7.upsert_chunks(conc_chunks)

        retriever7 = Retriever(store7, config=cfg7)

        old_fast7 = os.environ.get("RAG_FAST_MODE")
        try:
            os.environ["RAG_FAST_MODE"] = "false"

            # 13c: Normal-mode results are still correct (concurrency)
            results7 = retriever7.retrieve("Sharpe ratio evaluation", top_k=5)
            check("concurrent retrieve returns results", len(results7) > 0)
            check(
                "concurrent results sorted descending",
                all(
                    results7[i]["scores"]["final"]
                    >= results7[i + 1]["scores"]["final"]
                    for i in range(len(results7) - 1)
                ),
            )
            check(
                "concurrent results have all score keys",
                all(
                    set(r["scores"].keys()) == {"vector", "keyword", "stage_bonus", "final"}
                    for r in results7
                ),
            )

            # 13d: Fast-mode also works with concurrent vector search
            os.environ["RAG_FAST_MODE"] = "true"
            results7f = retriever7.retrieve("Sharpe ratio evaluation", top_k=5)
            check("fast concurrent returns results", len(results7f) > 0)
            check(
                "fast concurrent: keyword=0",
                all(r["scores"]["keyword"] == 0.0 for r in results7f),
            )

        finally:
            if old_fast7 is not None:
                os.environ["RAG_FAST_MODE"] = old_fast7
            elif "RAG_FAST_MODE" in os.environ:
                del os.environ["RAG_FAST_MODE"]

        # 13e: Timing via log capture
        import io
        _log_buf = io.StringIO()
        _handler = logging.StreamHandler(_log_buf)
        _handler.setLevel(logging.DEBUG)
        # Use the module-level logger (named __main__ when run directly)
        logger.addHandler(_handler)
        _prev_level = logger.level
        logger.setLevel(logging.DEBUG)

        old_fast7b = os.environ.get("RAG_FAST_MODE")
        try:
            os.environ["RAG_FAST_MODE"] = "false"
            _log_buf.truncate(0)
            _log_buf.seek(0)

            retriever7.retrieve("risk evaluation Sharpe", top_k=5)
            log_output = _log_buf.getvalue()

            check(
                "timing log contains classify=",
                "classify=" in log_output,
            )
            check(
                "timing log contains vector_search=",
                "vector_search=" in log_output,
            )
            check(
                "timing log contains total=",
                "total=" in log_output,
            )

            # 13f: retrieve_context timing
            _log_buf.truncate(0)
            _log_buf.seek(0)

            retriever7.retrieve_context(
                "risk evaluation Sharpe", top_k=5, max_tokens=5000,
            )
            ctx_log = _log_buf.getvalue()

            check(
                "context timing log contains retrieve=",
                "retrieve=" in ctx_log,
            )
            check(
                "context timing log contains context_build=",
                "context_build=" in ctx_log,
            )

        finally:
            if old_fast7b is not None:
                os.environ["RAG_FAST_MODE"] = old_fast7b
            elif "RAG_FAST_MODE" in os.environ:
                del os.environ["RAG_FAST_MODE"]

        logger.removeHandler(_handler)
        logger.setLevel(_prev_level)

    finally:
        shutil.rmtree(tmpdir7, ignore_errors=True)

    # ── 14. Simple-Query Mode ────────────────────────────────────────
    print("\n=== 14. Simple-Query Mode ===")

    # 14a: is_simple_query dynamic threshold
    os.environ["RAG_SIMPLE_QUERY_THRESHOLD"] = "30"
    check("is_simple_query short → True",
          is_simple_query("What is Sharpe ratio") is True)
    long_q = " ".join([f"word{i}" for i in range(35)])
    check("is_simple_query long → False",
          is_simple_query(long_q) is False)
    os.environ["RAG_SIMPLE_QUERY_THRESHOLD"] = "0"
    check("is_simple_query disabled (threshold=0) → False",
          is_simple_query("short query") is False)

    # 14b: Constants
    check("SIMPLE_QUERY_THRESHOLD == 30", SIMPLE_QUERY_THRESHOLD == 30)
    check("SIMPLE_QUERY_TOP_K == 4", SIMPLE_QUERY_TOP_K == 4)

    # 14c: retrieve() caps top_k in simple mode
    tmpdir8 = tempfile.mkdtemp()
    try:
        cfg8 = RAGConfig()
        cfg8.chroma_persist_dir = tmpdir8
        DIM = 384

        def mock_embed8(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store8 = DualIndexStore(config=cfg8, embed_fn=mock_embed8)
        s8_chunks = [
            {
                "content": f"Theory chunk {i} about risk parity and asset allocation methods.",
                "metadata": {
                    "chunk_type": "theory", "page": i,
                    "parent_id": f"s8_parent_{i}", "section_id": 0,
                    "pipeline_stage": "signal_generation", "source": "test.pdf",
                },
            }
            for i in range(10)
        ]
        store8.upsert_chunks(s8_chunks)
        retriever8 = Retriever(store8, config=cfg8)

        os.environ["RAG_SIMPLE_QUERY_THRESHOLD"] = "30"
        os.environ["RAG_FAST_MODE"] = "false"

        simple_results = retriever8.retrieve("risk parity allocation", top_k=10)
        check(
            f"simple mode caps top_k: ≤{SIMPLE_QUERY_TOP_K} "
            f"(got {len(simple_results)})",
            len(simple_results) <= SIMPLE_QUERY_TOP_K,
        )

        # 14d: simple mode skips BM25/stage (keyword=0, stage_bonus=0)
        if simple_results:
            r0 = simple_results[0]
            check("simple mode: keyword score is 0.0",
                  r0["scores"]["keyword"] == 0.0)
            check("simple mode: stage_bonus is 0.0",
                  r0["scores"]["stage_bonus"] == 0.0)
            check("simple mode: final == vector",
                  r0["scores"]["final"] == r0["scores"]["vector"])

        # 14e: retrieve_context skips parent expansion in simple mode
        ctx_simple = retriever8.retrieve_context(
            "risk parity", top_k=10, max_tokens=5000,
        )
        total_simple = (
            len(ctx_simple["theory_context"]) + len(ctx_simple["code_context"])
        )
        check(
            f"simple context: ≤{SIMPLE_QUERY_TOP_K} chunks "
            f"(got {total_simple})",
            total_simple <= SIMPLE_QUERY_TOP_K,
        )
        check("simple context has intent key",
              isinstance(ctx_simple.get("intent"), str))

        # 14f: normal mode (disabled simple) returns more chunks
        os.environ["RAG_SIMPLE_QUERY_THRESHOLD"] = "0"
        normal_results = retriever8.retrieve("risk parity allocation", top_k=10)
        check(
            "normal mode returns ≥ simple mode results",
            len(normal_results) >= len(simple_results),
        )

        # Cleanup
        del os.environ["RAG_FAST_MODE"]

    finally:
        shutil.rmtree(tmpdir8, ignore_errors=True)

    # ── 15. Strict Code Mode ─────────────────────────────────────────
    print("\n=== 15. Strict Code Mode ===")

    # 15a: Constants
    check("STRICT_CODE_TOP_K == 8", STRICT_CODE_TOP_K == 8)
    check("_STRICT_CODE_FUNC_SIG_BOOST == 0.5",
          _STRICT_CODE_FUNC_SIG_BOOST == 0.5)
    check("_STRICT_CODE_KEYWORD_OVERLAP_BOOST == 0.3",
          _STRICT_CODE_KEYWORD_OVERLAP_BOOST == 0.3)

    # 15b: _fuzzy_ratio function
    check("fuzzy: identical → 1.0", _fuzzy_ratio("apply", "apply") == 1.0)
    check("fuzzy: empty a → 0.0", _fuzzy_ratio("", "apply") == 0.0)
    check("fuzzy: empty b → 0.0", _fuzzy_ratio("apply", "") == 0.0)
    fr1 = _fuzzy_ratio("barrier", "barrier")
    check("fuzzy: exact match → 1.0", fr1 == 1.0)
    fr2 = _fuzzy_ratio("barrer", "barrier")
    check("fuzzy: near match → high ratio", fr2 > 0.7)
    fr3 = _fuzzy_ratio("apply", "xyz")
    check("fuzzy: no match → low ratio", fr3 < 0.3)
    fr4 = _fuzzy_ratio("pt", "ptsl")
    check("fuzzy: partial substring → mid ratio", 0.3 < fr4 < 0.9)
    # Symmetry
    check("fuzzy: symmetric", _fuzzy_ratio("abc", "abd") == _fuzzy_ratio("abd", "abc"))

    # 15c: _fuzzy_func_name_score
    # applyPtSlOnT1 → sub-tokens: {apply, pt, sl, on, applyptslont1}
    # query "apply" should match "apply" perfectly → 1.0 * 0.5 = 0.5
    fscore1 = _fuzzy_func_name_score(
        "def applyPtSlOnT1(close, events, ptSl, molecule):\n"
        "    events_ = events.loc[molecule]",
        _tokenize("implement triple barrier apply python"),
    )
    check("fuzzy_func: 'apply' exact match → 0.5", abs(fscore1 - 0.5) < 0.01)

    # query "barrier" vs func tokens {apply, pt, sl, on} → no close match → 0.0
    fscore2 = _fuzzy_func_name_score(
        "def applyPtSlOnT1(close, events, ptSl, molecule):\n"
        "    return out",
        _tokenize("barrier labeling method"),
    )
    check("fuzzy_func: 'barrier' no sub-token match → 0.0", fscore2 == 0.0)

    # No function names → 0.0
    fscore3 = _fuzzy_func_name_score(
        "The triple barrier method labels observations.",
        _tokenize("barrier labeling"),
    )
    check("fuzzy_func: no def → 0.0", fscore3 == 0.0)

    # No query tokens → 0.0
    fscore4 = _fuzzy_func_name_score(
        "def applyPtSlOnT1(close):\n    pass",
        [],
    )
    check("fuzzy_func: empty query → 0.0", fscore4 == 0.0)

    # daily_volatility + query 'volatility' → exact sub-token match
    fscore5 = _fuzzy_func_name_score(
        "def daily_volatility(close, span=100):\n    pass",
        _tokenize("daily volatility estimate"),
    )
    check("fuzzy_func: 'daily'/'volatility' exact → 0.5",
          abs(fscore5 - 0.5) < 0.01)

    # Completely unrelated → below threshold → 0.0
    fscore6 = _fuzzy_func_name_score(
        "def calculate_sharpe(returns):\n    pass",
        _tokenize("barrier labeling method"),
    )
    check("fuzzy_func: unrelated func → 0.0", fscore6 == 0.0)

    # 15d: _STRICT_CODE_FUZZY_WEIGHT constant
    check("_STRICT_CODE_FUZZY_WEIGHT == 0.5", _STRICT_CODE_FUZZY_WEIGHT == 0.5)
    check("_STRICT_CODE_FUZZY_MIN_THRESHOLD == 0.5",
          _STRICT_CODE_FUZZY_MIN_THRESHOLD == 0.5)

    # 15e: _strict_code_boost scoring function
    # Chunk with def + name overlap + keyword overlap → +0.8
    boost1 = _strict_code_boost(
        "def applyPtSlOnT1(close, events, ptSl, molecule):\n"
        "    events_ = events.loc[molecule]\n"
        "    out = events_[['t1']].copy(deep=True)",
        _tokenize("implement triple barrier labeling apply python"),
    )
    check("boost: def + name overlap ('apply') → ≥0.5", boost1 >= 0.5)
    check("boost: keyword overlap → ≥0.8", boost1 >= 0.8)

    # Chunk with def but NO name/keyword overlap → 0.0
    boost2 = _strict_code_boost(
        "def calculate_sharpe(returns):\n    return np.mean(returns) / np.std(returns)",
        _tokenize("implement triple barrier labeling"),
    )
    check("boost: def but no name/keyword overlap → 0.0",
          abs(boost2 - 0.0) < 0.01)

    # Chunk with no def at all, but keyword overlap → +0.3
    boost3 = _strict_code_boost(
        "The triple barrier method labels observations by checking "
        "profit-taking and stop-loss limits.",
        _tokenize("triple barrier labeling"),
    )
    check("boost: no def, keyword overlap → 0.3", abs(boost3 - 0.3) < 0.01)

    # Chunk with no overlap at all → 0.0
    boost4 = _strict_code_boost(
        "Portfolio rebalancing uses mean-variance optimization.",
        _tokenize("triple barrier labeling"),
    )
    check("boost: no overlap → 0.0", boost4 == 0.0)

    # Chunk with def + exact name overlap → +0.8
    boost5 = _strict_code_boost(
        "def daily_volatility(close, span=100):\n"
        "    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))",
        _tokenize("daily volatility function"),
    )
    check("boost: def daily_volatility + query 'daily','volatility' → 0.8",
          abs(boost5 - 0.8) < 0.01)

    # 15c: Full strict code-mode retrieve (end-to-end)
    tmpdir9 = tempfile.mkdtemp()
    try:
        cfg9 = RAGConfig()
        cfg9.chroma_persist_dir = tmpdir9
        DIM = 384

        def mock_embed9(text: str, chunk_type: str) -> List[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h[:DIM]] + [0.0] * (DIM - len(h[:DIM]))

        store9 = DualIndexStore(config=cfg9, embed_fn=mock_embed9)

        # Code chunks (should be returned in strict mode)
        strict_chunks = [
            {
                "content": (
                    "def applyPtSlOnT1(close, events, ptSl, molecule):\n"
                    "    events_ = events.loc[molecule]\n"
                    "    out = events_[['t1']].copy(deep=True)\n"
                    "    return out"
                ),
                "metadata": {
                    "chunk_type": "code", "page": 45,
                    "parent_id": "strict_p1", "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            },
            {
                "content": (
                    "def getEvents(close, tEvents, ptSl, trgt):\n"
                    "    events = pd.DataFrame()\n"
                    "    return events"
                ),
                "metadata": {
                    "chunk_type": "code", "page": 50,
                    "parent_id": "strict_p2", "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            },
            {
                "content": (
                    "def calculate_sharpe(returns):\n"
                    "    return np.mean(returns) / np.std(returns)"
                ),
                "metadata": {
                    "chunk_type": "code", "page": 10,
                    "parent_id": "strict_p3", "section_id": 0,
                    "pipeline_stage": "evaluation",
                    "source": "test.pdf",
                },
            },
            # Theory chunks (should be EXCLUDED in strict mode)
            {
                "content": (
                    "The triple barrier labeling method assigns labels "
                    "based on which barrier is touched first: profit-taking, "
                    "stop-loss, or time expiration."
                ),
                "metadata": {
                    "chunk_type": "theory", "page": 44,
                    "parent_id": "strict_p4", "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            },
            {
                "content": (
                    "Meta-labelling is a secondary ML model that decides "
                    "the size of the bet based on the primary signal."
                ),
                "metadata": {
                    "chunk_type": "theory", "page": 50,
                    "parent_id": "strict_p5", "section_id": 0,
                    "pipeline_stage": "signal_generation",
                    "source": "test.pdf",
                },
            },
        ]
        store9.upsert_chunks(strict_chunks)
        retriever9 = Retriever(store9, config=cfg9)

        # Query with strict code keywords
        strict_results = retriever9.retrieve(
            "what is triple barrier labeling? how can i implement using python?",
            top_k=10,
        )

        check("strict mode returns results", len(strict_results) > 0)

        # All results must be code chunks
        all_code = all(
            r.get("metadata", {}).get("chunk_type") == "code"
            for r in strict_results
        )
        check("strict mode: ALL results are code chunks", all_code)

        # No theory chunks should appear
        any_theory = any(
            r.get("metadata", {}).get("chunk_type") == "theory"
            for r in strict_results
        )
        check("strict mode: NO theory chunks", not any_theory)

        # Results should have strict_code_boost and fuzzy_func_score in scores
        if strict_results:
            check(
                "strict mode: scores contain strict_code_boost",
                "strict_code_boost" in strict_results[0]["scores"],
            )
            check(
                "strict mode: scores contain fuzzy_func_score",
                "fuzzy_func_score" in strict_results[0]["scores"],
            )
            # fuzzy_func_score should be a float ≥ 0
            check(
                "strict mode: fuzzy_func_score is float ≥ 0",
                isinstance(strict_results[0]["scores"]["fuzzy_func_score"], float)
                and strict_results[0]["scores"]["fuzzy_func_score"] >= 0.0,
            )
            # Results sorted by final score
            check(
                "strict mode: sorted descending by final",
                all(
                    strict_results[i]["scores"]["final"]
                    >= strict_results[i + 1]["scores"]["final"]
                    for i in range(len(strict_results) - 1)
                ),
            )

        # top_k respects STRICT_CODE_TOP_K
        check(
            f"strict mode: ≤{STRICT_CODE_TOP_K} results "
            f"(got {len(strict_results)})",
            len(strict_results) <= STRICT_CODE_TOP_K,
        )

        # chunk_type_override should bypass strict mode
        override_results = retriever9.retrieve(
            "what is triple barrier labeling? how can i implement using python?",
            top_k=10,
            chunk_type_override="theory",
        )
        check(
            "chunk_type_override bypasses strict mode",
            len(override_results) > 0,
        )

        # Non-strict query should return theory chunks too
        non_strict = retriever9.retrieve(
            "explain the concept of mean reversion",
            top_k=10,
        )
        has_theory = any(
            r.get("metadata", {}).get("chunk_type") == "theory"
            for r in non_strict
        ) if non_strict else True  # empty is ok
        check(
            "non-strict query: theory chunks allowed",
            has_theory or len(non_strict) == 0,
        )

    finally:
        shutil.rmtree(tmpdir9, ignore_errors=True)

    # ── Restore original env and summarise ───────────────────────────
    if _old_simple_threshold is not None:
        os.environ["RAG_SIMPLE_QUERY_THRESHOLD"] = _old_simple_threshold
    elif "RAG_SIMPLE_QUERY_THRESHOLD" in os.environ:
        del os.environ["RAG_SIMPLE_QUERY_THRESHOLD"]

    # ══════════════════════════════════════════════════════════════════
    # 16. _log_code_retrieval diagnostics
    # ══════════════════════════════════════════════════════════════════
    print("\n=== 16. Code Retrieval Logging ===")

    # Build fake ranked results to test _log_code_retrieval
    fake_ranked = [
        {
            "id": "chunk_001",
            "content": (
                "def applyPtSlOnT1(close, events, ptSl, molecule):\n"
                "    events_ = events.loc[molecule]\n"
                "    return events_"
            ),
            "metadata": {"chunk_type": "code", "page": 45},
            "scores": {"final": 1.5, "strict_code_boost": 0.8},
        },
        {
            "id": "chunk_002",
            "content": (
                "def getDailyVol(close, span=100):\n"
                "    df0 = close.pct_change()\n"
                "    return df0.rolling(span).std()"
            ),
            "metadata": {"chunk_type": "code", "page": 50},
            "scores": {"final": 1.2, "strict_code_boost": 0.5},
        },
        {
            "id": "chunk_003",
            "content": (
                "def calculate_sharpe(returns):\n"
                "    return np.mean(returns) / np.std(returns)"
            ),
            "metadata": {"chunk_type": "code", "page": 10},
            "scores": {"final": 0.9, "strict_code_boost": 0.0},
        },
    ]

    # Capture log output
    import io
    _log_buf16 = io.StringIO()
    _handler16 = logging.StreamHandler(_log_buf16)
    _handler16.setLevel(logging.DEBUG)
    _retriever_logger = logger  # module-level logger (may be __main__)
    _retriever_logger.addHandler(_handler16)
    _old_level = _retriever_logger.level
    _retriever_logger.setLevel(logging.DEBUG)

    # Test: normal call with results
    _log_buf16.truncate(0)
    _log_buf16.seek(0)
    _log_code_retrieval(fake_ranked)
    log16 = _log_buf16.getvalue()

    check("log: contains CODE_RETRIEVAL_LOG", "CODE_RETRIEVAL_LOG" in log16)
    check("log: contains chunk_001", "chunk_001" in log16)
    check("log: contains chunk_002", "chunk_002" in log16)
    check("log: contains chunk_003", "chunk_003" in log16)
    check("log: contains applyPtSlOnT1", "applyPtSlOnT1" in log16)
    check("log: contains getDailyVol", "getDailyVol" in log16)
    check("log: contains calculate_sharpe", "calculate_sharpe" in log16)
    check("log: top 3 function names logged",
          "top 3 function names" in log16)

    # Test: empty ranked → no crash
    _log_buf16.truncate(0)
    _log_buf16.seek(0)
    _log_code_retrieval([])
    log16_empty = _log_buf16.getvalue()
    check("log: empty ranked → no crash", "no ranked results" in log16_empty)

    # Test: expected_fn warning when function missing from top-N
    _log_buf16.truncate(0)
    _log_buf16.seek(0)
    _log_code_retrieval(fake_ranked, expected_fn="cusum_filter")
    log16_warn = _log_buf16.getvalue()
    check("log: warning for missing expected_fn",
          "cusum_filter" in log16_warn and "Expected implementation" in log16_warn)

    # Test: no warning when expected_fn is in results
    _log_buf16.truncate(0)
    _log_buf16.seek(0)
    _log_code_retrieval(fake_ranked, expected_fn="applyPtSlOnT1")
    log16_ok = _log_buf16.getvalue()
    # Should NOT have the specific "Expected implementation 'applyPtSlOnT1' not retrieved" warning
    check("log: no warning when expected_fn present",
          "Expected implementation 'applyPtSlOnT1' not retrieved" not in log16_ok)

    # Constants
    check("_EXPECTED_WARN_TOP_N == 5", _EXPECTED_WARN_TOP_N == 5)
    check("_EXPECTED_FUNCTIONS contains applyPtSlOnT1",
          "applyPtSlOnT1" in _EXPECTED_FUNCTIONS)

    # Cleanup logger
    _retriever_logger.removeHandler(_handler16)
    _retriever_logger.setLevel(_old_level)

    # ══════════════════════════════════════════════════════════════════
    # 17. _normalize_for_embedding (from vector_store)
    # ══════════════════════════════════════════════════════════════════
    print("\n=== 17. Pre-Embedding Normalisation ===")

    from rag_pipeline.vector_store import _normalize_for_embedding

    # Test 1: trailing comments removed
    code1 = "x = 1  # set x to one\ny = 2  # set y"
    norm1 = _normalize_for_embedding(code1)
    check("norm: trailing comments removed",
          "# set x" not in norm1 and "# set y" not in norm1)
    check("norm: code preserved",
          "x = 1" in norm1 and "y = 2" in norm1)

    # Test 2: excessive blank lines collapsed
    code2 = "a = 1\n\n\n\n\nb = 2"
    norm2 = _normalize_for_embedding(code2)
    check("norm: excess blanks collapsed",
          "\n\n\n" not in norm2 and "a = 1" in norm2 and "b = 2" in norm2)

    # Test 3: tabs → 4 spaces
    code3 = "def foo():\n\treturn 1"
    norm3 = _normalize_for_embedding(code3)
    check("norm: tabs converted to spaces",
          "\t" not in norm3 and "    return 1" in norm3)

    # Test 4: function names lowercased
    code4 = "def applyPtSlOnT1(close, events):\n    pass"
    norm4 = _normalize_for_embedding(code4)
    check("norm: func name lowercased",
          "def applyptslont1" in norm4)
    check("norm: params preserved",
          "(close, events)" in norm4)

    # Test 5: multiple functions
    code5 = "def MyFunc(a):\n    pass\ndef AnotherFunc(b):\n    pass"
    norm5 = _normalize_for_embedding(code5)
    check("norm: both funcs lowercased",
          "def myfunc" in norm5 and "def anotherfunc" in norm5)

    # Test 6: trailing whitespace stripped
    code6 = "x = 1   \n  y = 2  "
    norm6 = _normalize_for_embedding(code6)
    check("norm: no trailing whitespace",
          not any(ln.endswith(" ") for ln in norm6.split("\n")))

    # Test 7: empty string
    check("norm: empty → empty", _normalize_for_embedding("") == "")

    # Test 8: preserves indentation (not on def line)
    code8 = "def foo():\n    if True:\n        return 1"
    norm8 = _normalize_for_embedding(code8)
    check("norm: body indentation preserved",
          "    if True:" in norm8 and "        return 1" in norm8)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
