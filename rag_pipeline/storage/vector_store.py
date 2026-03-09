"""
ChromaDB Vector Store for Centurion Capital LLC RAG Pipeline.

Dual-index architecture:
    - **code_index**   — stores code chunks (functions, classes, imports)
    - **theory_index** — stores theory / prose chunks

Features:
    - Embedding-function abstraction (``get_embedding``)
    - Deterministic IDs (SHA-256 of content) — no duplicate insertions
    - Batch upsert with automatic index routing by ``chunk_type``
    - Filtered similarity search (``query_index``)
    - Full backward compatibility (``VectorStoreManager`` preserved)

Tech stack: Python 3.11 · raw chromadb client · No LangChain.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from rag_pipeline.config import RAGConfig

if TYPE_CHECKING:
    import chromadb
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CODE_COLLECTION = "code_index"
THEORY_COLLECTION = "theory_index"
_BATCH_SIZE = 256  # ChromaDB recommended max per upsert call

# LRU cache for query embeddings — avoids redundant encode calls
# when the same query hits both code_index and theory_index or is retried.
_QUERY_EMBED_CACHE_SIZE = 128

# Valid chunk types that map to a collection
_CHUNK_TYPE_MAP: Dict[str, str] = {
    "code": CODE_COLLECTION,
    "theory": THEORY_COLLECTION,
    "text": THEORY_COLLECTION,  # alias
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Lazy ChromaDB import
# ═══════════════════════════════════════════════════════════════════════════

def _import_chromadb():
    """Lazy import of chromadb to avoid ~1 s startup penalty."""
    import chromadb as _chromadb
    return _chromadb


# ═══════════════════════════════════════════════════════════════════════════
# 2. Deterministic ID generation
# ═══════════════════════════════════════════════════════════════════════════

def _content_id(content: str) -> str:
    """Return a deterministic, collision-resistant ID from content.

    Uses SHA-256 truncated to 16 hex chars (64 bits). Probability of
    collision is ~1 in 10^18 at 1 M documents — more than sufficient.

    Args:
        content: The chunk text.

    Returns:
        16-character hex string.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Embedding abstraction
# ═══════════════════════════════════════════════════════════════════════════

# Type alias for an embedding function
EmbeddingFn = Callable[[str, str], List[float]]


def get_embedding(text: str, chunk_type: str) -> List[float]:
    """Placeholder embedding function.

    Override this by injecting a real ``EmbeddingFn`` into
    ``DualIndexStore`` at initialisation time, or monkey-patch
    this module-level function for quick prototyping.

    Args:
        text: Text to embed.
        chunk_type: ``"code"`` or ``"theory"`` (allows future
            per-type model routing).

    Returns:
        A list of floats (embedding vector).

    Raises:
        NotImplementedError: Always — replace with a real backend.
    """
    raise NotImplementedError(
        "get_embedding() is a placeholder. Provide a real EmbeddingFn "
        "to DualIndexStore or override this function."
    )


def _make_embedding_fn_from_service(embedding_service) -> EmbeddingFn:
    """Wrap an ``EmbeddingService`` instance as an ``EmbeddingFn``.

    Args:
        embedding_service: An ``EmbeddingService`` (from embeddings.py).

    Returns:
        A callable matching the ``EmbeddingFn`` signature.
    """
    def _fn(text: str, chunk_type: str) -> List[float]:
        return embedding_service.embed_texts([text])[0]
    return _fn


# ═══════════════════════════════════════════════════════════════════════════
# 4. DualIndexStore — the new dual-collection manager
# ═══════════════════════════════════════════════════════════════════════════

class DualIndexStore:
    """Manages two ChromaDB collections: ``code_index`` and ``theory_index``.

    Chunks are automatically routed to the correct collection based on
    ``metadata["chunk_type"]``.  IDs are deterministic (content hash),
    so duplicate upserts are idempotent.

    Usage::

        store = DualIndexStore(config, embed_fn=my_embed_fn)
        store.upsert_chunks(enriched_chunks)
        results = store.query_index("Sharpe ratio", chunk_type="theory", top_k=5)

    Args:
        config: ``RAGConfig`` instance (ChromaDB path, HNSW params, etc.).
        embed_fn: Callable ``(text, chunk_type) -> list[float]``.
            Defaults to the module-level ``get_embedding`` placeholder.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embed_fn: Optional[EmbeddingFn] = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._embed_fn: EmbeddingFn = embed_fn or get_embedding
        self._client = None
        self._collections: Dict[str, Any] = {}

        # Build a per-instance LRU cache for query embeddings.
        # Key = (query_text, chunk_type) → embedding vector.
        @functools.lru_cache(maxsize=_QUERY_EMBED_CACHE_SIZE)
        def _cached_embed(text: str, chunk_type: str) -> Tuple[float, ...]:
            vec = self._embed_fn(text, chunk_type)
            return tuple(vec)  # tuples are hashable → cache-friendly

        self._cached_embed = _cached_embed

    # ------------------------------------------------------------------
    # Query embedding cache helpers
    # ------------------------------------------------------------------

    def clear_embed_cache(self) -> None:
        """Clear the LRU query-embedding cache."""
        self._cached_embed.cache_clear()
        logger.debug("Query embedding cache cleared.")

    def embed_cache_info(self):
        """Return ``CacheInfo(hits, misses, maxsize, currsize)``."""
        return self._cached_embed.cache_info()

    # ------------------------------------------------------------------
    # Lazy client / collection initialisation
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Lazy-initialised ``chromadb.PersistentClient``."""
        if self._client is None:
            _chromadb = _import_chromadb()
            from chromadb.config import Settings as ChromaSettings
            self._client = _chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(
                "ChromaDB client initialised at %s",
                self._config.chroma_persist_dir,
            )
        return self._client

    def _get_collection(self, name: str):
        """Get or create a named collection with HNSW tuning.

        Args:
            name: Collection name (``code_index`` or ``theory_index``).

        Returns:
            A ``chromadb.Collection`` object.
        """
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": self._config.hnsw_m,
                    "hnsw:construction_ef": self._config.hnsw_ef_construction,
                    "hnsw:search_ef": self._config.hnsw_ef_search,
                },
            )
            logger.info(
                "Collection '%s' ready (%d documents) "
                "[HNSW M=%d, ef_construction=%d, ef_search=%d]",
                name,
                self._collections[name].count(),
                self._config.hnsw_m,
                self._config.hnsw_ef_construction,
                self._config.hnsw_ef_search,
            )
        return self._collections[name]

    @property
    def code_collection(self):
        """The ``code_index`` collection."""
        return self._get_collection(CODE_COLLECTION)

    @property
    def theory_collection(self):
        """The ``theory_index`` collection."""
        return self._get_collection(THEORY_COLLECTION)

    # ------------------------------------------------------------------
    # Collection routing
    # ------------------------------------------------------------------

    def _resolve_collection(self, chunk_type: str):
        """Return the collection for a given ``chunk_type``.

        Args:
            chunk_type: ``"code"``, ``"theory"``, or ``"text"``.

        Returns:
            The corresponding ``chromadb.Collection``.

        Raises:
            ValueError: If *chunk_type* is not recognised.
        """
        col_name = _CHUNK_TYPE_MAP.get(chunk_type)
        if col_name is None:
            raise ValueError(
                f"Unknown chunk_type {chunk_type!r}. "
                f"Expected one of {list(_CHUNK_TYPE_MAP)}."
            )
        return self._get_collection(col_name)

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed_batch(
        self, texts: List[str], chunk_type: str
    ) -> List[List[float]]:
        """Embed a batch of texts using the configured embedding function.

        Args:
            texts: Texts to embed.
            chunk_type: Used for potential per-type model routing.

        Returns:
            List of embedding vectors.
        """
        return [self._embed_fn(t, chunk_type) for t in texts]

    # ------------------------------------------------------------------
    # Core API: upsert_chunks
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Insert or update enriched chunks into the appropriate index.

        Each chunk is expected to be a dict with keys:
        - ``"content"`` (str): The chunk text.
        - ``"metadata"`` (dict): Must include ``"chunk_type"``
          (``"code"`` or ``"theory"``).

        IDs are computed as SHA-256 hashes of the content — inserting
        the same content twice is a no-op (idempotent upsert).

        Chunks are grouped by ``chunk_type``, embedded, and batch-
        upserted into the corresponding ChromaDB collection.

        Args:
            chunks: List of enriched chunk dicts (from ``chunking.py``).

        Returns:
            Dict with counts: ``{"code": N, "theory": M, "skipped": S}``.
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty list.")
            return {"code": 0, "theory": 0, "skipped": 0}

        # Group chunks by target collection
        grouped: Dict[str, List[Dict[str, Any]]] = {
            CODE_COLLECTION: [],
            THEORY_COLLECTION: [],
        }
        skipped = 0

        for chunk in chunks:
            content = chunk.get("content", "").strip()
            meta = chunk.get("metadata", {})
            chunk_type = meta.get("chunk_type", "theory")

            if not content:
                skipped += 1
                continue

            col_name = _CHUNK_TYPE_MAP.get(chunk_type)
            if col_name is None:
                logger.warning(
                    "Skipping chunk with unknown chunk_type=%r", chunk_type,
                )
                skipped += 1
                continue

            grouped[col_name].append(chunk)

        stats: Dict[str, int] = {"code": 0, "theory": 0, "skipped": skipped}

        for col_name, col_chunks in grouped.items():
            if not col_chunks:
                continue

            n = self._upsert_to_collection(col_name, col_chunks)
            key = "code" if col_name == CODE_COLLECTION else "theory"
            stats[key] = n

        logger.info(
            "upsert_chunks complete: code=%d, theory=%d, skipped=%d",
            stats["code"], stats["theory"], stats["skipped"],
        )
        return stats

    def _upsert_to_collection(
        self, col_name: str, chunks: List[Dict[str, Any]]
    ) -> int:
        """Upsert a batch of chunks into a single collection.

        Handles batching (ChromaDB has per-call size limits),
        deduplication by ID, and error recovery.

        Args:
            col_name: Target collection name.
            chunks: Chunks to upsert (all same chunk_type).

        Returns:
            Number of documents upserted.
        """
        collection = self._get_collection(col_name)
        chunk_type = "code" if col_name == CODE_COLLECTION else "theory"

        # Build parallel arrays
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for chunk in chunks:
            content = chunk["content"].strip()
            doc_id = _content_id(content)

            # Deduplicate within this batch
            if doc_id in seen_ids:
                logger.debug("Skipping intra-batch duplicate: %s", doc_id)
                continue
            seen_ids.add(doc_id)

            # Prepare metadata (ChromaDB requires flat string / int / float)
            raw_meta = dict(chunk.get("metadata", {}))
            flat_meta = _flatten_metadata(raw_meta)

            ids.append(doc_id)
            documents.append(content)
            metadatas.append(flat_meta)

        if not ids:
            return 0

        # Embed all documents
        # For code chunks, normalise text before embedding for better
        # semantic matching while keeping the original as the stored
        # document (so retrieval returns verbatim code).
        if chunk_type == "code":
            embed_texts = [_normalize_for_embedding(d) for d in documents]
            logger.info(
                "Embedding %d code chunks (normalised) for '%s'...",
                len(embed_texts), col_name,
            )
        else:
            embed_texts = documents
            logger.info(
                "Embedding %d %s chunks for collection '%s'...",
                len(embed_texts), chunk_type, col_name,
            )
        try:
            embeddings = self._embed_batch(embed_texts, chunk_type)
        except NotImplementedError:
            # Fallback: let ChromaDB use its default embedding function
            logger.warning(
                "Embedding function not configured — upserting without "
                "pre-computed embeddings (ChromaDB default will be used).",
            )
            embeddings = None

        # Batch upsert (respect ChromaDB size limits)
        total = len(ids)
        for start in range(0, total, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, total)
            batch_kwargs: Dict[str, Any] = {
                "ids": ids[start:end],
                "documents": documents[start:end],
                "metadatas": metadatas[start:end],
            }
            if embeddings is not None:
                batch_kwargs["embeddings"] = embeddings[start:end]

            try:
                collection.upsert(**batch_kwargs)
                logger.debug(
                    "Upserted batch %d–%d into '%s'.",
                    start, end - 1, col_name,
                )
            except Exception:
                logger.exception(
                    "Failed to upsert batch %d–%d into '%s'.",
                    start, end - 1, col_name,
                )
                raise

        logger.info(
            "Upserted %d documents into '%s' (total now %d).",
            total, col_name, collection.count(),
        )
        return total

    # ------------------------------------------------------------------
    # Core API: query_index
    # ------------------------------------------------------------------

    def query_index(
        self,
        query: str,
        chunk_type: str = "theory",
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform similarity search on the specified index.

        Args:
            query: Natural-language query string.
            chunk_type: ``"code"`` or ``"theory"`` — selects the
                target collection.
            filters: Optional ChromaDB ``where`` filter dict
                (e.g. ``{"pipeline_stage": "risk_management"}``).
            top_k: Number of results to return.
            include: Fields to include in results. Defaults to
                ``["documents", "metadatas", "distances"]``.

        Returns:
            ChromaDB-style results dict::

                {
                    "ids": [[...]], "documents": [[...]],
                    "metadatas": [[...]], "distances": [[...]],
                }
        """
        collection = self._resolve_collection(chunk_type)
        include = include or ["documents", "metadatas", "distances"]

        # Embed the query (LRU-cached — repeated identical queries
        # are free after the first call).
        try:
            cached_vec = self._cached_embed(query, chunk_type)
            query_embedding = list(cached_vec)
            logger.debug(
                "query_index: embed cache info %s",
                self._cached_embed.cache_info(),
            )
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": include,
            }
        except NotImplementedError:
            # Fallback: let ChromaDB embed the query
            logger.warning("Using ChromaDB default embedding for query.")
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k,
                "include": include,
            }

        if filters:
            query_kwargs["where"] = filters

        try:
            results = collection.query(**query_kwargs)
            n_found = len(results.get("ids", [[]])[0])
            logger.info(
                "query_index('%s', type=%s, top_k=%d) → %d results.",
                query[:60], chunk_type, top_k, n_found,
            )
            return results
        except Exception:
            logger.exception(
                "query_index failed for query=%r, chunk_type=%s.",
                query[:60], chunk_type,
            )
            raise

    # ------------------------------------------------------------------
    # Inspection / statistics
    # ------------------------------------------------------------------

    def count(self, chunk_type: Optional[str] = None) -> Dict[str, int]:
        """Return document counts per collection (or one specific type).

        Args:
            chunk_type: If provided, return count only for that type.

        Returns:
            Dict like ``{"code_index": 123, "theory_index": 456}``.
        """
        if chunk_type:
            col = self._resolve_collection(chunk_type)
            name = _CHUNK_TYPE_MAP[chunk_type]
            return {name: col.count()}
        code_n = self.code_collection.count()
        theory_n = self.theory_collection.count()
        return {
            CODE_COLLECTION: code_n,
            THEORY_COLLECTION: theory_n,
            "total": code_n + theory_n,
        }

    def list_sources(self, chunk_type: Optional[str] = None) -> List[str]:
        """Return unique source filenames across collections.

        Args:
            chunk_type: Limit to one collection type.

        Returns:
            Sorted list of source names.
        """
        sources: set = set()
        collections = (
            [self._resolve_collection(chunk_type)]
            if chunk_type
            else [self.code_collection, self.theory_collection]
        )
        for col in collections:
            try:
                result = col.get(include=["metadatas"])
                for meta in result.get("metadatas") or []:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
            except Exception:
                logger.debug("list_sources: collection not yet populated.")
        return sorted(sources)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics for both collections.

        Returns:
            Dict with counts, sources, and config info.
        """
        counts = self.count()
        sources = self.list_sources()
        return {
            "collections": counts,
            "total_documents": counts.get("total", 0),
            "total_sources": len(sources),
            "sources": sources,
            "persist_directory": self._config.chroma_persist_dir,
            "hnsw_m": self._config.hnsw_m,
            "hnsw_ef_construction": self._config.hnsw_ef_construction,
            "hnsw_ef_search": self._config.hnsw_ef_search,
        }

    # ------------------------------------------------------------------
    # Metadata-based retrieval (parent-child support)
    # ------------------------------------------------------------------

    def get_by_metadata(
        self,
        where: Dict[str, Any],
        chunk_type: Optional[str] = None,
        include: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch chunks matching a metadata ``where`` filter.

        This is the primary API for **parent-child retrieval**: given a
        ``parent_id``, retrieve all sibling chunks that share it.

        Args:
            where: ChromaDB ``where`` filter dict, e.g.
                ``{"parent_id": "abc1234567890def"}``.
            chunk_type: Limit the search to ``"code"`` or ``"theory"``.
                If ``None``, **both** collections are searched.
            include: Fields to include in results
                (default ``["documents", "metadatas"]``).

        Returns:
            List of dicts with keys ``"id"``, ``"content"``,
            ``"metadata"``.  Ordered by ``section_id`` when present.

        Example::

            siblings = store.get_by_metadata(
                where={"parent_id": "a1b2c3d4e5f60718"},
                chunk_type="theory",
            )
        """
        if include is None:
            include = ["documents", "metadatas"]

        collections: List[Tuple[str, Any]] = []
        if chunk_type:
            collections.append((chunk_type, self._resolve_collection(chunk_type)))
        else:
            collections.append(("code", self.code_collection))
            collections.append(("theory", self.theory_collection))

        results: List[Dict[str, Any]] = []

        for ctype, col in collections:
            try:
                raw = col.get(where=where, include=include)
            except Exception:
                logger.debug(
                    "get_by_metadata: query failed on %s collection.", ctype,
                )
                continue

            ids = raw.get("ids") or []
            docs = raw.get("documents") or [None] * len(ids)
            metas = raw.get("metadatas") or [{}] * len(ids)

            for doc_id, doc, meta in zip(ids, docs, metas):
                results.append({
                    "id": doc_id,
                    "content": doc or "",
                    "metadata": meta or {},
                })

        # Sort by section_id so siblings are in document order
        results.sort(
            key=lambda r: (
                r.get("metadata", {}).get("section_id", 0)
                if isinstance(r.get("metadata", {}).get("section_id"), (int, float))
                else 0
            )
        )

        return results

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_by_source(
        self, source_name: str, chunk_type: Optional[str] = None
    ) -> int:
        """Delete all chunks from a given source.

        Args:
            source_name: Source filename to remove.
            chunk_type: Limit deletion to one collection.

        Returns:
            Total number of documents deleted.
        """
        total_deleted = 0
        collections = (
            [(chunk_type, self._resolve_collection(chunk_type))]
            if chunk_type
            else [
                ("code", self.code_collection),
                ("theory", self.theory_collection),
            ]
        )
        for ctype, col in collections:
            try:
                results = col.get(
                    where={"source": source_name},
                    include=[],
                )
                ids = results.get("ids") or []
                if ids:
                    col.delete(ids=ids)
                    total_deleted += len(ids)
                    logger.info(
                        "Deleted %d chunks from '%s' (%s).",
                        len(ids), source_name, ctype,
                    )
            except Exception:
                logger.debug(
                    "delete_by_source: no results in %s for '%s'.",
                    ctype, source_name,
                )
        return total_deleted

    def delete_by_ids(
        self, ids: List[str], chunk_type: Optional[str] = None
    ) -> None:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
            chunk_type: Target collection. If ``None``, tries both.
        """
        if not ids:
            return
        collections = (
            [self._resolve_collection(chunk_type)]
            if chunk_type
            else [self.code_collection, self.theory_collection]
        )
        for col in collections:
            try:
                col.delete(ids=ids)
            except Exception:
                pass
        logger.info("Deleted %d documents by ID.", len(ids))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self, chunk_type: Optional[str] = None) -> None:
        """Drop and recreate collection(s) (destructive).

        Args:
            chunk_type: Reset only one collection; ``None`` resets both.
        """
        names = (
            [_CHUNK_TYPE_MAP[chunk_type]]
            if chunk_type
            else [CODE_COLLECTION, THEORY_COLLECTION]
        )
        for name in names:
            try:
                self.client.delete_collection(name)
                self._collections.pop(name, None)
                logger.warning("Collection '%s' dropped.", name)
            except Exception:
                logger.debug("Collection '%s' did not exist.", name)
            # Re-create immediately
            self._get_collection(name)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Metadata flattening (ChromaDB requires flat types)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 5b. Pre-embedding normalisation for code chunks
# ═══════════════════════════════════════════════════════════════════════════

import re as _re

_TRAILING_COMMENT_RE = _re.compile(r"\s+#[^\n]*$", _re.MULTILINE)
_EXCESS_BLANK_RE = _re.compile(r"\n{3,}")
_DEF_FUNC_NAME_RE = _re.compile(r"(\bdef\s+)(\w+)")


def _normalize_for_embedding(text: str) -> str:
    """Normalise code text for embedding while preserving semantics.

    Transformations (applied in order):

    1. **Remove trailing comments** — inline ``# …`` after code.
    2. **Normalise whitespace** — collapse 3+ consecutive blank lines
       to 2; strip leading/trailing whitespace on each line.
    3. **Consistent indentation** — convert tabs to 4 spaces.
    4. **Lowercase function names** in ``def`` statements so that
       ``applyPtSlOnT1`` and ``applyptslont1`` produce the same
       embedding neighbourhood.

    The **original** (untouched) text should still be stored as the
    ``document`` in ChromaDB so that retrieval returns verbatim code.
    Only the *embedding vector* is computed from this normalised form.

    Returns:
        Normalised text suitable for embedding.
    """
    # 1. Remove trailing comments
    text = _TRAILING_COMMENT_RE.sub("", text)

    # 2. Tab → 4 spaces
    text = text.expandtabs(4)

    # 3. Strip trailing whitespace per line, collapse excess blanks
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _EXCESS_BLANK_RE.sub("\n\n", text)

    # 4. Lowercase function names in def statements
    text = _DEF_FUNC_NAME_RE.sub(
        lambda m: m.group(1) + m.group(2).lower(), text
    )

    return text.strip()


def _flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten metadata values to types ChromaDB accepts.

    ChromaDB metadata values must be ``str``, ``int``, ``float``, or
    ``bool``.  Lists are joined with ``", "``; nested dicts are
    JSON-serialised; ``None`` is replaced with ``""``.

    Args:
        meta: Raw metadata dict.

    Returns:
        Flat metadata dict safe for ChromaDB.
    """
    flat: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            flat[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            flat[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            import json
            flat[key] = json.dumps(value, default=str)
        else:
            flat[key] = str(value)
    return flat


# ═══════════════════════════════════════════════════════════════════════════
# 6. Module-level convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def upsert_chunks(
    chunks: List[Dict[str, Any]],
    config: Optional[RAGConfig] = None,
    embed_fn: Optional[EmbeddingFn] = None,
) -> Dict[str, int]:
    """Module-level convenience for one-shot chunk upsert.

    Creates a ``DualIndexStore``, upserts, and returns stats.

    Args:
        chunks: Enriched chunk dicts from ``chunking.py``.
        config: Optional ``RAGConfig``.
        embed_fn: Optional embedding function.

    Returns:
        Upsert stats dict.
    """
    store = DualIndexStore(config=config, embed_fn=embed_fn)
    return store.upsert_chunks(chunks)


def query_index(
    query: str,
    chunk_type: str = "theory",
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    config: Optional[RAGConfig] = None,
    embed_fn: Optional[EmbeddingFn] = None,
) -> Dict[str, Any]:
    """Module-level convenience for one-shot query.

    Args:
        query: Search query.
        chunk_type: ``"code"`` or ``"theory"``.
        filters: Optional metadata filters.
        top_k: Number of results.
        config: Optional ``RAGConfig``.
        embed_fn: Optional embedding function.

    Returns:
        ChromaDB results dict.
    """
    store = DualIndexStore(config=config, embed_fn=embed_fn)
    return store.query_index(
        query=query,
        chunk_type=chunk_type,
        filters=filters,
        top_k=top_k,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. VectorStoreManager — BACKWARD COMPATIBLE legacy wrapper
# ═══════════════════════════════════════════════════════════════════════════
#
# The rest of the codebase imports VectorStoreManager extensively.
# This class is preserved with its exact original API, but now
# internally delegates to the same ChromaDB client.
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """Manages a persistent ChromaDB collection for the RAG pipeline.

    **Backward-compatible** — this is the original single-collection
    manager used by ``PDFIngestionService``, ``RAGQueryEngine``,
    ``HybridSearcher``, and other pipeline components.

    For the new dual-index architecture, use ``DualIndexStore``.

    Usage::

        config = RAGConfig()
        vs = VectorStoreManager(config)
        vs.add_documents(ids, texts, metadatas, embeddings)
        results = vs.query(query_embeddings, n_results=5)
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self._config = config or RAGConfig()
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            _chromadb = _import_chromadb()
            from chromadb.config import Settings as ChromaSettings
            self._client = _chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(
                "ChromaDB client initialised at %s",
                self._config.chroma_persist_dir,
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._config.chroma_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": self._config.hnsw_m,
                    "hnsw:construction_ef": self._config.hnsw_ef_construction,
                    "hnsw:search_ef": self._config.hnsw_ef_search,
                },
            )
            logger.info(
                "Collection '%s' ready (%d documents) "
                "[HNSW M=%d, ef_construction=%d, ef_search=%d]",
                self._config.chroma_collection_name,
                self._collection.count(),
                self._config.hnsw_m,
                self._config.hnsw_ef_construction,
                self._config.hnsw_ef_search,
            )
        return self._collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Insert or upsert documents into the collection."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d documents into collection", len(ids))

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform similarity search.

        Returns ChromaDB-style results dict with keys:
            ids, documents, metadatas, distances
        """
        include = include or ["documents", "metadatas", "distances"]
        kwargs: Dict[str, Any] = {
            "n_results": n_results,
            "include": include,
        }
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def delete_by_metadata(
        self, where: Dict[str, Any]
    ) -> None:
        """Delete documents matching a metadata filter."""
        results = self.collection.get(where=where)
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(
                "Deleted %d documents matching %s", len(results["ids"]), where
            )

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info("Deleted %d documents by ID", len(ids))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return total document count in the collection."""
        return self.collection.count()

    def source_exists(self, source_name: str) -> bool:
        """Check whether any chunks from *source_name* are stored."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=[],
            )
            return bool(results.get("ids"))
        except Exception:
            return False

    def get_source_file_hash(self, source_name: str) -> Optional[str]:
        """Return the ``file_hash`` stored for *source_name*, or None."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            metas = results.get("metadatas") or []
            if metas and metas[0]:
                return metas[0].get("file_hash")
        except Exception:
            pass
        return None

    def get_source_chunk_count(self, source_name: str) -> int:
        """Return the number of chunks stored for *source_name*."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=[],
            )
            return len(results.get("ids") or [])
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        """Return unique source filenames stored in the collection."""
        all_meta = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in (all_meta.get("metadatas") or []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the collection."""
        count_val = self.count()
        sources = self.list_sources()
        return {
            "collection_name": self._config.chroma_collection_name,
            "total_documents": count_val,
            "total_sources": len(sources),
            "sources": sources,
            "persist_directory": self._config.chroma_persist_dir,
            "hnsw_m": self._config.hnsw_m,
            "hnsw_ef_construction": self._config.hnsw_ef_construction,
            "hnsw_ef_search": self._config.hnsw_ef_search,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset_collection(self) -> None:
        """Drop and recreate the collection (destructive)."""
        self.client.delete_collection(self._config.chroma_collection_name)
        self._collection = None
        logger.warning(
            "Collection '%s' has been reset",
            self._config.chroma_collection_name,
        )
        _ = self.collection
