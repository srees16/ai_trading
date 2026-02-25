"""
PDF Ingestion Service for Centurion Capital LLC RAG Pipeline.

Responsibilities:
    1. Parse PDF files to plain text (via PyMuPDF / fitz)
    2. Split text into overlapping chunks
    3. Embed chunks
    4. Store in ChromaDB via VectorStoreManager
"""

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text chunking (zero external dependency)
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    # Collapse multiple spaces / tabs
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[str]:
    """
    Split *text* into overlapping chunks by character count.

    Tries to break on sentence boundaries (`.`, `!`, `?`) when possible.
    """
    text = _clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to land on a sentence boundary
        if end < length:
            # Look back from `end` for a sentence-ending punctuation
            lookback = text[start:end]
            for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                last = lookback.rfind(sep)
                if last != -1 and last > chunk_size // 4:
                    end = start + last + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = max(start + 1, end - chunk_overlap)

    return chunks


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract all text from a PDF file.

    Returns:
        (full_text, metadata_dict)
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    for page in doc:
        pages_text.append(page.get_text("text"))

    full_text = "\n\n".join(pages_text)
    metadata = {
        "source": os.path.basename(pdf_path),
        "full_path": str(Path(pdf_path).resolve()),
        "page_count": len(doc),
        "title": doc.metadata.get("title", "") or os.path.basename(pdf_path),
        "author": doc.metadata.get("author", ""),
    }
    doc.close()
    return full_text, metadata


# ---------------------------------------------------------------------------
# Ingestion service
# ---------------------------------------------------------------------------

class PDFIngestionService:
    """
    End-to-end PDF → ChromaDB ingestion.

    Usage:
        svc = PDFIngestionService(vector_store, config)
        stats = svc.ingest_pdf("/path/to/file.pdf")
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: Optional[RAGConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self._vs = vector_store
        self._config = config or RAGConfig()
        self._embedder = embedding_service or EmbeddingService(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_pdf(
        self,
        pdf_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse, chunk, embed, and store a single PDF.

        Returns a summary dict with ingestion statistics.
        """
        logger.info("Ingesting PDF: %s", pdf_path)

        # 1. Extract text
        full_text, file_meta = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            logger.warning("PDF contains no extractable text: %s", pdf_path)
            return {"status": "skipped", "reason": "empty", "source": pdf_path}

        # 2. Chunk
        chunks = chunk_text(
            full_text,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

        # 3. Build IDs & metadata
        source_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{source_hash}_{idx:04d}"
            meta = {
                "source": file_meta["source"],
                "page_count": file_meta["page_count"],
                "title": file_meta["title"],
                "author": file_meta["author"],
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            if extra_metadata:
                meta.update(extra_metadata)
            ids.append(chunk_id)
            metadatas.append(meta)

        # 4. Embed
        embeddings = self._embedder.embed_texts(chunks)

        # 5. Store
        self._vs.add_documents(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        stats = {
            "status": "success",
            "source": file_meta["source"],
            "pages": file_meta["page_count"],
            "chunks": len(chunks),
            "collection_total": self._vs.count(),
        }
        logger.info("Ingestion complete: %s", stats)
        return stats

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Ingest all PDFs in a directory."""
        pattern = "**/*.pdf" if recursive else "*.pdf"
        results: List[Dict[str, Any]] = []
        for pdf_file in Path(directory).glob(pattern):
            result = self.ingest_pdf(str(pdf_file))
            results.append(result)
        return results

    def ingest_uploaded_bytes(
        self,
        file_name: str,
        file_bytes: bytes,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a PDF from raw bytes (Streamlit file uploader).

        Saves to ``pdf_upload_dir`` then delegates to ``ingest_pdf``.
        """
        save_dir = Path(self._config.pdf_upload_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name

        save_path.write_bytes(file_bytes)
        logger.info("Saved uploaded file to %s", save_path)

        return self.ingest_pdf(str(save_path), extra_metadata=extra_metadata)

    def delete_source(self, source_name: str) -> int:
        """Remove all chunks belonging to a given source PDF."""
        results = self._vs.collection.get(
            where={"source": source_name},
        )
        count = len(results["ids"]) if results["ids"] else 0
        if count:
            self._vs.delete_by_ids(results["ids"])
        return count
