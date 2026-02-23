# RAG Pipeline — Centurion Capital LLC

Ingest trading strategy PDFs and get context-aware, LLM-powered answers — runs 100% locally, no API keys required.

---

## Architecture

**Two-stage retrieval + LLM generation:**

```
PDF Upload → Parse (PyMuPDF) → Chunk (1000 chars) → Embed (MiniLM-L6) → ChromaDB
                                                                            │
User Query → Embed Query ──────────────────────────────────────────────────▶ │
                                                                            ▼
                                                              Stage 1: Top-20 cosine retrieval
                                                                            │
                                                              Stage 2: Cross-encoder re-rank → Top 5
                                                                            │
                                                              Ollama LLM (Mistral 7B) → Grounded answer
```

**Design:** Modular protocol/interface pattern — swap any component (embeddings, re-ranker, LLM, vector store) without touching the rest. Lazy model loading. Streamlit session-state singletons.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama & pull model
#    Download from https://ollama.com/download, then:
ollama pull mistral

# 3. Run (pick one)
streamlit run app.py                      # as part of main app (navigate to RAG page)
streamlit run rag_pipeline/rag_page.py    # standalone mode
```

**Programmatic usage:**

```python
from rag_pipeline import RAGConfig, VectorStoreManager, PDFIngestionService, RAGQueryEngine

config = RAGConfig()
vs = VectorStoreManager(config)
PDFIngestionService(vs, config).ingest_pdf("path/to/strategy.pdf")

response = RAGQueryEngine(vs, config).query("What are the entry rules?")
print(response.answer)
```

---

## Modules

| Module | Purpose |
|---|---|
| `config.py` | `RAGConfig` dataclass — all settings with env var overrides |
| `vector_store.py` | ChromaDB wrapper (CRUD, query, stats, reset) |
| `embeddings.py` | Sentence-transformers embeddings (swappable backend) |
| `pdf_ingestion.py` | PDF → text → chunks → embeddings → ChromaDB |
| `query_engine.py` | RAG orchestrator — embed → retrieve → re-rank → LLM |
| `reranker.py` | Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) |
| `llm_service.py` | Ollama LLM backend with RAG-grounded system prompts |
| `ui_components.py` | Reusable Streamlit widgets (toggle, uploader, query, KB) |
| `rag_page.py` | Standalone Streamlit page / main app route |

---

## Dependencies

| Component | Size | Purpose |
|---|---|---|
| `chromadb` | — | Persistent vector DB (HNSW + cosine) |
| `sentence-transformers` | — | Bi-encoder + cross-encoder models |
| `PyMuPDF` | — | PDF text extraction |
| `all-MiniLM-L6-v2` | ~90 MB | Embedding model (384-dim, auto-downloaded) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | Re-ranker model (auto-downloaded) |
| **Ollama** + `mistral` | ~4.1 GB | Local LLM server + model |

---

## Configuration

All settings live in `RAGConfig` and can be overridden via `CENTURION_RAG_*` environment variables.

**Key settings:**

| Setting | Default | Env Variable |
|---|---|---|
| Chunk size | `1000` chars | `CENTURION_RAG_CHUNK_SIZE` |
| Chunk overlap | `200` chars | `CENTURION_RAG_CHUNK_OVERLAP` |
| Retrieval top-k | `20` | `CENTURION_RAG_TOP_K` |
| Re-rank top N | `5` | `CENTURION_RAG_RERANK_TOP_N` |
| LLM model | `mistral` | `CENTURION_RAG_LLM_MODEL` |
| LLM timeout | `600` sec | `CENTURION_RAG_LLM_TIMEOUT` |
| ChromaDB dir | `data/chroma_db/` | `CENTURION_RAG_CHROMA_DIR` |

See `config.py` for the full list of 20+ configurable fields.

```bash
# Example: switch to Llama 3 with higher token limit
set CENTURION_RAG_LLM_MODEL=llama3
set CENTURION_RAG_LLM_MAX_TOKENS=2048
```

---

## Directory Structure

```
rag_pipeline/
├── config.py            # Centralized configuration
├── vector_store.py      # ChromaDB wrapper
├── embeddings.py        # Embedding service
├── pdf_ingestion.py     # PDF parsing & chunking
├── query_engine.py      # RAG orchestrator
├── reranker.py          # Cross-encoder re-ranking
├── llm_service.py       # Ollama LLM backend
├── ui_components.py     # Streamlit widgets
├── rag_page.py          # Streamlit page entry point
└── __init__.py          # Package exports & logging

data/                    # Auto-created at project root
├── chroma_db/           # ChromaDB persistent storage
└── rag_uploads/         # Uploaded PDF files
```

---

## Extending

Every component uses a **protocol/interface** — implement the protocol and pass your backend in:

| Swap | Protocol | Example |
|---|---|---|
| Embeddings | `EmbeddingBackend.embed(texts) → vectors` | OpenAI, Cohere |
| Re-ranker | `RerankerBackend.rerank(query, texts, top_n) → indices` | Cohere Rerank |
| LLM | `LLMBackend.generate(query, context) → answer` | OpenAI, Anthropic |

```python
# Example: custom LLM
engine = RAGQueryEngine(vector_store, llm_backend=MyCustomLLMBackend())
```

For new file types (DOCX, CSV, etc.), create an ingestion class that extracts text → `chunk_text()` → `EmbeddingService.embed_texts()` → `VectorStoreManager.add_documents()`.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| **Cannot connect to Ollama** | Ensure Ollama is running: `ollama serve`. Windows path: `%LOCALAPPDATA%\Programs\Ollama\` |
| **LLM timeout** | First query after restart loads model into memory (can take minutes). Increase: `CENTURION_RAG_LLM_TIMEOUT=900` |
| **Model not found** | Pull it: `ollama pull mistral`. List available: `ollama list` |
| **No results** | Ensure PDFs are uploaded & ingested. Check KB stats in the UI. |
| **Import errors** | Run from project root: `cd centurion_core && streamlit run rag_pipeline/rag_page.py` |
| **Slow first run** | Embedding (~90 MB) and re-ranker (~80 MB) models download on first use. Cached after that. |
