# Centurion Core — Module Documentation

> Auto-generated structured documentation for **kite_connect/**, **rag_pipeline/**, and **trading_strategies/** modules.

---

## Table of Contents

1. [kite_connect/ Module](#1-kite_connect-module)
2. [rag_pipeline/ Module](#2-rag_pipeline-module)
3. [trading_strategies/ Module](#3-trading_strategies-module)

---

## 1. kite_connect/ Module

Zerodha Kite Connect integration for live Indian stock market monitoring, order placement, option chains, NSE data scraping, and automated RSI-based trading — all orchestrated through a Streamlit web UI.

### 1.1 `kite_connect/zerodha_live.py` (~1326 lines)

**Purpose:** Main Streamlit application for live Indian equity monitoring, order management, holdings/positions display, option chain analysis, and RSI strategy scanning.

| Symbol | Type | Description |
|--------|------|-------------|
| `REFRESH_INTERVAL` | Constant | Auto-refresh interval (30 seconds) |
| `get_kite_session()` | Function | Returns cached authenticated `KiteConnect` instance via `create_kite_session()` |
| `get_db_connection()` | Function | Returns PostgreSQL connection to `livestocks_ind` database |
| `fetch_index_groups()` | Function | Queries `index_groups` table for watchlist segment names |
| `fetch_stock_names_by_index(conn, group)` | Function | Retrieves stock symbols for a given index group from `index_stocks` join |
| `update_stocks_in_db(conn, stocks)` | Function | Bulk-upserts stock data (LTP, change %, volume, OHLC) into `stocks` table |
| `fetch_stocks_from_db(conn, symbols)` | Function | Reads latest stock rows from DB, returns as DataFrame |
| `_parse_quote(raw)` | Function | Normalises a single Kite quote dict into flat dict with `ltp`, `change`, `volume`, etc. |
| `fetch_realtime_quotes(kite, symbols)` | Function | Batch-fetches quotes (200/batch) from Kite, parses each via `_parse_quote()` |
| `_render_option_chain_tab(kite)` | Function | Renders full option chain UI — index picker, expiry discovery, Sensibull-style coloring, ATM highlighting, PCR metric, Quick Option Trade expander |
| `main()` | Function | Entry point — page config, sidebar (right-side), two top-level tabs ("📈 Stocks" / "🔗 Options"), sub-tabs for Order Book / Positions / Holdings / RSI Strategy |

**Key features:**
- `@st.fragment(run_every=timedelta(seconds=REFRESH_INTERVAL))` for auto-refresh
- Market status pill indicators from NSE API
- Color-coded change % and PnL
- Smallcase vs. Kite holdings detection with JSON cache
- Custom enterprise CSS styling

---

### 1.2 `kite_connect/auth/kite_auth.py` (~257 lines)

**Purpose:** Automates the Kite Connect OAuth login flow to capture `request_token` via a local HTTP callback server and Selenium browser automation.

| Symbol | Type | Description |
|--------|------|-------------|
| `captured_token` | Global | Stores the request token captured from redirect callback |
| `CallbackHandler` | Class (`BaseHTTPRequestHandler`) | Handles GET redirect from Kite, extracts `request_token` query param |
| `update_kite_app(token)` | Function | Regex-replaces `request_token` value in `kite_token_store.py` on disk |
| `fetch_request_token()` | Function | Main flow: starts HTTP server on `127.0.0.1:5000`, launches Selenium for auto-fill (user ID + password), waits for TOTP/2FA, captures token on redirect. Falls back to manual browser if Selenium unavailable. |

---

### 1.3 `kite_connect/auth/kite_session.py` (~65 lines)

**Purpose:** Shared Kite Connect session management — creates a single reusable authenticated session.

| Symbol | Type | Description |
|--------|------|-------------|
| `_read_request_token()` | Function | Reads current `request_token` from `kite_token_store.py` via module import |
| `create_kite_session()` | Function | Creates `KiteConnect` instance, generates access token, auto-launches `fetch_request_token()` if token is expired. Returns authenticated `KiteConnect` object. |

---

### 1.4 `kite_connect/auth/kite_token_store.py` (~50 lines)

**Purpose:** Stores the active `request_token` (updated in-place by `kite_auth.py` via regex) and provides convenience login/instrument helpers.

| Symbol | Type | Description |
|--------|------|-------------|
| `request_token` | Constant | Current OAuth request token (dynamically updated) |
| `zerodha_login()` | Function | Authenticates with API key/secret and returns user profile dict |
| `add_stocks()` | Function | Fetches all NSE equity instruments from Kite, filters out indices/SDLs/T-bills, returns list of `tradingsymbol` names |

---

### 1.5 `kite_connect/core/config.py` (~50 lines)

**Purpose:** Centralized configuration constants for all kite_connect sub-modules.

| Symbol | Type | Description |
|--------|------|-------------|
| `DB_HOST` | Constant | `"localhost"` |
| `DB_PORT` | Constant | `5432` |
| `DB_USER` | Constant | `"postgres"` |
| `DB_PASSWORD` | Constant | `"superadmin1"` |
| `DB_NAME` | Constant | `"livestocks_ind"` |
| `TABLE_NAME` | Constant | `"stocks"` |
| `API_KEY`, `API_SECRET` | Constants | Zerodha API credentials |
| `LOGIN_URL` | Constant | Kite Connect login URL |
| `ZERODHA_USER_ID`, `ZERODHA_PASSWORD` | Constants | Zerodha account credentials |
| `NSE_URL` | Constant | NSE live equity market data URL |
| `DOWNLOAD_DIR` | Constant | Directory for NSE CSV downloads |
| `INDEX_GROUPS` | Constant | `["NIFTY50", "NIFTYBANK", "NIFTYIT", "NIFTYENERGY"]` |
| `REFRESH_INTERVAL` | Constant | `30` (seconds) |
| `KITE_APP_FILE` | Constant | Path to `kite_token_store.py` for in-place token updates |

---

### 1.6 `kite_connect/core/db_service.py` (~35 lines)

**Purpose:** Shared PostgreSQL connection factory.

| Symbol | Type | Description |
|--------|------|-------------|
| `get_connection(dbname=None)` | Function | Returns `psycopg2` connection; defaults to `livestocks_ind` DB using config constants |

---

### 1.7 `kite_connect/core/selenium_service.py` (~130 lines)

**Purpose:** Shared Selenium browser utilities — creates Chrome or Edge WebDriver with configurable options.

| Symbol | Type | Description |
|--------|------|-------------|
| `get_driver(download_dir=None, stealth=False)` | Function | Creates WebDriver (Chrome first, Edge fallback). Optional stealth mode (anti-bot user-agent, hidden webdriver flag). |
| `_common_options()` | Function | Shared browser options (headless, no-sandbox, etc.) |
| `_chrome_options()` | Function | Chrome-specific options + optional webdriver-manager |
| `_edge_options()` | Function | Edge-specific options |

---

### 1.8 `kite_connect/nse/nse_db_loader.py` (~115 lines)

**Purpose:** Loads downloaded NSE CSV files (`MW-*.csv`) into the PostgreSQL `stocks` table.

| Symbol | Type | Description |
|--------|------|-------------|
| `find_latest_csv()` | Function | Scans download directory for most recent `MW-*.csv` |
| `clean_numeric(val)` | Function | Strips commas and converts to float |
| `clean_volume(val)` | Function | Cleans volume strings to int |
| `load_csv_to_db()` | Function | Maps CSV columns (SYMBOL→name, HIGH→high, LOW→low, VOLUME→volume, LTP→ltp, %CHNG→change) via fuzzy column matching; truncates table before insert |

---

### 1.9 `kite_connect/nse/nse_downloader.py` (~165 lines)

**Purpose:** Downloads live equity market data CSV from NSE India using Selenium browser automation.

| Symbol | Type | Description |
|--------|------|-------------|
| `wait_for_download(directory, prefix, timeout)` | Function | Polls for completed download file (`.csv` not `.crdownload`) |
| `download_nse_csv()` | Function | 5-step process: establish NSE session → open live equity page → wait for table → find/click download button (multiple CSS selector strategies) → verify download |

---

### 1.10 `kite_connect/options/option_chain.py` (~262 lines)

**Purpose:** Option chain service for NIFTY/BANKNIFTY — fetches live CE/PE data (LTP, OI, OI Change, Volume).

| Symbol | Type | Description |
|--------|------|-------------|
| `INDEX_META` | Constant | Dict mapping `"NIFTY"` (step=50) and `"BANKNIFTY"` (step=100) with quote_key and prefix |
| `discover_expiries(kite, index)` | Function | Probes next 45 days + monthly expiries for valid NFO contracts; returns codes like `"2602D"`, `"26FEB"` |
| `fetch_option_chain(kite, index, expiry_code, num_strikes, timeframe)` | Function | Returns dict with `spot`, `atm_strike`, `step`, `strikes` (per-strike: `ce_ltp`, `ce_oi`, `ce_oi_chg`, `pe_ltp`, `pe_oi`, `pe_oi_chg`, `ce_volume`, `pe_volume`, `ce_change`, `pe_change`) |

**Key features:**
- `ThreadPoolExecutor` (max 20 workers) for parallel OI-change fetching via `historical_data` API

---

### 1.11 `kite_connect/setup/db_setup.py` (~135 lines)

**Purpose:** Database bootstrapping — creates `livestocks_ind` database and its schema.

| Symbol | Type | Description |
|--------|------|-------------|
| `create_database()` | Function | Creates `livestocks_ind` DB if it doesn't exist |
| `create_table()` | Function | Creates `stocks`, `index_groups`, and `index_stocks` tables |
| `populate_index_groups()` | Function | Inserts `INDEX_GROUPS` and distributes stocks across groups |

---

### 1.12 `kite_connect/trading/order_service.py` (~105 lines)

**Purpose:** Order placement service for Zerodha Kite Connect.

| Symbol | Type | Description |
|--------|------|-------------|
| `place_order(kite, symbol, exchange, transaction_type, quantity, order_type, product, price, trigger_price, validity)` | Function | Returns `{success, order_id/error}`. Supports MARKET, LIMIT, SL, SL-M orders; CNC, MIS, NRML products; DAY, IOC validity. |
| `get_order_book(kite)` | Function | Returns list of today's orders |
| `get_positions(kite)` | Function | Returns net and day positions |
| `get_holdings(kite)` | Function | Returns portfolio holdings |
| `cancel_order(kite, order_id, variety)` | Function | Cancels a pending order |

---

### 1.13 `kite_connect/trading/rsi_strategy.py` (~364 lines)

**Purpose:** RSI-based auto-order strategy — scans a watchlist, calculates 14-period RSI on 5-minute candles, generates BUY (RSI<30 + bullish reversal) and SELL (RSI>70 + bearish reversal) signals.

| Symbol | Type | Description |
|--------|------|-------------|
| `calculate_rsi(candles, period=14)` | Function | Wilder smoothing RSI calculation |
| `_rsi_from_avgs(avg_gain, avg_loss)` | Function | RSI formula helper |
| `detect_signal(candles, rsi_low, rsi_high)` | Function | Returns `{rsi, signal, close, prev_close}` — `BUY`/`SELL`/`HOLD` |
| `compute_sl_and_qty(kite, symbol, side, capital, max_loss)` | Function | Derives quantity and stop-loss from capital/max-loss constraints with margin check |
| `place_strategy_order(kite, symbol, side, capital, max_loss, order_type)` | Function | Places Cover Order (CO/MIS) with auto-calculated stop-loss |
| `scan_watchlist(kite, symbols, capital, max_loss, order_limit, order_type, rsi_low, rsi_high, interval, lookback_days, auto_place)` | Function | Batch scanner returning per-symbol RSI / signal / order results |

---

## 2. rag_pipeline/ Module

Retrieval-Augmented Generation pipeline with PDF ingestion, ChromaDB vector storage, BM25 hybrid search, cross-encoder re-ranking, multi-provider LLM generation (Ollama/Claude/OpenAI), semantic caching, query rewriting, and a full Streamlit-based UI.

### 2.1 `rag_pipeline/config.py` (~429 lines)

**Purpose:** Centralized RAG configuration via a `@dataclass` with environment variable overrides (`CENTURION_RAG_*` prefix).

| Symbol | Type | Description |
|--------|------|-------------|
| `RAGConfig` | Dataclass | ~60+ fields covering every pipeline stage |
| `ensure_directories()` | Method | Creates required directories on startup |

**Key configuration groups:**
- **ChromaDB:** `persist_dir`, `collection_name`, HNSW params (M=32, ef_construction=200, ef_search=150)
- **Embedding:** model=`BAAI/bge-base-en-v1.5`, dim=768, `query_prefix`
- **Chunking:** token-based, size=512, overlap=128, min=30, max=800
- **Retrieval:** top_k=25, similarity_threshold=0.45
- **Query Rewriting:** enabled, n=3, HyDE enabled
- **Hybrid Search:** BM25 weight=0.4, vector weight=0.6
- **Context:** max_chunks=5, dedup_sim=0.92, token_budget=4000
- **Re-ranking:** cross-encoder/ms-marco-MiniLM-L-6-v2, top_n=5, score_threshold=0.25
- **Semantic Cache:** disabled by default, threshold=0.95, TTL=3600s, max=256
- **LLM:** provider=ollama, model=mistral, Claude=claude-sonnet-4-20250514, OpenAI=gpt-4o

---

### 2.2 `rag_pipeline/embeddings.py` (~120 lines)

**Purpose:** Embedding service wrapping `sentence-transformers`.

| Symbol | Type | Description |
|--------|------|-------------|
| `EmbeddingBackend` | Protocol | `embed(texts)`, `embed_query(text)` |
| `SentenceTransformerBackend` | Class | Lazy model loading, L2-normalised embeddings, query prefix support |
| `EmbeddingService` | Class | High-level service delegating to backend; `embed_texts()`, `embed_query()` |

---

### 2.3 `rag_pipeline/evaluation.py` (~583 lines)

**Purpose:** Retrieval evaluation framework with offline metrics and LLM-as-judge faithfulness scoring.

| Symbol | Type | Description |
|--------|------|-------------|
| `EvalQuery` | Dataclass | Single evaluation query with expected IDs/sources |
| `EvalDataset` | Dataclass | Collection of `EvalQuery` items with `save()`/`load()` JSON serialisation |
| `QueryEvalResult` | Dataclass | Per-query evaluation result |
| `EvalReport` | Dataclass | Aggregated results with `.summary()` |
| `FaithfulnessResult` | Dataclass | LLM-judged faithfulness + relevance scores (1–5) |
| `recall_at_k()`, `precision_at_k()`, `hit_rate_at_k()`, `reciprocal_rank()`, `ndcg_at_k()` | Functions | Standard IR metrics |
| `RetrievalLogger` | Class | JSONL logging for offline analysis |
| `run_evaluation(query_engine, dataset, top_k)` | Function | Batch evaluation |
| `evaluate_faithfulness(query, answer, context, llm_backend)` | Function | LLM-as-judge scoring |
| `create_golden_eval_dataset()` | Function | 12 starter queries (factual, procedural, comparative, analytical, summary) |

---

### 2.4 `rag_pipeline/hybrid_search.py` (~396 lines)

**Purpose:** Hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF).

| Symbol | Type | Description |
|--------|------|-------------|
| `BM25Index` | Class | In-memory BM25 index (k1=1.5, b=0.75), stopword removal, standard BM25 scoring |
| `HybridSearcher` | Class | Lazy BM25 rebuild, adaptive weight tracking, metadata filter compliance; `search()` method combines vector + BM25 via RRF |
| `_tokenize(text)` | Function | Lowercased whitespace tokeniser with stopword removal |
| `reciprocal_rank_fusion(*ranked_lists, k=60)` | Function | Merges ranked lists using RRF formula |
| `_matches_where(meta, where)` | Method | Client-side ChromaDB-style filter evaluation (`$and`/`$or`/`$gte`/`$lte`/etc.) |

---

### 2.5 `rag_pipeline/llm_service.py` (~844 lines)

**Purpose:** Multi-provider LLM service (Ollama, Claude, OpenAI) with streaming, retry logic, and fallback chains.

| Symbol | Type | Description |
|--------|------|-------------|
| `RAG_SYSTEM_PROMPT` | Constant | Detailed grounding/citation/code reproduction rules |
| `NO_CONTEXT_SYSTEM_PROMPT` | Constant | Prompt for queries with no retrieved context |
| `_build_user_message(query, context)` | Function | Formats query + context into user message |
| `_pick_system_prompt(context)` | Function | Selects system prompt based on context presence |
| `OllamaLLMBackend` | Class | Persistent `requests.Session`, Ollama chat API, `generate()`, `generate_stream()` (iter_lines), `is_available()`, `list_models()` |
| `ClaudeLLMBackend` | Class | Anthropic Messages API, 3× retry with exponential backoff, streaming via `messages.stream()` |
| `OpenAILLMBackend` | Class | Chat Completions API, 3× retry, quota/billing error detection, streaming |
| `_FallbackChainBackend` | Class | Primary → Ollama fallback; transparent error detection and rerouting for both `generate()` and `generate_stream()` |
| `_FallbackLLMBackend` | Class | Helpful error messages when API keys are missing |
| `create_llm_backend(config)` | Function | Factory — currently hardcoded to Ollama (Claude/OpenAI paths commented out but implemented) |

---

### 2.6 `rag_pipeline/pdf_ingestion.py` (~1280 lines)

**Purpose:** PDF parsing, text chunking, embedding, and ChromaDB storage with structure-aware chunking and layout-aware text extraction.

| Symbol | Type | Description |
|--------|------|-------------|
| **Text Processing** | | |
| `_approx_token_count(text)` | Function | Fast token estimate (~words × 1.3) |
| `_is_code_line(line)` | Function | Regex-based Python syntax detection |
| `_is_code_block(lines)` | Function | Returns True if >40% of lines look like code |
| `_split_sentences(text)` | Function | Sentence splitting with abbreviation awareness |
| `_split_paragraphs(text)` | Function | Code-block-aware paragraph splitting |
| `_clean_text(text)` | Function | Whitespace normalisation preserving indentation |
| **Chunking** | | |
| `chunk_text(text, chunk_size, chunk_overlap, unit, max_tokens)` | Function | Top-level dispatcher (token or char-based) |
| `_chunk_by_tokens(text, ...)` | Function | Paragraph → sentence → word hierarchy; code blocks kept intact |
| `_chunk_by_chars(text, ...)` | Function | Legacy character-based chunking |
| **Structural Parsing** | | |
| `StructuralChunk` | Dataclass | Carries `chapter`, `section_id`, `section_title`, `snippet_id`, `is_code` metadata |
| `_parse_structural_segments(text)` | Function | Chapter/section/snippet boundary detection via regex |
| `structure_aware_chunk(text, ...)` | Function | Structure-respecting chunking with code preservation |
| **Noise Removal** | | |
| `_strip_repeated_lines(full_text, pages_text, min_occurrences)` | Function | Removes repeated header/footer lines appearing on ≥N pages |
| **Layout Extraction** | | |
| `_MONO_FONTS` | Constant | Frozenset of monospace font names for code detection |
| `_is_monospace(font_name)` | Function | Checks if font is monospace |
| `_extract_page_with_layout(page)` | Function | PyMuPDF layout-aware extraction — reconstructs code indentation from x-coordinates of monospace spans |
| `extract_text_by_page(pdf_path)` | Function | Returns `(list_of_page_texts, file_metadata)` using layout-aware extraction |
| `extract_text_from_pdf(pdf_path)` | Function | Legacy helper returning full text + metadata |
| `_detect_section_heading(text)` | Function | Heuristic heading detection (numbered, ALL-CAPS, title-case, colon-ending) |
| **Ingestion Service** | | |
| `PDFIngestionService` | Class | End-to-end PDF → ChromaDB ingestion |
| `.ingest_pdf(path, extra_metadata, force)` | Method | SHA-256 dedup, page-aware chunking, structural metadata enrichment, embedding, ChromaDB upsert |
| `.ingest_directory(directory, recursive)` | Method | Batch-ingest all PDFs in a directory |
| `.reingest_all()` | Method | Delete + re-process all PDFs with latest pipeline |
| `.ingest_uploaded_bytes(name, bytes, metadata)` | Method | Streamlit file uploader integration |
| `.delete_source(source_name)` | Method | Remove all chunks for a given PDF |
| `._notify_change(source, action)` | Method | Invokes callback to invalidate downstream caches |

---

### 2.7 `rag_pipeline/perf_trace.py` (~175 lines)

**Purpose:** Lightweight latency instrumentation for pipeline stages.

| Symbol | Type | Description |
|--------|------|-------------|
| `Span` | Dataclass | `name`, `start`/`end` time, `metadata`, `elapsed_ms`/`elapsed_s` properties |
| `PipelineTrace` | Class | Thread-safe (Lock), context manager `span(name, **meta)`, `start()`/`stop()`, `as_dict()`, `summary()`, `get_span(name)`, `total_ms` property |

---

### 2.8 `rag_pipeline/query_engine.py` (~968 lines)

**Purpose:** Full RAG pipeline orchestrator with a 10-stage latency-optimised query flow and streaming support.

| Symbol | Type | Description |
|--------|------|-------------|
| `RetrievedChunk` | Dataclass | `text`, `source`, `chunk_index`, `distance`, `metadata` |
| `RAGResponse` | Dataclass | `query`, `answer`, `chunks`, `rag_enabled`, `cached`, `faq_hit`, `trace` |
| `RAGQueryEngine` | Class | Central orchestrator wiring all components |

**10-stage `query()` pipeline:**

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Cache lookup | `SemanticCache.lookup()` with source-aware cache keys |
| 2 | FAQ fast-path | `TieredRetriever.check_faq()` with strict threshold |
| 3 | Query rewrite | `QueryRewriter.rewrite()` — multi-query + HyDE expansion |
| 4 | Concurrent embed + retrieve | `ThreadPoolExecutor` (max 4 workers) parallel embedding and search across query variants |
| 5 | Threshold + dedup | Similarity threshold filter + `chunk_hash`-based dedup |
| 6 | Metadata boost | Regex-based snippet/section reference matching from query → distance reduction |
| 7 | Re-rank | `CrossEncoderReranker.rerank()` with score threshold |
| 8 | Token-budget context build | `budget_chunks()` to fit LLM prompt window |
| 9 | LLM generate | `generate()` or `generate_stream()` with structured context |
| 10 | Cache store + retrieval log | JSONL logging for offline analysis |

**Additional methods:**

| Symbol | Type | Description |
|--------|------|-------------|
| `query_stream(query, top_k, where, source_filter)` | Method | Streaming variant with TTFT tracing |
| `_embed_and_search(query, top_k, where)` | Method | Embed single query variant and retrieve |
| `_build_where_filter(where, source_filter)` | Method | Merges filters + multi-tenant isolation |
| `_deduplicate_chunks(chunks)` | Method | `chunk_hash`-based O(n) dedup |
| `_parse_results(results)` | Static | Converts raw ChromaDB results to `RetrievedChunk` |
| `_apply_metadata_boost(query, chunks)` | Static | Snippet/section boost: regex patterns → distance reduction; code-chunk boost when query asks for code |
| `_build_context(chunks, token_budget)` | Static | Formats chunks with structured headers (source, page, section, similarity, rerank score, code warnings) |

---

### 2.9 `rag_pipeline/query_rewriter.py` (~290 lines)

**Purpose:** LLM-powered query expansion via multi-query rewriting, HyDE passage generation, and query classification.

| Symbol | Type | Description |
|--------|------|-------------|
| `QueryRewriter` | Class | Main rewriter using LLM backend |
| `.rewrite(query, n)` | Method | Generates N reformulations + HyDE hypothetical passage |
| `.classify_query(query)` | Method | Returns one of: `FACTUAL`, `PROCEDURAL`, `COMPARATIVE`, `SUMMARY`, `ANALYTICAL` |
| `_REWRITE_SYSTEM` | Constant | System prompt for query reformulation (includes snippet/textbook naming guidance) |
| `_HYDE_SYSTEM` | Constant | System prompt for hypothetical document generation |
| `_CLASSIFY_SYSTEM` | Constant | System prompt for query type classification |

---

### 2.10 `rag_pipeline/rag_page.py` (~330 lines)

**Purpose:** Standalone Streamlit page rendering the full RAG interface.

| Symbol | Type | Description |
|--------|------|-------------|
| `render_rag_page()` | Function | Full page with custom CSS, logo, RAG toggle, KB source selector, query input, submit button with multi-color spinner, response rendering, PDF uploader, knowledge base expander |

---

### 2.11 `rag_pipeline/reranker.py` (~240 lines)

**Purpose:** Cross-encoder re-ranking (two-stage retrieval).

| Symbol | Type | Description |
|--------|------|-------------|
| `RerankerBackend` | Protocol | `score(query, texts) -> List[float]` |
| `CrossEncoderBackend` | Class | Wraps `cross-encoder/ms-marco-MiniLM-L-6-v2` with lazy model loading |
| `CrossEncoderReranker` | Class | Applies re-ranking with `score_threshold` filtering; enriches chunk metadata with `rerank_score` |
| `.rerank(query, chunks, top_n)` | Method | Sorts chunks by cross-encoder score, filters by threshold, returns top N |

---

### 2.12 `rag_pipeline/semantic_cache.py` (~240 lines)

**Purpose:** Embedding-similarity response cache for near-duplicate queries (<5ms lookup).

| Symbol | Type | Description |
|--------|------|-------------|
| `CacheEntry` | Dataclass | `query`, `embedding`, `answer`, `chunks_summary`, `space_id`, `timestamp`, `hit_count` |
| `SemanticCache` | Class | Thread-safe (Lock); cosine similarity-based lookup with TTL eviction, LRU-based store, targeted `invalidate(source)` |
| `.lookup(query, space_id)` | Method | Returns `CacheEntry` if similarity ≥ threshold and TTL not expired |
| `.store(query, answer, chunks_summary, space_id)` | Method | Embeds query, stores entry, evicts LRU if over max size |
| `.invalidate(source)` | Method | Clears entries referencing a specific source, or full cache if `source=None` |

---

### 2.13 `rag_pipeline/tiered_retrieval.py` (~245 lines)

**Purpose:** Two-tier retrieval: FAQ fast-path → full pipeline fallback.

| Symbol | Type | Description |
|--------|------|-------------|
| `FAQEntry` | Dataclass | `question`, `answer`, `source`, `metadata` |
| `TieredRetriever` | Class | Separate ChromaDB collection for FAQs |
| `.check_faq(query)` | Method | Returns `FAQEntry` if similarity ≥ 0.90 (strict threshold) |
| `.add_faq(question, answer, source, metadata)` | Method | Upserts FAQ entry |
| `.remove_faq(question)` | Method | Deletes FAQ by question text |
| `.list_faqs()` | Method | Returns all FAQ entries |

---

### 2.14 `rag_pipeline/token_counter.py` (~135 lines)

**Purpose:** Fast approximate token counting (tiktoken if available, else whitespace×1.3 heuristic).

| Symbol | Type | Description |
|--------|------|-------------|
| `count_tokens(text)` | Function | Returns token count using tiktoken or heuristic |
| `truncate_to_budget(text, max_tokens)` | Function | Truncates text to fit within token budget |
| `budget_chunks(chunk_texts, max_total_tokens, separator_tokens, greedy_pack)` | Function | Returns indices of chunks fitting within budget (greedy packing) |

---

### 2.15 `rag_pipeline/triplet_export.py` (~270 lines)

**Purpose:** Fine-tuning triplet generation (query, positive, negative) for contrastive embedding training.

| Symbol | Type | Description |
|--------|------|-------------|
| `Triplet` | Dataclass | `query`, `positive` (relevant passage), `negative` (irrelevant passage) |
| `TripletExporter` | Class | Generates training triplets from eval datasets or retrieval logs |
| `.generate_from_eval_dataset(dataset, engine, top_k)` | Method | Hard negatives + random negatives |
| `.generate_from_retrieval_log(log_path)` | Method | Mines triplets from JSONL retrieval logs |
| `.export_jsonl(triplets, path)` | Method | Exports as JSONL |
| `.export_csv(triplets, path)` | Method | Exports as CSV |

---

### 2.16 `rag_pipeline/ui_components.py` (~657 lines)

**Purpose:** Reusable Streamlit widgets for the RAG interface.

| Symbol | Type | Description |
|--------|------|-------------|
| **Session helpers** | | |
| `_get_config()`, `_get_vector_store()`, `_get_embedding_service()`, `_get_ingestion_service()`, `_get_query_engine()` | Functions | Cached session state singletons |
| `_log_feedback(query, answer, feedback)` | Function | JSONL feedback logging |
| **Widget functions** | | |
| `render_rag_toggle()` | Function | RAG enable/disable toggle |
| `render_pdf_uploader()` | Function | Multi-file PDF uploader with progress spinner and cancel |
| `render_query_input()` | Function | Query text input |
| `render_rag_response(response)` | Function | Full response UI: answer with code block handling, thumbs up/down feedback, re-submit button, source chunks expander |
| `_render_answer_content(text)` | Function | Splits fenced code blocks for `st.code()` rendering |
| `_render_code_apply_section(code_blocks, query, answer)` | Function | "Apply Code to Strategy" UI panel — preview/apply/revert workflow with LLM-assisted merge |
| `render_kb_source_selector()` | Function | Dropdown to filter queries by uploaded PDF source |
| `render_knowledge_base()` | Function | Collection stats, source management, re-ingest, reset |

---

### 2.17 `rag_pipeline/vector_store.py` (~300 lines)

**Purpose:** ChromaDB wrapper for collection lifecycle, CRUD, and inspection.

| Symbol | Type | Description |
|--------|------|-------------|
| `VectorStoreManager` | Class | Lazy init of `PersistentClient` + collection (cosine HNSW with configurable M/ef) |
| `.add_documents(ids, documents, metadatas, embeddings)` | Method | Upsert documents with embeddings |
| `.query(query_embeddings, n_results, where, include)` | Method | Vector similarity search |
| `.delete_by_metadata(where)` | Method | Delete chunks matching filter |
| `.delete_by_ids(ids)` | Method | Delete by document IDs |
| `.count()` | Method | Total chunks in collection |
| `.source_exists(source)` | Method | Check if a source PDF is indexed |
| `.get_source_file_hash(source)` | Method | Retrieve stored file hash for dedup |
| `.get_source_chunk_count(source)` | Method | Number of chunks for a given source |
| `.list_sources()` | Method | Distinct source PDF names |
| `.get_collection_stats()` | Method | Summary dict (total docs, sources, collection name) |
| `.reset_collection()` | Method | Delete and recreate the collection |

---

### 2.18 `rag_pipeline/code_applier.py` (~492 lines)

**Purpose:** Extracts code blocks from RAG answers and applies them to strategy files via LLM-assisted merging, with backup/revert support.

| Symbol | Type | Description |
|--------|------|-------------|
| `CodeBlock` | Dataclass | `language`, `code`, `index` |
| `StrategyFileInfo` | Dataclass | `path`, `rel_path`, `name`, `category` |
| `PatchResult` | Dataclass | `success`, `target_file`, `backup_file`, `message`, `diff_summary` |
| `_CODE_APPLY_SYSTEM` | Constant | LLM system prompt for intelligent code merging |
| `_BACKUP_DIR` | Constant | Directory for timestamped backups |
| `_APPLY_LOG` | Constant | JSONL audit log path |
| `extract_code_blocks(answer)` | Function | Regex extraction of fenced code blocks from Markdown answer |
| `list_strategy_files()` | Function | Scans `trading_strategies/` subdirectories for `.py` files |
| `generate_patch(target_file, code_blocks, query, config)` | Function | LLM-assisted merge — reads target, sends to LLM with code snippets, returns modified source + summary |
| `_strip_markdown_fences(text)` | Function | Removes surrounding Markdown code fences |
| `_diff_summary(original, modified)` | Function | Human-readable line-level diff summary |
| `apply_patch(target_file, modified_source, query)` | Function | Writes modified file with timestamped backup, `py_compile` syntax check (rolls back on error), audit logging |
| `_log_application(target, backup, query, diff_summary)` | Function | JSONL audit trail |
| `revert_last_patch(target_file)` | Function | Restores most recent backup for a given file |

---

## 3. trading_strategies/ Module

Backtesting implementations spanning derivatives, FX intraday, portfolio optimisation, and risk modelling.

### 3.1 `trading_strategies/derivatives/options_straddle_bktest.py` (~349 lines)

**Purpose:** Long straddle options backtest using yfinance for AAPL options data.

| Symbol | Type | Description |
|--------|------|-------------|
| `contractsize` | Constant | `1` |
| `threshold` | Constant | `10` — trigger when `|call−put| < threshold` |
| `find_strike_price(df)` | Function | Extracts common strike prices from call/put column names |
| `straddle(options, spot, contractsize, strikeprice)` | Function | Merges option prices + spot data for a given strike |
| `signal_generation(df, threshold)` | Function | Generates entry signals when call−put price difference is below threshold |
| `plot(df, strikeprice, contractsize)` | Function | Payoff diagram with profit/loss coloring, breakeven points, annotations |
| `main()` | Function | Fetches AAPL options via `yf.Ticker`, iterates through strikes, runs straddle analysis |

---

### 3.2 `trading_strategies/derivatives/vix_calculator.py` (~512 lines)

**Purpose:** VIX calculation following the CBOE white paper methodology (variance swap based), adapted for AAPL equity options using yfinance.

| Symbol | Type | Description |
|--------|------|-------------|
| `cmt_rate_fill_date(cmt_rate)` | Function | Fills weekend/holiday gaps in CMT (Constant Maturity Treasury) rates |
| `get_settlement_day(current_day, time_horizon, expiration_day, expiration_hour, public_holidays)` | Function | Calculates settlement day skipping weekends/holidays |
| `get_time_to_expiration(...)` | Function | Computes time to expiration in minutes |
| `get_forward_strike(options, interest_rate, time_to_expiration)` | Function | Finds forward price level and closest strike ≤ forward |
| `get_options_call_inclusion(options, strike)` | Function | Selects OTM call options with zero-prior-settle exclusion |
| `get_options_put_inclusion(options, strike)` | Function | Selects OTM put options with zero-prior-settle exclusion |
| `compute_sigma(forward, strike, calls, puts, rate, tte)` | Function | Variance swap sigma formula |
| `compute_vix(tte_front, tte_rear, sigma_front, sigma_rear, mins_timeframe, mins_year)` | Function | Weighted average of front/rear term sigma → VIX index |
| `vix_calculator(df, cmt_rate, calendar, ...)` | Function | Aggregates all functions for end-to-end VIX computation |
| `main()` | Function | Downloads AAPL options + ^IRX treasury rates, builds options/cmt DataFrames, calculates VIX-style volatility |

---

### 3.3 `trading_strategies/fx_intraday/dual_thrust_bktest.py` (~300 lines)

**Purpose:** Opening range breakout strategy (Dual Thrust) — sets upper/lower thresholds from previous days' OHLC, takes long/short positions when price exceeds thresholds, clears at day end.

| Symbol | Type | Description |
|--------|------|-------------|
| `rg` | Parameter | `5` — lookback days for range calculation |
| `param` | Parameter | `0.5` — trigger range multiplier |
| `min2day(df, column, year, month, rg)` | Function | Converts minute-frequency data to intraday OHLC with range calculation |
| `signal_generation(df, intraday, param, column, rg)` | Function | Generates long/short signals using threshold breach + `cumsum` position control; supports position reversal |
| `plot(signals, intraday, column)` | Function | Visualises signals with entry/exit markers |
| `main()` | Function | Loads GBP/USD minute data, runs strategy |

---

### 3.4 `trading_strategies/fx_intraday/london_breakout_bktest.py` (~300 lines)

**Purpose:** London FX session breakout strategy — uses last Tokyo trading hour to set price thresholds, trades within first 30 minutes of London open, clears by London close.

| Symbol | Type | Description |
|--------|------|-------------|
| `risky_stop` | Parameter | 100 bps stop-loss (risky variant) |
| `open_minutes` | Parameter | 30 — minutes after London open for entries |
| `london_breakout(df)` | Function | Initialises signal columns |
| `signal_generation(df, method)` | Function | Core logic: Tokyo price threshold construction, entry within first 30 min of London, 50 bps stop-loss, `cumsum` position tracking |
| `plot(new)` | Function | Two-panel visualisation: full day + market opening zoom |
| `main()` | Function | Loads GBP/USD minute data from histdata.com format |

---

### 3.5 `trading_strategies/portfolio_analysis/asset_allocation.py` (~240 lines)

**Purpose:** Portfolio optimisation for quantum computing stocks (RGTI, QBTS, IONQ, NBIS) using Sharpe ratio maximisation, median return maximisation, and dynamic asset allocation.

| Symbol | Type | Description |
|--------|------|-------------|
| `tickers` | Constant | `['RGTI', 'QBTS', 'IONQ', 'NBIS']` |
| `ROLL_WINDOW` | Constant | Adaptive rolling window (`min(21, len(data)-1)`) |
| `marr` | Constant | `0` — minimal acceptable rate of return |
| `getdata(tickers, start, end)` | Function | Downloads OHLC from yfinance for multiple tickers |
| `multi(x)` | Function | Returns weight tuple for optimiser |
| `maximize_sharpe(x)` | Function | Objective: maximise rolling Sharpe ratio (SLSQP) |
| `maximize_median_yearly_return(x)` | Function | Objective: maximise rolling median return |
| `constraint(x)` | Function | Weights sum ≤ 1, no leverage, no short |

**Analysis pipeline:**
1. Buy & hold cumulative return comparison
2. Rolling return analysis
3. Sharpe ratio computation per asset + equal-weight
4. SLSQP optimisation — max Sharpe weights
5. SLSQP optimisation — max median return weights
6. Dynamic asset allocation (conditional weight switching based on rolling return thresholds)
7. Lookback-period performance comparison

---

### 3.6 `trading_strategies/risk_modelling/monte_carlo_bktest.py` (~220 lines)

**Purpose:** Monte Carlo simulation for stock price forecasting using geometric Brownian motion, with prediction accuracy testing.

| Symbol | Type | Description |
|--------|------|-------------|
| `get_gradient_colors(n)` | Function | Generates N gradient colors (yellow → red) for visualisation |
| `monte_carlo(data, testsize=0.5, simulation=100)` | Function | Core simulation: train/test split → log returns → drift calculation → GBM simulation → selects best fit by minimum standard deviation against actuals |
| `plot(df, forecast_horizon, d, pick, ticker)` | Function | Two plots: (1) all simulations with best-fit highlighted vs actual (training), (2) best-fit vs actual across training + testing with boundary marker |
| `test(df, ticker, simu_start=100, simu_end=1000, simu_delta=100)` | Function | Tests whether increasing simulation count improves directional prediction accuracy; horizontal bar chart showing success/failure per simulation count |
| `main()` | Function | Runs on GE stock data (2016–2019), worst performer of 2018 — tests Monte Carlo against extreme price movements |

**Key insight documented in code:**
> "Monte Carlo simulation in trading is house of cards — it is merely illusion that Monte Carlo can forecast any asset price or direction."

---

## Appendix: Technology Stack Summary

| Component | Technology |
|-----------|-----------|
| Web UI | Streamlit |
| Broker API | Zerodha Kite Connect SDK |
| Database | PostgreSQL (psycopg2) |
| Vector Store | ChromaDB (PersistentClient, HNSW cosine) |
| Embeddings | sentence-transformers (`BAAI/bge-base-en-v1.5`, 768-dim) |
| Re-ranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM (primary) | Ollama (local, default: mistral) |
| LLM (optional) | Anthropic Claude (`claude-sonnet-4-20250514`), OpenAI (`gpt-4o`) |
| PDF Parsing | PyMuPDF (fitz) with layout-aware extraction |
| Browser Automation | Selenium (Chrome/Edge) |
| Market Data | yfinance, NSE India |
| Optimisation | scipy.optimize (SLSQP) |
| ML | scikit-learn (train_test_split) |
