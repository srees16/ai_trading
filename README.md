# Centurion Capital LLC — Enterprise AI Trading Platform

A Python-based enterprise trading platform combining multi-source news scraping, AI-powered sentiment analysis, fundamental & technical analysis, strategy backtesting, persistent data storage, live Indian market trading via Zerodha Kite Connect, and a RAG-powered document intelligence pipeline. Built on Streamlit with PostgreSQL persistence, MinIO object storage, ChromaDB vector search, and multi-provider LLM integration.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Analysis Engine](#2-core-analysis-engine)
3. [Strategy Backtesting](#3-strategy-backtesting)
4. [Live Trading — Zerodha Kite Connect](#4-live-trading--zerodha-kite-connect)
5. [RAG Document Intelligence](#5-rag-document-intelligence)
6. [Database Layer](#6-database-layer)
7. [Object Storage (MinIO)](#7-object-storage-minio)
8. [Interactive Web Interface](#8-interactive-web-interface)
9. [Project Structure](#9-project-structure)
10. [Installation](#10-installation)
11. [Usage Guide](#11-usage-guide)
12. [API Reference](#12-api-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Dependencies](#14-dependencies)

---

## 1. Architecture Overview

The application follows a modular, deferred-import architecture for fast startup:

```
app.py (Streamlit Router)
  ├── apply_custom_styles() → initialize_session_state() → check_authentication()
  ├── Page routing via st.session_state.current_page:
  │     main → analysis → fundamental → backtesting → crypto → history
  └── All page imports deferred to route branches
```

### Core Pipeline

```
AlgoTradingSystem (main.py)
  ├── NewsAggregator     → Yahoo Finance, Finviz, Investing.com, TradingView, r/WallStreetBets
  ├── SentimentAnalyzer   → DistilBERT transformer model
  ├── MetricsCalculator   → Fundamentals (yfinance) + Technicals (RSI, MACD, Bollinger)
  ├── DecisionEngine      → Weighted scoring → STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
  ├── NotificationManager → Desktop popups (plyer) + HTML email via SMTP
  └── StorageManager      → Excel/CSV export + MinIO object storage
```

### Dual Strategy System

| System | Location | Base Class | Data Source | Output |
|--------|----------|------------|-------------|--------|
| **Framework** | `strategies/` + `trading_strategies/` | `BaseStrategy` (ABC) | `DataService` (yfinance, cached) | `StrategyResult` (charts, tables, metrics, signals) |
| **Standalone** | `*_bktest.py` files | None | Direct yfinance or CSV | matplotlib plots, printed stats |

---

## 2. Core Analysis Engine

### News Scraping

Five concurrent scrapers with 3-layer caching (session → scraper cache → DB freshness):

| Source | Method | Limit |
|--------|--------|-------|
| Yahoo Finance | `yfinance` library (`Ticker.news`) | 10/ticker |
| Finviz | HTTP scraping (optional Selenium for Elite) | 10/ticker |
| Investing.com | HTTP with custom headers | 10/ticker |
| TradingView | JSON API (`news-headlines.tradingview.com`) | 10/ticker |
| r/WallStreetBets | Reddit public JSON API (8 flairs) | 50/flair |

- `asyncio.Semaphore(5)` for concurrency control
- SHA-256 content deduplication
- Adaptive rate limiting with exponential backoff (0.5s base, 30s max)

### Sentiment Analysis

- **Model**: `distilbert-base-uncased-finetuned-sst-2-english` (HuggingFace)
- **Input**: `title + ". " + summary`, truncated to 512 characters
- **Output**: `sentiment_score` (±confidence), `sentiment_label` (POSITIVE/NEGATIVE/NEUTRAL)
- **High-confidence threshold**: 0.85

### Financial Metrics

| Category | Metrics |
|----------|---------|
| **Technical** | RSI (14-period Wilder), MACD (12/26/9), Bollinger Bands (20, ±2σ), Fibonacci levels, Max Drawdown |
| **Fundamental** | PEG ratio, ROE, EPS, Free Cash Flow, DCF value, Graham Intrinsic Value |
| **Scoring** | Altman Z-Score (safe >2.99, distress <1.81), Beneish M-Score (manipulator > -2.22), Piotroski F-Score (0–9) |

### Decision Engine

Weighted combination of three signal components:

```
Combined Score = Sentiment × 0.4 + Fundamentals × 0.3 + Technicals × 0.3
```

| Score | Decision |
|-------|----------|
| ≥ 0.7 | STRONG_BUY |
| ≥ 0.4 | BUY |
| ≤ -0.7 | STRONG_SELL |
| ≤ -0.4 | SELL |
| else | HOLD |

- Sentiment: raw score with 1.2x boost when confidence > 0.85, clamped to [-1, 1]
- Fundamentals: averaged factor scores from PEG, ROE, EPS, and intrinsic/price ratio
- Technicals: averaged factor scores from RSI, MACD histogram, Bollinger position, and drawdown

---

## 3. Strategy Backtesting

### Registered Strategies (11)

| ID | Name | Category | Key Parameters |
|---|---|---|---|
| `macd` | MACD Oscillator | Momentum | `ma_short=10`, `ma_long=21`, `use_ema=True` |
| `awesome_oscillator` | Awesome Oscillator | Momentum | `ao_short=5`, `ao_long=34` |
| `heikin_ashi` | Heikin-Ashi | Momentum | `confirmation_candles=1`, `use_ma_filter=False`, `ma_period=20` |
| `parabolic_sar` | Parabolic SAR | Momentum | `af_start=0.02`, `af_increment=0.02`, `af_max=0.2` |
| `rsi_pattern` | RSI Pattern | Pattern Recognition | `rsi_period=14`, `oversold=30`, `overbought=70` |
| `shooting_star` | Shooting Star | Pattern Recognition | `lower_bound=0.2`, `body_size=0.5`, `stop=5%`, `hold=7d` |
| `support_resistance` | Support & Resistance | Pattern Recognition | `n1=2`, `n2=2`, `back_candles=30`, `proximity=2%` |
| `bollinger_pattern` | Bollinger Pattern | Pattern Recognition | `bb_period=20`, `bb_std=2.0`, `pattern_period=75` |
| `pairs_trading` | Pairs Trading | Statistical Arbitrage | `bandwidth=60`, `z_entry=1.0`, `z_exit=0.0` (requires 2 tickers) |
| `mean_reversion` | Mean Reversion (Z-Score) | Statistical Arbitrage | `lookback=30`, `threshold=2.0`, `stoploss=5%` (requires ≥2 tickers) |
| `crypto_mean_reversion` | Crypto Mean Reversion | Crypto | Same Z-Score params + Binance API (requires ≥2 tickers) |

### Standalone Scripts (6 additional)

| Script | Category | Description |
|--------|----------|-------------|
| `london_breakout_bktest.py` | FX Intraday | London session breakout on GBP/USD minute data (Tokyo range thresholds, 50 bps stop-loss) |
| `dual_thrust_bktest.py` | FX Intraday | Opening range breakout with configurable lookback and trigger multiplier |
| `options_straddle_bktest.py` | Derivatives | Long straddle on AAPL options (entry when \|call−put\| < $10) |
| `vix_calculator.py` | Derivatives | CBOE VIX methodology (variance swap formula) applied to equity options |
| `asset_allocation.py` | Portfolio Analysis | SLSQP portfolio optimisation for Sharpe ratio & median return maximisation |
| `monte_carlo_bktest.py` | Risk Modelling | GBM price simulation with direction prediction accuracy testing |

### Strategy Framework

**BaseStrategy** (ABC) provides:
- Built-in metric calculators: Sharpe ratio, Sortino ratio, Max Drawdown
- Portfolio calculators: long/short and long-only variants
- Sentiment adjustment helpers: scale indicators, zero positions, filter signals
- Risk parameters: `stop_loss=5%`, `take_profit=10%`, `max_position_size=25%`, `max_drawdown=20%`

**DataService** (singleton with 1-hour cache):
- yfinance wrapper with technical indicator overlays (SMA, EMA, RSI, MACD, Bollinger, ATR)
- Batch preloading via `yf.download(group_by='ticker')`

**Strategy auto-discovery**: `StrategyRegistry` + `StrategyLoader` scan `trading_strategies/` subdirectories and register `BaseStrategy` subclasses dynamically.

### Crypto Mean Reversion Pipeline

Full statistical arbitrage pipeline via the Binance public REST API (no API key required):
1. **Data**: Paginated klines (1000/request), per-symbol CSV caching with incremental updates
2. **EDA**: Correlation matrices, price plots, distribution analysis
3. **Statistical tests**: ADF (stationarity), Hurst exponent (mean-reverting < 0.5), Variance Ratio, Half-Life
4. **Cointegration**: Engle-Granger pairwise + Johansen multi-asset eigenvector
5. **Portfolio construction**: OLS hedge ratio (2-asset) or Johansen weights (3+ assets)
6. **Backtesting**: via `backtesting.py` library with Z-Score naive strategy
7. **Optimisation**: Grid search over lookback/threshold/stoploss for max equity, min drawdown, min volatility, max Sharpe

---

## 4. Live Trading — Zerodha Kite Connect

Streamlit dashboard for real-time Indian equity monitoring, order management, option chain analysis, and automated RSI trading.

### Components

| Module | Purpose |
|--------|---------|
| `zerodha_live.py` | Main dashboard (~1326 lines) — live quotes, order book, positions, holdings, RSI scanner |
| `auth/kite_auth.py` | OAuth flow with local HTTP callback + Selenium auto-login |
| `auth/kite_session.py` | Reusable authenticated `KiteConnect` session |
| `core/config.py` | API credentials, DB config, index groups (NIFTY50, BANKNIFTY, NIFTYIT, NIFTYENERGY) |
| `core/database_service.py` | PostgreSQL connection pool for `livestocks_ind` database |
| `core/selenium_service.py` | Chrome/Edge WebDriver lifecycle management |
| `nse/nse_csv_downloader.py` | NSE bhavcopy CSV download via Selenium |
| `nse/nse_data_loader.py` | CSV → PostgreSQL bulk loader |
| `options/option_chain.py` | Concurrent option chain with OI, Greeks, and IV (ThreadPoolExecutor, 20 workers) |
| `trading/order_service.py` | Market/Limit/SL/SL-M orders, CNC/MIS/NRML products, DAY/IOC validity |
| `trading/rsi_strategy.py` | Live RSI scanner — BUY (RSI<30 + reversal), SELL (RSI>70 + reversal), auto-order placement |

### Key Features
- Auto-refresh every 30 seconds via `@st.fragment(run_every=...)`
- Market status pill indicators from NSE API (pre-open, live, post-market)
- Batch quote fetching (200 symbols/batch)
- Option chain: expiry discovery (45 days + monthly), Sensibull-style colouring, ATM highlighting, PCR metric

---

## 5. RAG Document Intelligence

Retrieval-Augmented Generation pipeline for document Q&A with PDF ingestion, hybrid search, and multi-provider LLM generation.

### 10-Stage Query Pipeline

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Semantic Cache | Embedding-based lookup (cosine ≥ 0.95, TTL 3600s) |
| 2 | FAQ Fast-Path | Dedicated ChromaDB collection (similarity ≥ 0.90) |
| 3 | Query Rewrite | LLM-powered multi-query expansion + HyDE hypothetical passage |
| 4 | Hybrid Retrieval | BM25 (weight 0.4) + vector similarity (weight 0.6) fused via RRF |
| 5 | Threshold + Dedup | Similarity filter + chunk-hash deduplication |
| 6 | Metadata Boost | Regex-based snippet/section reference matching |
| 7 | Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (score threshold 0.25) |
| 8 | Context Assembly | Token-budget constrained chunk selection (4000 tokens) |
| 9 | LLM Generation | Streaming with structured context and citations |
| 10 | Cache + Log | Store response + JSONL retrieval logging |

### Configuration

| Component | Setting |
|-----------|---------|
| **Embedding** | `BAAI/bge-base-en-v1.5` (768-dim) |
| **Vector Store** | ChromaDB HNSW cosine (M=32, ef_construction=200, ef_search=150) |
| **Chunking** | Token-based, size=512, overlap=128 |
| **LLM (local)** | Ollama — `mistral` (default) |
| **LLM (cloud)** | Anthropic Claude (`claude-sonnet-4-20250514`) or OpenAI (`gpt-4o`) |

### Additional Capabilities
- **PDF Ingestion**: Structure-aware chunking via PyMuPDF with layout-aware code extraction, SHA-256 file deduplication
- **Code Applicator**: Extracts code from RAG answers and applies to strategy files via LLM-assisted merging with `py_compile` verification and one-click revert
- **Evaluation Suite**: IR metrics (Hit Rate, MRR, NDCG, MAP) + LLM-as-Judge faithfulness scoring (1–5)
- **Triplet Export**: Training data generation (query, positive, negative) for embedding fine-tuning
- **Performance Tracing**: Stage-level latency instrumentation

---

## 6. Database Layer

### PostgreSQL Schema (12 tables)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `analysis_runs` | Tracks each analysis execution | status, tickers, duration, user_id |
| `news_items` | Scraped news with sentiment | ticker, source, sentiment_label, content_hash (SHA-256 dedup) |
| `stock_signals` | Trading signals | decision, decision_score, reasoning, technical indicators |
| `fundamental_metrics` | Per-ticker fundamental snapshots | PE, PEG, ROE, Z-Score, M-Score, F-Score |
| `backtest_results` | Strategy backtest outcomes | total_return, sharpe_ratio, max_drawdown, equity_curve (JSONB) |
| `backtest_trades` | Individual trade records | entry/exit price, PnL, holding period |
| `backtest_equity_points` | Equity curve data points | portfolio_value, drawdown, benchmark |
| `backtest_daily_returns` | Daily return series | daily_return, cumulative_return |
| `strategy_performance_summary` | Materialised gold layer | avg metrics across all backtests per strategy |
| `user_watchlists` | Custom ticker lists | tickers (ARRAY), is_default |
| `alert_configurations` | Price/signal alerts | conditions (JSONB), notification_channels |
| `raw_scraped_news` | Bronze/raw layer | raw content, is_processed flag |
| `data_freshness` | Cache staleness tracking | last_fetched_at, consecutive_errors |

**TimescaleDB** (optional): Hypertables on `stock_signals`, `fundamental_metrics`, `news_items` with 7-day chunk interval.

### Service Layer

`DatabaseService` (singleton) provides a unified API:
- Analysis lifecycle: `start_analysis_run()` → `complete_analysis_run()` / `fail_analysis_run()`
- Persistence: `save_signals()`, `save_news_items()` (SHA-256 dedup), `save_fundamental_metrics()` (upsert)
- Backtesting: `save_backtest_result()` with normalised detail tables + strategy summary refresh
- Freshness: `check_freshness()`, `record_fetch()`, `record_error()`

### Repository Pattern

| Repository | Key Methods |
|------------|-------------|
| `AnalysisRepository` | `create_run`, `start_run`, `complete_run`, `fail_run`, `get_recent_runs` |
| `SignalRepository` | `get_by_ticker`, `get_by_decision`, `get_top_signals`, `get_ticker_signal_history` |
| `NewsRepository` | `check_duplicate` (SHA-256), `create_with_dedup`, `get_sentiment_summary` |
| `FundamentalRepository` | `get_latest_by_ticker`, `upsert` (INSERT ON CONFLICT) |
| `BacktestRepository` | `get_top_performers`, `get_strategy_summary`, `compare_strategies` |
| `FreshnessRepository` | `is_stale`, `record_fetch`, `get_stale_tickers` |

---

## 7. Object Storage (MinIO)

S3-compatible storage for backtest chart images:

- **Path pattern**: `centurion-backtests/<run_id>/<TICKER>/<strategy_name>/<filename>`
- **Metadata tags**: `x-amz-meta-run-id`, `x-amz-meta-strategy`, `x-amz-meta-ticker`, `x-amz-meta-chart-title`
- **Formats**: matplotlib (base64 → PNG), plotly (JSON), backtesting.py (HTML)
- **Presigned URLs**: 1-hour expiry for History page viewing

```python
from storage.minio_service import get_minio_service

minio = get_minio_service()
minio.save_backtest_image(run_id, image_data, filename, strategy_name, ticker, chart_title)
images = minio.get_backtest_images(run_id)  # with presigned URLs
details = minio.list_runs_detailed()         # metadata: size, chart count, strategies
```

---

## 8. Interactive Web Interface

### Pages

| Page | Route | Description |
|------|-------|-------------|
| **Main** | `main` | Ticker selection (default / manual / CSV upload), output settings, Run Analysis button |
| **Stock Analysis** | `analysis` | Multi-colour CSS spinner during analysis → 4-tab results (Overview, Detailed Table, Top Signals, Sentiment) |
| **Fundamental** | `fundamental` | Z/M/F score interpretations, all-stocks table, three charts side-by-side |
| **Backtesting** | `backtesting` | Auto pre-computes all strategies on first visit; config panel + per-strategy result tabs with charts |
| **Crypto** | `crypto` | Isolated crypto strategies (default: ETH, BTC, LTC); Binance data, separate cache |
| **History** | `history` | 3 tabs: Analysis Runs (drill-down), Trading Signals (filterable), Backtest Results (with MinIO charts) |
| **RAG** | `rag` | PDF upload, query input with KB source selector, streaming response, code applicator |
| **Indian Equities** | Kite Connect | Live quotes, order book, positions, holdings, option chain, RSI scanner |

### Authentication
- YAML-based credentials (`auth/credentials.yaml`)
- SHA-256 password hashing with `hmac.compare_digest`
- Session timeout: 60 min absolute, 30 min inactivity
- Max 3 login attempts
- Default users: `admin`/`admin123`, `analyst`/`analyst123`

### Styling
- Enterprise CSS: dark gradient theme with Centurion branding
- Decision colours: STRONG_BUY `#00ff88`, BUY `#00cc44`, HOLD `#ffd700`, SELL `#ff6b6b`, STRONG_SELL `#ff0000`
- Background image overlay, custom buttons, consistent footer

---

## 9. Project Structure

```
centurion_core/
├── app.py                        # Streamlit application router
├── main.py                       # Core orchestration (AlgoTradingSystem)
├── config.py                     # Configuration (~140 settings, CENTURION_* env vars)
├── models.py                     # Data models (NewsItem, StockMetrics, TradingSignal)
├── utils.py                      # CSV parsing and ticker validation
├── setup_database.py             # Database schema initialisation
├── requirements.txt              # Python dependencies
├── sample_tickers.csv            # Example ticker list
├── run_streamlit.bat             # Windows quick-launch script
│
├── ui/                           # Modular UI layer
│   ├── components.py             # Header, footer, navigation, metrics cards
│   ├── charts.py                 # Plotly charts (decision, sentiment, scores)
│   ├── tables.py                 # Data tables with CSV download
│   ├── styles.py                 # CSS styling and colour constants
│   ├── assets/                   # Logo, background images
│   └── pages/
│       ├── main_page.py          # Dashboard & control panel
│       ├── analysis_page.py      # Analysis results with CSS spinner
│       ├── fundamental_page.py   # Fundamental analysis drill-down
│       ├── backtesting_page.py   # Strategy backtesting + MinIO/DB integration
│       ├── crypto_page.py        # Crypto strategy page (Binance API)
│       └── history_page.py       # Historical results browser
│
├── auth/                         # Authentication
│   ├── authenticator.py          # Login/session management
│   └── credentials.yaml          # User credentials (SHA-256 hashed)
│
├── database/                     # PostgreSQL persistence layer
│   ├── connection.py             # SQLAlchemy engine (QueuePool, pool_pre_ping)
│   ├── models.py                 # ORM models (12 tables)
│   ├── service.py                # DatabaseService singleton
│   └── repositories/             # Repository pattern (6 repos + base)
│
├── scrapers/                     # News scraping modules
│   ├── aggregator.py             # Concurrent coordinator (Semaphore, 3-layer cache)
│   ├── cache.py                  # Rate limiter + content deduplicator
│   ├── yahoo_finance.py          # yfinance library
│   ├── finviz.py                 # HTTP + optional Selenium Elite
│   ├── investing.py              # HTTP with custom headers
│   ├── tradingview.py            # JSON API
│   └── wallstreetbets.py         # Reddit public JSON (8 flairs)
│
├── sentiment/                    # AI sentiment analysis
│   └── analyzer.py               # DistilBERT implementation
│
├── metrics/                      # Financial metrics
│   └── calculator.py             # Fundamentals + technicals (yfinance)
│
├── decision_engine/              # Trading logic
│   └── engine.py                 # Weighted scoring algorithm
│
├── services/                     # Business logic
│   ├── analysis.py               # Analysis orchestration (async)
│   ├── session.py                # Session state initialisation
│   └── cache.py                  # SessionCache (TTL-aware, thread-safe)
│
├── strategies/                   # Strategy framework
│   ├── base_strategy.py          # BaseStrategy ABC + dataclasses (620 lines)
│   ├── registry.py               # StrategyRegistry singleton
│   ├── loader.py                 # Dynamic discovery + import
│   ├── data_service.py           # DataService (yfinance + indicator overlays)
│   └── utils.py                  # RSI, MDD, base64, plotly JSON, trading stats
│
├── trading_strategies/           # Strategy implementations
│   ├── __init__.py               # Lazy imports (11 strategies)
│   ├── backtest_utils.py         # Shared: MDD, candlestick, portfolio
│   ├── momentum_trading/         # MACD, Awesome Oscillator, Heikin-Ashi, Parabolic SAR
│   ├── pattern_recognition/      # RSI Pattern, Bollinger, Shooting Star, Support/Resistance
│   ├── statistical_arbitrage/    # Pairs Trading, Mean Reversion, edge utilities
│   ├── crypto/                   # Crypto Mean Reversion (Binance API + backtesting.py)
│   ├── fx_intraday/              # London Breakout, Dual Thrust (standalone)
│   ├── derivatives/              # Options Straddle, VIX Calculator (standalone)
│   ├── portfolio_analysis/       # Asset Allocation / SLSQP optimisation (standalone)
│   └── risk_modelling/           # Monte Carlo / GBM simulation (standalone)
│
├── kite_connect/                 # Zerodha live trading (Indian markets)
│   ├── zerodha_live.py           # Main Streamlit dashboard (~1326 lines)
│   ├── auth/                     # OAuth + Selenium 2FA login
│   ├── core/                     # Config, PostgreSQL, Selenium service
│   ├── nse/                      # NSE CSV download + DB loader
│   ├── options/                  # Concurrent option chain + Greeks
│   └── trading/                  # Order service + RSI strategy
│
├── rag_pipeline/                 # RAG document intelligence
│   ├── config.py                 # 60+ field configuration dataclass
│   ├── embeddings.py             # sentence-transformers (BGE-base-en-v1.5)
│   ├── vector_store.py           # ChromaDB HNSW cosine wrapper
│   ├── pdf_ingestion.py          # Structure-aware PDF chunking (~1280 lines)
│   ├── hybrid_search.py          # BM25 + vector RRF fusion
│   ├── reranker.py               # Cross-encoder re-ranking
│   ├── query_rewriter.py         # LLM query expansion + HyDE
│   ├── semantic_cache.py         # Embedding-based answer cache
│   ├── llm_service.py            # Ollama / Claude / OpenAI abstraction
│   ├── query_engine.py           # 10-stage pipeline (~968 lines)
│   ├── evaluation.py             # IR metrics + LLM-as-Judge
│   ├── tiered_retrieval.py       # FAQ tier (similarity ≥ 0.90)
│   ├── token_counter.py          # tiktoken / heuristic counter
│   ├── triplet_export.py         # Fine-tuning triplet generator
│   ├── code_applier.py           # RAG → strategy code applicator
│   ├── ui_components.py          # Streamlit RAG widgets
│   ├── rag_page.py               # RAG page entry point
│   └── perf_trace.py             # Pipeline stage timing
│
├── notifications/                # Desktop + email alerts
│   └── manager.py                # plyer popups + SMTP HTML email
│
├── storage/                      # Object storage
│   ├── manager.py                # Excel/CSV file export
│   └── minio_service.py          # MinIO S3 client (singleton)
│
└── deployment/                   # Deployment configs
    ├── docker-compose.yml        # App + MinIO containers
    ├── Dockerfile
    ├── deploy.ps1 / deploy.sh    # General deployment
    ├── deploy-azure.ps1          # Azure deployment
    ├── deploy-gcp.ps1            # GCP deployment
    └── DEPLOYMENT.md             # Cloud deployment guide
```

---

## 10. Installation

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| PostgreSQL | 14+ | Analysis & backtest persistence |
| Docker | 20+ | MinIO object storage |
| Ollama | Latest | Local LLM inference (RAG pipeline) |

### 1. Install Dependencies

```powershell
cd centurion_core
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> First run downloads the DistilBERT model (~250 MB).

### 2. Configure Environment

Create `.env` in the project root:

```ini
# ─── PostgreSQL ───────────────────────────────────────────
CENTURION_DB_HOST=localhost
CENTURION_DB_PORT=5432
CENTURION_DB_NAME=centurion_trading
CENTURION_DB_USER=admin
CENTURION_DB_PASSWORD=admin123
CENTURION_DB_ENABLED=true

# ─── MinIO Object Storage ────────────────────────────────
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_SECURE=false
MINIO_BUCKET=centurion-backtests
MINIO_ENABLED=true
```

All configuration values are overridable via `CENTURION_`-prefixed environment variables.

### 3. Set Up PostgreSQL

```powershell
# Connect as superuser
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres

# Inside psql:
CREATE USER admin WITH PASSWORD 'admin123';
CREATE DATABASE centurion_trading OWNER admin;
GRANT ALL PRIVILEGES ON DATABASE centurion_trading TO admin;
\q
```

Initialise tables:

```powershell
python setup_database.py
```

> **TimescaleDB** (optional): Hypertables are created automatically if the extension is available. Standard PostgreSQL works fine without it.

### 4. Set Up MinIO

```powershell
cd deployment
docker compose up -d minio
```

| Port | Purpose |
|---|---|
| `9000` | S3-compatible API |
| `9001` | Web Console (user: `minioadmin`, pass: `minioadmin123`) |

The `centurion-backtests` bucket is created automatically on first use.

### 5. Launch

```powershell
streamlit run app.py
```

Opens at **http://localhost:9090** (configured in `.streamlit/config.toml`).

---

## 11. Usage Guide

### Quick Start

1. Launch the app → log in → land on the **Main** page.
2. Select tickers (default list, manual entry, or CSV upload).
3. Click **Run Analysis** → results appear on the **Stock Analysis** page.
4. Navigate to **Fundamental Analysis** for Z/M/F score drill-down.
5. Navigate to **Backtest Strategy** to test any of the 11 strategies.
6. Navigate to **History** to review past runs, signals, and stored charts.

### Strategy Backtesting

1. Click **Backtest Strategy** from any page.
2. Select a strategy from the dropdown (filter by category).
3. Enter tickers and adjust period / capital / strategy-specific parameters.
4. Click **Run Backtest** — results include:
   - Per-ticker performance tabs with key metrics (return, Sharpe, Sortino, MDD)
   - Interactive charts (matplotlib & plotly)
   - Auto-persisted to PostgreSQL + MinIO
5. Switch strategies instantly — cached results load without re-computation.

### Crypto Backtesting

1. Navigate to the **Crypto** page.
2. Enter crypto tickers (e.g., `ETH, BTC, LTC`) — auto-mapped to USDT pairs.
3. The pipeline runs: EDA → statistical tests → portfolio construction → backtesting → optimisation.
4. With optimisation enabled (default), four targets are tested: max equity, min drawdown, min volatility, max Sharpe.

### RAG Document Q&A

1. Navigate to the **RAG** page.
2. Upload PDF documents to build a knowledge base.
3. Enter a query — the 10-stage pipeline retrieves and generates an answer with citations.
4. Use the Code Applicator to apply code snippets from RAG answers to strategy files.

### CSV Upload Format

Recognised headers: `Ticker`, `Symbol`, `Stock`, `Tickers`, `Symbols`, `Stocks`.

```csv
Ticker
AAPL
MSFT
GOOGL
```

### Navigation

All pages share consistent navigation buttons:

| Button | Action |
|---|---|
| 🏠 **Main** | Return to the main dashboard |
| 📈 **Stock Analysis** | View analysis results |
| 📊 **Fundamental Analysis** | Open fundamental metrics |
| 🔬 **Backtest Strategy** | Open backtesting |
| 📋 **History** | Browse historical results |

---

## 12. API Reference

### Database Service

```python
from database.service import get_database_service

db = get_database_service()
db.is_available  # True / False

with db.session_scope() as session:
    from database.repositories import AnalysisRepository, BacktestRepository
    repo = AnalysisRepository(session)
    runs = repo.get_recent_runs(days=7)
```

### MinIO Service

```python
from storage.minio_service import get_minio_service

minio = get_minio_service()
path = minio.save_backtest_image(run_id, png_bytes, "equity_curve.png", "MACD Oscillator", "AAPL", "Equity Curve")
images = minio.get_backtest_images(run_id)
runs = minio.list_runs_detailed()
minio.delete_run_images(run_id)
```

### Strategy Execution

```python
from trading_strategies import get_strategy, list_strategies

# List available strategies (no imports triggered)
for s in list_strategies():
    print(s['id'], s['name'], s['category'])

# Run a strategy
StrategyClass = get_strategy('macd')
strategy = StrategyClass()
result = strategy.run(
    tickers=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2025-01-01',
    capital=10000.0
)
print(result.metrics)  # total_return, sharpe_ratio, max_drawdown, etc.
```

### Docker Commands

```powershell
# Start everything
cd deployment && docker compose up -d

# Start only MinIO
docker compose up -d minio

# View logs
docker logs centurion-minio

# Stop
docker compose down

# Remove all data (destructive)
docker compose down -v
```

---

## 13. Troubleshooting

### Database

| Symptom | Fix |
|---------|-----|
| "no password supplied" | Set `CENTURION_DB_PASSWORD` in `.env` |
| "relation analysis_runs does not exist" | Run `python setup_database.py` |
| TimescaleDB warnings | Harmless — TimescaleDB is optional |

### MinIO

| Symptom | Fix |
|---------|-----|
| Charts not appearing after backtest | Verify `docker ps --filter name=centurion-minio` + `MINIO_ENABLED=true` |
| "minio module not found" | `pip install minio` |
| Connection refused on port 9000 | `cd deployment && docker compose up -d minio` |

### General

| Symptom | Fix |
|---------|-----|
| Import errors | `pip install -r requirements.txt --upgrade` |
| Port in use | `streamlit run app.py --server.port 8502` |
| Slow first run | DistilBERT model download (~250 MB); subsequent runs are fast |

---

## 14. Dependencies

| Category | Packages |
|---|---|
| **Web Framework** | streamlit, plotly |
| **Data** | pandas, numpy, openpyxl |
| **Financial Data** | yfinance |
| **Crypto Data** | Binance public REST API (no key required) |
| **Live Trading** | kiteconnect (Zerodha Kite Connect SDK) |
| **Scraping** | aiohttp, beautifulsoup4, lxml, requests, selenium, webdriver-manager |
| **AI/ML** | transformers, torch, scikit-learn |
| **LLM Providers** | anthropic, openai (Ollama via HTTP) |
| **RAG / Embeddings** | chromadb, sentence-transformers, PyMuPDF, tiktoken |
| **Analysis** | matplotlib, statsmodels, backtesting (0.6+), arch, scipy, seaborn |
| **Database** | sqlalchemy ≥ 2.0, psycopg2-binary ≥ 2.9, python-dotenv ≥ 1.0 |
| **Object Storage** | minio ≥ 7.2 |
| **Auth** | pyyaml ≥ 6.0 |
| **Notifications** | plyer |

---

## ⚠️ Disclaimer

This software is provided for **educational and informational purposes only**. It does not constitute financial advice, investment recommendations, or professional trading guidance. Stock trading involves substantial risk of loss. Always consult qualified financial advisors before making investment decisions. Use at your own risk.

---

**Ready to get started? Run `streamlit run app.py` and begin analysing! 🚀📈**
