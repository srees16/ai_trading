# Centurion Capital LLC — Enterprise AI Trading Platform

A Python-based enterprise trading platform combining multi-source news scraping, AI-powered sentiment analysis, fundamental & technical analysis, strategy backtesting, persistent data storage, live Indian market trading via Zerodha Kite Connect, and a RAG-powered document intelligence pipeline. Built on Streamlit with PostgreSQL persistence, MinIO object storage, ChromaDB vector search, and multi-provider LLM integration.

---

## ⚡ Quick Start (5 Minutes)

Get the app running on a new machine with these commands:

### 1️⃣ Clone & Install Dependencies
```powershell
git clone https://github.com/srees16/centurion_core.git
cd centurion_core
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2️⃣ Start PostgreSQL (Docker)
```powershell
docker run -d --name centurion-postgres -p 9003:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=superadmin1 -e POSTGRES_DB=centurion_rag postgres:15
```

### 3️⃣ Initialize Database
```powershell
python setup_database.py
```
Expected: `✓ Database tables created successfully`

### 4️⃣ Start MinIO (Docker) — for Backtest Charts
```powershell
docker run -d --name centurion-minio -p 9004:9000 -p 9002:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin123 minio/minio:latest server /data --console-address ":9001"
```

### 5️⃣ (Optional) Install Ollama — for RAG Document Q&A
```powershell
# Download from https://ollama.ai/download, then:
ollama pull qwen2.5:3b

OR
```powershell
curl -fsSL https://ollama.com/install.sh | sh
# then:
ollama pull qwen2.5:3b
```

### 6️⃣ Create `.env` File
Copy the template from **Section 10 > Step 6** of this README. Save as `centurion_core/.env`

### 7️⃣ Launch the App
**Terminal 1 — Streamlit UI:**
```powershell
streamlit run app.py
```
Opens at: **http://localhost:9000** — App login with `admin` / `admin123`, Minio login: `minioadmin` / `minioadmin123`

**Terminal 2 (optional) — FastAPI REST API:**
```powershell
python run_api.py --port 9001
```
API docs at: **http://localhost:9001/docs**

### ✅ Verify Everything Works
- [ ] Streamlit opens at http://localhost:9000
- [ ] Login succeeds with `admin` / `admin123`
- [ ] No database errors in console
- [ ] Run a quick analysis with 2 tickers (AAPL, MSFT) — should complete in <2 min
- [ ] Check **History** tab — results persist to PostgreSQL

**Stuck?** Jump to **Section 13: Troubleshooting** or **Section 10: Installation** for detailed setup.

---

## Table of Contents

0. [⚡ Quick Start](#-quick-start-5-minutes)
1. [Architecture Overview](#1-architecture-overview)
2. [Core Analysis Engine](#2-core-analysis-engine)
3. [Strategy Backtesting](#3-strategy-backtesting)
4. [Live Trading — Zerodha Kite Connect](#4-live-trading--zerodha-kite-connect)
5. [RAG Document Intelligence](#5-rag-document-intelligence)
6. [Database Layer](#6-database-layer)
7. [Object Storage (MinIO)](#7-object-storage-minio)
8. [Interactive Web Interface](#8-interactive-web-interface)
9. [Project Structure](#9-project-structure)
10. [Installation (Detailed)](#10-installation)
11. [Usage Guide](#11-usage-guide)
12. [API Reference](#12-api-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Dependencies](#14-dependencies)
15. [Changelog](#15-changelog)

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

### Real-time Streaming Architecture

Push-based tick distribution via Kite WebSocket (KiteTicker) with an internal event dispatcher:

| Component | Purpose |
|-----------|--------|
| `webhooks/ticker.py` | `KiteWebSocketService` — manages KiteTicker connection, batch-flushes ticks every 0.5 s |
| `webhooks/dispatcher.py` | `WebhookDispatcher` — singleton fan-out to subscribers via ThreadPoolExecutor |
| `webhooks/handlers.py` | `DBTickHandler` (PostgreSQL), `UITickCache` (Streamlit), `NSEMarketStatusMonitor`, `SessionWatchdog` |
| `webhooks/alert_engine.py` | `PriceAlertEngine` — evaluates price/volume/change conditions on every tick batch |
| `webhooks/timescale_handler.py` | `TimescaleTickHandler` — writes raw ticks to a hypertable; continuous aggregates for 1m/5m/15m/1h OHLC |
| `webhooks/service.py` | `WebhookService` — orchestrator that wires all components at startup |

**Streaming endpoints** (FastAPI):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/stream/sse` | Server-Sent Events tick stream (optional `?symbols=` filter) |
| `WS` | `/stream/ws` | WebSocket proxy — subscribe/unsubscribe/ping protocol |
| `POST` | `/stream/postback` | Kite order postback receiver (SHA-256 checksum verification) |
| `GET` | `/stream/ohlc/{symbol}` | OHLC bars from TimescaleDB continuous aggregates |
| `CRUD` | `/stream/alerts` | Price alert management (create, list, delete) |
| `GET` | `/stream/status` | Full streaming pipeline status |

### Key Features
- Auto-refresh every 30 seconds via `@st.fragment(run_every=...)`
- Market status pill indicators from NSE API (pre-open, live, post-market)
- Batch quote fetching (200 symbols/batch)
- Option chain: expiry discovery (45 days + monthly), Sensibull-style colouring, ATM highlighting, PCR metric
- Price alerts: `price_above`, `price_below`, `change_pct_above`, `change_pct_below`, `volume_above` with desktop notifications
- All-combinations pairs trading: C(n,2) pair analysis when >2 tickers provided

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
│   ├── trading/                  # Order service + RSI strategy
│   └── webhooks/                 # Real-time streaming infrastructure
│       ├── ticker.py             # KiteWebSocketService (KiteTicker wrapper)
│       ├── dispatcher.py         # WebhookDispatcher (in-process event fan-out)
│       ├── handlers.py           # DBTickHandler, UITickCache, NSEMarketStatusMonitor
│       ├── alert_engine.py       # PriceAlertEngine (condition-based alerts)
│       ├── timescale_handler.py  # TimescaleDB tick writer + OHLC aggregates
│       ├── service.py            # WebhookService orchestrator
│       └── events.py             # EventType enum, TickData, WebhookEvent
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
├── api/                          # FastAPI REST API layer
│   ├── main.py                   # App factory, auth-gated /docs
│   ├── auth.py                   # Token signing, login/logout
│   ├── dependencies.py           # Dependency injection (DB, Kite, RAG)
│   ├── schemas/                  # Pydantic v2 request/response models
│   │   ├── common.py             # Shared: SuccessResponse, Pagination
│   │   ├── us_stocks.py          # Analysis, news, signals, backtest
│   │   ├── ind_stocks.py         # Kite auth, quotes, orders, options
│   │   ├── rag.py                # Ingest, query, evaluation
│   │   ├── crypto.py             # Prices, backtest, strategies
│   │   └── streaming.py          # SSE, WebSocket, Postback, OHLC, Alerts
│   └── routers/                  # Route modules (50 endpoints)
│       ├── health.py             # GET /health
│       ├── us_stocks.py          # 9 endpoints
│       ├── ind_stocks.py         # 11 endpoints
│       ├── rag.py                # 10 endpoints
│       ├── crypto.py             # 4 endpoints
│       └── streaming.py          # 9 endpoints (SSE, WS, postback, OHLC, alerts, status)
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

Complete step-by-step setup guide for fresh machine deployment.

### Prerequisites & System Check

| Component | Requirement | Windows Install |
|---|---|---|
| **Python** | 3.10+ | https://www.python.org (add to PATH) |
| **PostgreSQL** | 14+ | https://www.postgresql.org/download OR Docker |
| **Docker** | 20+ | https://www.docker.com/products/docker-desktop |
| **Git** | Latest | https://git-scm.com |
| **Ollama** (RAG only) | Latest | https://ollama.ai (optional) |

**Port Availability Check** — ensure these ports are free:

```powershell
# Windows PowerShell (Admin)
$ports = @(9000, 9001, 9002, 9003, 9004, 11434)
foreach ($port in $ports) {
    $connection = Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet
    if ($connection) {
        Write-Host "⚠️  Port $port is in use" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Port $port is available" -ForegroundColor Green
    }
}
```

If ports are in use, either:
1. Kill the process: `Get-Process -Name processname | Stop-Process`
2. Or update `.env` to use different ports

---

### Step 1: Clone Repository & Setup Python Environment

```powershell
# Clone the repository
git clone https://github.com/srees16/centurion_core.git
cd centurion_core

# Create and activate virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Verify Python version
python --version  # should be 3.10+

# Install Python dependencies (installs DistilBERT ~250MB on first run)
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed streamlit==1.X.X psycopg2-binary==2.9.X ...
```

---

### Step 2: Set Up PostgreSQL Database

Choose **Option A (Docker)** or **Option B (Local PostgreSQL)**.

#### **Option A: PostgreSQL via Docker** (Recommended)

```powershell
# Pull and run PostgreSQL container
docker run -d `
  --name centurion-postgres `
  -p 9003:5432 `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=superadmin1 `
  -e POSTGRES_DB=centurion_rag `
  postgres:15

# Wait 5 seconds for container to start
Start-Sleep -Seconds 5

# Verify container is running
docker ps | findstr centurion-postgres

# Check logs for startup
docker logs centurion-postgres
```

#### **Option B: Local PostgreSQL Installation**

```powershell
# Verify PostgreSQL is installed and running
# Windows Service Status:
Get-Service postgresql-x64-15

# If not running, start it:
Start-Service postgresql-x64-15

# Or use the Windows PostgreSQL CLI:
& "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres -c "SELECT version();"
```

---

### Step 3: Create Database & Initialize Schema

```powershell
# Navigate to project directory
cd centurion_core

# Run the database setup script
# This will create all tables automatically
python setup_database.py
```

**Expected output:**
```
✓ Database connection successful
✓ Database tables created successfully
✓ Database service layer ready
✅ Database setup completed successfully!
```

**If this fails:**
- Check PostgreSQL is running: `docker ps | findstr centurion-postgres`
- Verify port 9003 is listening: `Test-NetConnection -ComputerName localhost -Port 9003`
- Check password matches in `.env`

---

### Step 4: Set Up MinIO Object Storage (for Backtest Charts)

```powershell
# Pull and run MinIO container
docker run -d `
  --name centurion-minio `
  -p 9004:9000 `
  -p 9002:9001 `
  -e MINIO_ROOT_USER=minioadmin `
  -e MINIO_ROOT_PASSWORD=minioadmin123 `
  minio/minio:latest server /data --console-address ":9001"

# Wait for startup
Start-Sleep -Seconds 5

# Verify container
docker ps | findstr centurion-minio

# Create the bucket (optional — created auto on first use)
docker exec centurion-minio mc mb minio/centurion-backtests
```

**Access MinIO Console:**
- **URL**: http://localhost:9002
- **Username**: minioadmin
- **Password**: minioadmin123

---

### Step 5: Set Up Ollama (Optional, for RAG Pipeline)

If you plan to use the RAG document Q&A feature, install Ollama:

```powershell
# Download from https://ollama.ai/download
# Or via PowerShell:
Invoke-WebRequest -Uri "https://ollama.ai/download/OllamaSetup.exe" -OutFile OllamaSetup.exe
.\OllamaSetup.exe

# After installation, download the default model
ollama pull qwen2.5:3b

# Verify Ollama is running (should listen on port 11434)
Test-NetConnection -ComputerName localhost -Port 11434
```

---

### Step 6: Configure Environment Variables

Create `.env` file in the `centurion_core/` root directory with **all** required variables:

```ini
# ═══════════════════════════════════════════════════════════════════
# CRITICAL: Copy this entire block to .env (replace /path with actual)
# ═══════════════════════════════════════════════════════════════════

# ─── Streamlit App ─────────────────────────────────────────────────
STREAMLIT_SERVER_PORT=9000

# ─── FastAPI Backend ──────────────────────────────────────────────
API_PORT=9001

# ─── PostgreSQL (US Stocks Analysis & Backtesting) ──────────────────
CENTURION_DB_HOST=localhost
CENTURION_DB_PORT=9003
CENTURION_DB_NAME=centurion_rag
CENTURION_DB_USER=postgres
CENTURION_DB_PASSWORD=superadmin1
CENTURION_DB_ENABLED=true

# ─── Separate PostgreSQL for Kite Connect (Live Trading) ────────────
KITE_DB_HOST=localhost
KITE_DB_PORT=9003
KITE_DB_NAME=livestocks_ind
KITE_DB_USER=postgres
KITE_DB_PASSWORD=superadmin1

# ─── MinIO (S3-compatible Object Storage) ──────────────────────────
MINIO_ENDPOINT=localhost:9004
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_SECURE=false
MINIO_BUCKET=centurion-backtests
MINIO_ENABLED=true

# ─── Zerodha Kite Connect (Live Indian Trading) ────────────────────
# Obtain from Zerodha – https://kite.zerodha.com/app/settings/api
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_USER_ID=your_user_id_here
ZERODHA_PASSWORD=your_password_here

# ─── KiteConnect Connection Pool ──────────────────────────────────
KITE_POOL_MAXSIZE=40

# ─── RAG Document Pipeline ────────────────────────────────────────
CENTURION_RAG_LLM_URL=http://localhost:11434
RAG_MODEL=qwen2.5:3b
CENTURION_RAG_CHROMA_DIR=./chroma_store
CENTURION_RAG_EMBED_MODEL=BAAI/bge-base-en-v1.5
CENTURION_RAG_CONTEXT_TOKEN_BUDGET=1200
CENTURION_RAG_MAX_CONTEXT_CHUNKS=8
CENTURION_RAG_TOP_K=15
CENTURION_RAG_SIMILARITY_THRESHOLD=0.70
CENTURION_RAG_LLM_NUM_CTX=2048
CENTURION_RAG_LLM_NUM_PREDICT=400
CENTURION_RAG_LLM_MAX_TOKENS=400
CENTURION_RAG_LLM_TEMPERATURE=0.2
CENTURION_RAG_LLM_FIRST_TOKEN_TIMEOUT=300
CENTURION_RAG_LLM_CHUNK_TIMEOUT=30
CENTURION_RAG_QUERY_BUDGET=300
CENTURION_RAG_QUERY_REWRITE=false
CENTURION_RAG_STREAMING=true
CENTURION_RAG_FAQ_ENABLED=false
RAG_FAST_MODE=false

# ─── Authentication ───────────────────────────────────────────────
CENTURION_DEFAULT_ADMIN_PASSWORD=admin123
CENTURION_DEFAULT_ANALYST_PASSWORD=analyst123

# ─── Optional: Cloud LLM (Alternative to Ollama) ───────────────────
# Uncomment to use Claude or OpenAI instead of Ollama
# CENTURION_RAG_LLM_PROVIDER=anthropic  # or "openai"
# ANTHROPIC_API_KEY=your_claude_key_here
# OPENAI_API_KEY=your_openai_key_here
```

**Verify .env is in the correct location:**
```powershell
Test-Path centurion_core\.env  # Should return True
```

---

### Step 7: Verify All Services Are Running

```powershell
# Check containers
docker ps

# Expected output:
# centurion-postgres     postgres:15      Up 2 minutes    0.0.0.0:9003->5432/tcp
# centurion-minio        minio:latest     Up 2 minutes    0.0.0.0:9004->9000/tcp, 0.0.0.0:9002->9001/tcp

# Test PostgreSQL connection
python -c "
import psycopg2
try:
    conn = psycopg2.connect('host=localhost port=9003 user=postgres password=superadmin1 dbname=centurion_rag')
    print('✓ PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f'✗ PostgreSQL error: {e}')
"

# Test MinIO connection
python -c "
from minio import Minio
try:
    client = Minio('localhost:9004', access_key='minioadmin', secret_key='minioadmin123', secure=False)
    client.bucket_exists('centurion-backtests')
    print('✓ MinIO connection successful')
except Exception as e:
    print(f'✗ MinIO error: {e}')
"

# Test Ollama (if using RAG)
# Test-NetConnection -ComputerName localhost -Port 11434
```

---

### Step 8: Launch the Application

**Terminal 1 — Streamlit UI:**

```powershell
cd centurion_core
.\myenv\Scripts\Activate.ps1
streamlit run app.py
```

Opens at: **http://localhost:9000**

**Terminal 2 — FastAPI REST API (optional):**

```powershell
cd centurion_core
.\myenv\Scripts\Activate.ps1
python run_api.py --port 9001
```

API docs at: **http://localhost:9001/docs** (login required)

---

### Step 9: Login & Verify Application

1. Open http://localhost:9000 in your browser
2. Login with default credentials:
   - **Username**: `admin`
   - **Password**: `admin123`
3. Navigate to **Main** page — ensure no error messages appear
4. Try a quick analysis with 2-3 tickers (e.g., AAPL, MSFT, GOOGL)
5. Check **History** → **Analysis Runs** to verify database persistence

**Expected UI state:**
- No red error boxes
- Database health check passes
- Tickers load from cache successfully
- Analysis completes within 2 minutes for 3 tickers

---

### Troubleshooting Setup Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `connection to server ... failed` | PostgreSQL not running | `docker ps` and check container status |
| `database "centurion_rag" does not exist` | Setup script didn't run | Run `python setup_database.py` again |
| `[Errno 48] Address already in use` | Port conflict (9000/9004) | Check `Test-NetConnection` or change `.env` ports |
| `ModuleNotFoundError: No module named 'streamlit'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `Connection refused to port 9003` | PostgreSQL password mismatch | Verify `CENTURION_DB_PASSWORD=superadmin1` in `.env` |
| `MinIO bucket not found` | Bucket not created | Run `docker exec centurion-minio mc mb minio/centurion-backtests` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | SSL cert issue (news scraping) | Usually auto-resolved; check internet connection |
| `No module named 'torch'` | Heavy dependencies download | First run is slow (~5 min); be patient or pre-install: `pip install torch` |

---

### Optional: Stop Services

```powershell
# Stop containers (keep data)
docker stop centurion-postgres centurion-minio

# Remove containers (lose data)
docker rm centurion-postgres centurion-minio

# Remove images
docker rmi postgres:15 minio/minio:latest

# Deactivate virtual environment
deactivate
```

---

### Production Deployment

For production, see [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md) for:
- AWS EC2 / Azure VM setup
- Kubernetes (k8s) manifests
- SSL/TLS certificates
- Database backups and recovery
- Load balancing & horizontal scaling

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

### REST API (FastAPI)

A full REST API runs alongside the Streamlit UI on a separate port (default `9001`).

**Interactive docs** — **http://localhost:9001/docs** (Swagger UI) and **http://localhost:9001/redoc** (ReDoc) are available after authenticating. On first visit you are redirected to a login page; use the same credentials as the Streamlit app (e.g. `admin` / `admin123`). A signed session cookie (8-hour TTL) keeps you logged in.

| Module | Prefix | Endpoints | Examples |
|--------|--------|-----------|----------|
| Health | `/api/health` | 1 | DB, RAG, Kite status check |
| US Stocks | `/api/us-stocks` | 9 | `/analysis`, `/news`, `/sentiment`, `/backtest`, `/strategies` |
| Indian Stocks | `/api/ind-stocks` | 11 | `/auth`, `/quotes`, `/orders`, `/positions`, `/option-chain` |
| RAG Pipeline | `/api/rag` | 10 | `/ingest`, `/query`, `/collection/stats`, `/evaluate` |
| Crypto | `/api/crypto` | 4 | `/prices`, `/backtest`, `/strategies` |
| Streaming | `/stream` | 9 | `/sse`, `/ws`, `/postback`, `/ohlc/{symbol}`, `/alerts`, `/status` |

```powershell
# Launch the API server
python run_api.py --port 9001

# Or via uvicorn directly
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 9001
```

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
| Connection refused on port 9004 | `cd deployment && docker compose up -d minio` |

### General

| Symptom | Fix |
|---------|-----|
| Import errors | `pip install -r requirements.txt --upgrade` |
| Port in use | `streamlit run app.py --server.port 9005` |
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
| **Auth** | pyyaml ≥ 6.0, itsdangerous |
| **Notifications** | plyer |
| **API** | fastapi, uvicorn[standard], python-multipart |
---

## 15. Changelog

### 2026-02-28

- **FastAPI REST API** — 50 JSON endpoints across 6 modules with Pydantic v2 schemas, auth-gated `/docs` (signed session cookie, 8-hour TTL)
- **Real-time Streaming** — SSE tick stream, WebSocket proxy, Kite Postback receiver, TimescaleDB OHLC aggregates (1m/5m/15m/1h), price alert engine with CRUD endpoints
- **MinIO Auto-Bucket** — `centurion-backtests` bucket created automatically on first use; `MinIOService.ensure_bucket_ready()`
- **Pairs Trading All-Combinations** — C(n,2) pair analysis when >2 tickers provided
- **Lazy Sentiment Loading** — DistilBERT model deferred to first `analyze()` call (class-level singleton)

---

## ⚠️ Disclaimer

This software is provided for **educational and informational purposes only**. It does not constitute financial advice, investment recommendations, or professional trading guidance. Stock trading involves substantial risk of loss. Always consult qualified financial advisors before making investment decisions. Use at your own risk.

---

**Ready to get started? Run `streamlit run app.py` and begin analysing! 🚀📈**
