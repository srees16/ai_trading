# Centurion Capital LLC — Comprehensive Codebase Inventory

## 1. Architecture & Application Flow

### Entry Point: `app.py` (75 lines)
- **Framework**: Streamlit (`page_title="Centurion Capital LLC"`, `layout="wide"`, `initial_sidebar_state="collapsed"`)
- **Boot sequence**: `apply_custom_styles()` → `initialize_session_state()` → `check_authentication()` → `render_user_menu()` → page router
- **Page routing** via `st.session_state.current_page`:
  - `'main'` → `render_main_page()` (default)
  - `'analysis'` → `render_analysis_page()`
  - `'fundamental'` → `render_fundamental_page()`
  - `'backtesting'` → `render_backtesting_page()`
  - `'crypto'` → `render_crypto_page()`
  - `'history'` → `render_history_page()`
- All page imports are **deferred** (inside route branches) to keep startup fast

### Orchestrator: `main.py` (~150 lines)
- `AlgoTradingSystem` class: initializes all core components
- Pipeline (sequential per-ticker): scrape news → analyze sentiment → notify high-confidence → calculate metrics + generate signals → save results → display summary
- Components instantiated: `NewsAggregator`, `SentimentAnalyzer`, `MetricsCalculator`, `DecisionEngine`, `NotificationManager`, `StorageManager`

---

## 2. Configuration: `config.py` (~140 lines)

All values overridable via `CENTURION_` prefixed environment variables.

### Sentiment
| Constant | Value |
|---|---|
| `SENTIMENT_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` |
| `SENTIMENT_HIGH_CONFIDENCE_THRESHOLD` | `0.85` |

### Networking
| Constant | Value |
|---|---|
| `REQUEST_TIMEOUT` | `10` seconds |
| `MAX_CONCURRENT_REQUESTS` | `5` |

### Cache TTLs
| Constant | Value |
|---|---|
| `CACHE_TTL_MINUTES` | `30` |
| `NEWS_CACHE_TTL_MINUTES` | `15` |
| `METRICS_CACHE_TTL_MINUTES` | `30` |

### Technical Indicator Periods
| Constant | Value |
|---|---|
| `RSI_PERIOD` | `14` |
| `MACD_FAST` | `12` |
| `MACD_SLOW` | `26` |
| `MACD_SIGNAL` | `9` |
| `BOLLINGER_PERIOD` | `20` |
| `BOLLINGER_STD` | `2` |

### Decision Engine Weights & Thresholds
| Constant | Value |
|---|---|
| `SENTIMENT_WEIGHT` | `0.4` |
| `FUNDAMENTAL_WEIGHT` | `0.3` |
| `TECHNICAL_WEIGHT` | `0.3` |
| `STRONG_BUY_THRESHOLD` | `0.7` |
| `BUY_THRESHOLD` | `0.4` |
| `SELL_THRESHOLD` | `-0.4` |
| `STRONG_SELL_THRESHOLD` | `-0.7` |

### Database
| Constant | Value |
|---|---|
| `DB_HOST` | `localhost` |
| `DB_PORT` | `5432` |
| `DB_NAME` | `centurion_trading` |
| `DB_USER` | `centurion` |
| `DB_POOL_SIZE` | `10` |
| `DB_MAX_OVERFLOW` | `20` |
| `DB_POOL_TIMEOUT` | `30` |
| `DB_POOL_RECYCLE` | `1800` |
| `TIMESCALEDB_CHUNK_INTERVAL` | `7 days` |
| `DB_RETENTION_DAYS` | `365` |

### Defaults
| Constant | Value |
|---|---|
| `HISTORICAL_DAYS` | `365` |
| `NOTIFICATION_DURATION` | `10` seconds |
| `DEFAULT_TICKERS` | `AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, V, WMT` |

### News Categorization Keywords
- `BREAKING_KEYWORDS`: breaking, urgent, alert, emergency, crash, plunge, surge, soar, halt
- `DEALS_KEYWORDS`: merger, acquisition, acquire, deal, buyout, takeover, partnership, joint venture
- `MACRO_KEYWORDS`: fed, federal reserve, interest rate, inflation, gdp, unemployment, treasury, monetary policy
- `EARNINGS_KEYWORDS`: earnings, revenue, profit, quarterly, annual report, guidance, forecast, beat, miss, eps

---

## 3. Data Models: `models.py` (~160 lines)

### Enums
| Enum | Values |
|---|---|
| `NewsCategory` | `BREAKING`, `DEALS_MA`, `MACRO_ECONOMIC`, `EARNINGS`, `GENERAL` |
| `SentimentLabel` | `POSITIVE`, `NEGATIVE`, `NEUTRAL` |
| `DecisionTag` | `STRONG_BUY`, `BUY`, `HOLD`, `SELL`, `STRONG_SELL` |

### Dataclasses
**`NewsItem`**: `title`, `summary`, `url`, `timestamp`, `source`, `ticker`, `category` (NewsCategory), `sentiment_score` (float), `sentiment_label` (SentimentLabel), `sentiment_confidence` (float)

**`StockMetrics`**: `ticker`, `timestamp`, `peg_ratio`, `roe`, `eps`, `free_cash_flow`, `dcf_value`, `intrinsic_value`, `altman_z_score`, `beneish_m_score`, `piotroski_f_score`, `rsi`, `macd`, `macd_signal`, `macd_histogram`, `fibonacci_levels` (dict), `bollinger_upper`, `bollinger_middle`, `bollinger_lower`, `max_drawdown`, `current_price`

**`TradingSignal`**: `news_item` (NewsItem), `metrics` (StockMetrics), `decision` (DecisionTag), `decision_score` (float), `reasoning` (str), `timestamp`

---

## 4. Utilities: `utils.py` (~100 lines)

- `parse_ticker_csv(content)`: Parses CSV text, supports multiple column layouts (with/without header, "Ticker"/"Symbol"/"Stock" columns)
- `validate_tickers(tickers)`: Validates 1-5 char alphanumeric symbols; returns `(valid, invalid)` tuple
- `create_sample_csv()`: Returns sample CSV string for download button

---

## 5. Authentication: `auth/authenticator.py` (556 lines)

### Credential Storage
- YAML file: `auth/credentials.yaml`
- Password hashing: SHA-256 via `hashlib.sha256`, comparison via `hmac.compare_digest`
- Default users:
  - `admin` / `admin123` (role: `admin`)
  - `analyst` / `analyst123` (role: `analyst`)

### Session Management
| Parameter | Value |
|---|---|
| Session timeout (absolute) | 60 minutes |
| Inactivity timeout | 30 minutes |
| Max login attempts | 3 |

### `Authenticator` Class
- `login()`: Renders Streamlit login form with Centurion branding, handles attempts
- `logout()`: Clears session state
- `_check_session_timeout()`: Enforces both absolute and inactivity timeouts
- `_heartbeat()`: Auto-update `last_activity` on page interaction
- `add_user(username, password, role)`: Appends to YAML (admin only)
- `change_password(username, old_password, new_password)`: Validates old, writes new hash

### Module-Level Functions
- `check_authentication()`: Gate; redirects to login form if unauthenticated
- `render_user_menu()`: Sidebar user info, system health, logout button
- `check_system_health()`: Returns dict with `database`, `strategies`, `scrapers`, `storage` status
- `logout()`: Clears all `auth_*` session state keys and reruns

---

## 6. Database Layer

### 6.1 Connection: `database/connection.py` (334 lines)

- **`DatabaseConfig`**: Reads env vars (`CENTURION_DB_HOST`, `CENTURION_DATABASE_URL`, etc.)
- **`DatabaseManager`**: Singleton pattern
  - Engine: `create_engine(url, poolclass=QueuePool, pool_pre_ping=True, pool_size=10, max_overflow=20, pool_timeout=30, pool_recycle=1800, connect_args={'application_name': 'CenturionCapital'})`
  - `get_session()`: Context manager with auto commit/rollback
  - `session_scope()`: Same as `get_session()` (alias)
  - `health_check()`: Returns connectivity status, TimescaleDB version, pool stats (size, checked_in, checked_out, overflow)
  - `initialize_database()`: `Base.metadata.create_all()` + TimescaleDB hypertable creation on 3 tables

### 6.2 ORM Models: `database/models.py` (615 lines)

**Complete Table Schema:**

#### `analysis_runs`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | `uuid4` |
| `run_type` | String(50) | |
| `status` | Enum | pending/running/completed/failed |
| `tickers` | ARRAY(String) | |
| `parameters` | JSONB | |
| `total_signals` | Integer | |
| `total_news_items` | Integer | |
| `started_at` | DateTime(tz) | |
| `completed_at` | DateTime(tz) | |
| `duration_seconds` | Float | |
| `error_message` | Text | |
| `error_traceback` | Text | |
| `user_id` | String(100) | |
| `source` | String(50) | |
| `created_at` | DateTime(tz) | server_default=now() |
| `updated_at` | DateTime(tz) | onupdate=now() |

#### `news_items`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `analysis_run_id` | FK → analysis_runs | |
| `ticker` | String(20) | indexed |
| `title` | Text | |
| `content` | Text | |
| `url` | Text | |
| `source` | String(100) | indexed |
| `author` | String(200) | |
| `published_at` | DateTime(tz) | |
| `scraped_at` | DateTime(tz) | |
| `sentiment_label` | Enum | POSITIVE/NEGATIVE/NEUTRAL |
| `sentiment_confidence` | Float | |
| `sentiment_scores` | JSONB | |
| `keywords` | ARRAY(String) | |
| `entities` | JSONB | |
| `content_hash` | String(64) | unique, SHA-256 |
| `created_at` / `updated_at` | DateTime(tz) | |

#### `stock_signals`
| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `analysis_run_id` | FK → analysis_runs | |
| `news_item_id` | FK → news_items | |
| `ticker` | String(20) | indexed |
| `decision` | Enum | STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL |
| `decision_score` | Float | |
| `reasoning` | Text | |
| `current_price` | Numeric(12,4) | |
| `price_change_pct` | Float | |
| `volume` | Integer | |
| `rsi` | Float | |
| `macd` / `macd_signal` | Float | |
| `sma_20` / `sma_50` / `sma_200` | Numeric(12,4) | |
| `bollinger_upper` / `bollinger_lower` | Numeric(12,4) | |
| `altman_z_score` | Float | |
| `beneish_m_score` | Float | |
| `piotroski_f_score` | Integer | |
| `decision_factors` | JSONB | |
| `tags` | ARRAY(String) | |
| `created_at` / `updated_at` | DateTime(tz) | |

#### `fundamental_metrics`
| Column | Type |
|---|---|
| `id` | UUID PK |
| `ticker` | String(20), indexed |
| `recorded_at` | DateTime(tz) |
| `current_price` | Numeric(12,4) |
| `market_cap` / `enterprise_value` | Numeric(20,2) |
| `pe_ratio` / `forward_pe` / `peg_ratio` | Float |
| `pb_ratio` / `ps_ratio` / `ev_to_ebitda` / `ev_to_revenue` | Float |
| `altman_z_score` / `beneish_m_score` | Float |
| `piotroski_f_score` | Integer |
| `profit_margin` / `operating_margin` / `return_on_equity` / `return_on_assets` | Float |
| `current_ratio` / `quick_ratio` / `debt_to_equity` / `debt_to_assets` / `interest_coverage` | Float |
| `revenue_growth` / `earnings_growth` / `dividend_yield` / `payout_ratio` | Float |
| `raw_financials` | JSONB |
| `data_source` | String(50) |

#### `backtest_results`
| Column | Type |
|---|---|
| `id` | UUID PK |
| `strategy_id` / `strategy_name` / `strategy_category` / `strategy_version` | String |
| `tickers` | ARRAY(String) |
| `start_date` / `end_date` | DateTime(tz) |
| `initial_capital` | Numeric(15,2) |
| `parameters` | JSONB |
| `success` | Boolean |
| `error_message` | Text |
| `total_return` / `annualized_return` / `sharpe_ratio` / `sortino_ratio` / `max_drawdown` / `calmar_ratio` | Float |
| `total_trades` / `winning_trades` / `losing_trades` | Integer |
| `win_rate` / `avg_win` / `avg_loss` / `profit_factor` | Float |
| `metrics` / `signals` / `equity_curve` | JSONB |
| `execution_time_seconds` | Float |
| `data_points_processed` | Integer |

#### `backtest_trades` (detail, FK → backtest_results CASCADE)
Columns: `id`, `backtest_id`, `trade_number`, `trade_type` (String10), `ticker`, `entry_date`, `exit_date`, `entry_price` (Numeric12,4), `exit_price` (Numeric12,4), `quantity` (Numeric15,4), `pnl` (Float), `pnl_pct` (Float), `holding_period_days` (Int)

#### `backtest_equity_points` (detail, FK CASCADE)
Columns: `id`, `backtest_id`, `point_date`, `portfolio_value` (Numeric15,2), `drawdown` (Float), `benchmark_value` (Numeric15,2)

#### `backtest_daily_returns` (detail, FK CASCADE)
Columns: `id`, `backtest_id`, `return_date`, `daily_return` (Float), `cumulative_return` (Float)

#### `strategy_performance_summary` (Gold layer, materialized)
Columns: `id`, `strategy_id` (unique), `strategy_name`, `total_backtests`, `successful_backtests`, `avg_return`, `best_return`, `worst_return`, `median_return`, `avg_sharpe`, `avg_sortino`, `avg_max_drawdown`, `avg_calmar`, `avg_win_rate`, `avg_profit_factor`, `avg_total_trades`, `last_backtest_at`, `last_refreshed_at`

#### `user_watchlists`
Columns: `id`, `user_id`, `name`, `description`, `tickers` (ARRAY), `is_default` (Bool)

#### `alert_configurations`
Columns: `id`, `user_id`, `ticker`, `alert_type`, `conditions` (JSONB), `notification_channels` (ARRAY), `is_active` (Bool), `last_triggered_at`, `trigger_count`

#### `raw_scraped_news` (Bronze/raw layer)
Columns: `id`, `ticker`, `source`, `scraper_name`, `raw_title`, `raw_content`, `raw_url`, `raw_author`, `raw_published_at`, `content_hash` (unique), `is_processed` (Bool), `processed_at`, `enriched_news_id` (FK → news_items), `ingested_at`

#### `data_freshness`
Columns: `id`, `ticker`, `data_type`, `last_fetched_at`, `next_refresh_at`, `fetch_count`, `avg_fetch_seconds`, `last_fetch_seconds`, `last_record_count`, `consecutive_errors`, `last_error`, `last_error_at`

**TimescaleDB Hypertables** (chunk interval: 7 days):
- `stock_signals` on `created_at`
- `fundamental_metrics` on `recorded_at`
- `news_items` on `published_at`

### 6.3 Service Layer: `database/service.py` (836 lines)

`DatabaseService` (Singleton) — unified API:
- `start_analysis_run(run_type, tickers, params, user_id, source)` → UUID
- `complete_analysis_run(run_id, total_signals, total_news)` — sets duration
- `fail_analysis_run(run_id, error_msg, traceback)`
- `save_signals(run_id, signals_data)` — bulk insert
- `save_news_items(run_id, news_data)` — with SHA-256 content_hash dedup
- `save_fundamental_metrics(metrics_data)` — upsert (INSERT ON CONFLICT UPDATE)
- `save_backtest_result(backtest_data)` — normalizes into `backtest_results` + `backtest_trades` + `backtest_equity_points` + `backtest_daily_returns`, then refreshes `strategy_performance_summary`
- `check_freshness(ticker, data_type)` → dict with `is_stale`, `last_fetched_at`
- `record_fetch(ticker, data_type, seconds, record_count)` — upsert with running average
- `record_error(ticker, data_type, error_msg)` — increments `consecutive_errors`
- `save_raw_news(raw_items)` — bronze layer bulk insert
- `save_complete_analysis(run_id, signals, news, fundamentals)` — combined transaction

### 6.4 Repositories: `database/repositories/` (7 files)

**`BaseRepository`** (generic CRUD): `create`, `create_many`, `get_by_id`, `get_all` (pagination), `update`, `delete`, `count`, `exists`

**`AnalysisRepository`**: `create_run`, `start_run`, `complete_run`, `fail_run`, `get_recent_runs(limit, days)`

**`SignalRepository`**: `get_by_ticker`, `get_by_decision`, `get_top_signals(limit, min_score)`, `get_by_analysis_run(run_id)`, `get_recent_signals_summary`, `get_ticker_signal_history(ticker, days)`

**`NewsRepository`**: `get_by_ticker`, `get_by_source`, `get_by_sentiment`, `check_duplicate(content_hash)` — SHA-256, `create_with_dedup`, `get_sentiment_summary`, `get_source_statistics`

**`FundamentalRepository`**: `get_latest_by_ticker`, `get_history(ticker, days)`, `get_multiple_tickers(tickers)`, `upsert(data)` — INSERT ON CONFLICT

**`BacktestRepository`**: `get_by_strategy`, `get_by_ticker`, `get_top_performers(metric, limit)`, `get_strategy_summary(strategy_id)`, `get_all_strategies_summary` (GROUP BY), `get_recent_backtests(days, limit)`, `compare_strategies(strategy_ids)`

**`FreshnessRepository`**: `get_freshness`, `is_stale(ticker, data_type, max_age_minutes)`, `record_fetch` (upsert with running average), `record_error`, `get_stale_tickers`, `get_ticker_freshness`

---

## 7. Scrapers

### 7.1 Base: `scrapers/__init__.py` (~140 lines)
`BaseNewsScraper` (ABC):
- Properties: `source_name`, `base_url`
- Abstract: `fetch_news(ticker) → list[NewsItem]`
- `_fetch_html(url)`: aiohttp with `ssl=False`, `TCPConnector(limit=100)`, timeout=`Config.REQUEST_TIMEOUT`
- `_parse_html(html, parser)`: BeautifulSoup with lxml
- `_extract_text(element)`: Cleans text
- `_categorize_news(text)`: Checks keyword lists from Config → `NewsCategory`

### 7.2 Cache: `scrapers/cache.py` (376 lines)
- **`_SourceRateLimiter`**: `base_delay=0.5s`, `max_delay=30s`, exponential backoff `2^consecutive_errors` capped at 6
- **`_ContentDeduplicator`**: SHA-256 of `title.lower() + "|" + url`, pre-loads from DB (last 2 days)
- **`ScraperCache`** (Singleton): Integrates `SessionCache` + `DataFreshness` DB + rate limiting + dedup

### 7.3 Aggregator: `scrapers/aggregator.py` (~230 lines)
`NewsAggregator`:
- Instantiates all 5 scrapers: `YahooFinanceScraper`, `FinvizScraper`, `InvestingComScraper`, `TradingViewScraper`, `WallStreetBetsScraper`
- `asyncio.Semaphore(5)` for concurrency
- 3-layer caching: caller-provided `cached_news` → `ScraperCache` per-(ticker,source) → `DataFreshness` DB
- `_rate_limited_fetch()`: checks cache → adaptive delay from rate limiter → semaphore gate → fetch → dedup → store

### 7.4 Yahoo Finance: `scrapers/yahoo_finance.py` (~100 lines)
- Source: `yfinance` library (`yf.Ticker(ticker).news`), NOT HTTP scraping
- Extracts: `title`, `summary`/`description`, `canonicalUrl`/`clickThroughUrl`, `pubDate`/`providerPublishTime`
- Limit: 10 articles per ticker

### 7.5 Finviz: `scrapers/finviz.py` (~170 lines)
- URL: `https://finviz.com/quote.ashx?t={ticker}`
- HTTP: aiohttp `_fetch_html()`, optional Selenium for Elite auth
- Parses: `table.fullview-news-outer` → `tr` rows → `td[0]`=timestamp, `td[1]`=`a` link (title+href)
- Elite login: `https://elite.finviz.com/login.ashx` via Selenium headless Chrome (`FINVIZ_EMAIL`, `FINVIZ_PASSWORD` env vars)
- Limit: 10 articles

### 7.6 Investing.com: `scrapers/investing.py` (~65 lines)
- URL: `https://www.investing.com/search/?q={ticker}`
- HTTP: aiohttp `_fetch_html()` with custom headers (`User-Agent`, `Accept-Language`, etc.)
- Parses: `article.js-article-item` → `a.title` (title + href)
- Limit: 10 articles

### 7.7 TradingView: `scrapers/tradingview.py` (~100 lines)
- URL: `https://news-headlines.tradingview.com/v2/headlines?category=base&client=web&lang=en&limit=10&streaming=true&symbol={EXCHANGE}:{ticker}`
- HTTP: aiohttp, JSON API response
- Tries exchanges in order: `NASDAQ`, `NYSE`, `AMEX`
- Extracts: `title`, `shortDescription`/`description`, `link`/`storyPath`, `provider`, `published` (unix timestamp)
- Limit: 10 articles

### 7.8 WallStreetBets: `scrapers/wallstreetbets.py` (~200 lines)
- URL: `https://www.reddit.com/r/wallstreetbets/search.json?sort=hot&restrict_sr=on&t=day&q=flair%3A{flair}&limit=50`
- HTTP: aiohttp, Reddit public JSON API (no OAuth needed)
- `User-Agent`: `"Centurion/1.0 (WallStreetBets scraper; +https://github.com/centurion-capital)"`
- Flairs searched: `DD`, `Discussion`, `YOLO`, `Earnings Thread`, `Gain`, `Loss`, `News`, `Chart`
- Filtering: regex word-boundary match (`\b{ticker}\b`) or `${TICKER}` in `title + selftext`
- Enriched title format: `[WSB/{flair}] {title} (↑{score}, 💬{num_comments})`
- Rate limit: `1.5s` sleep between flair fetches

---

## 8. Sentiment Analysis: `sentiment/analyzer.py` (~100 lines)

- **Model**: `distilbert-base-uncased-finetuned-sst-2-english` (HuggingFace `transformers`)
- **Device**: CPU (`device=-1`)
- **Input**: `title + ". " + summary`, truncated to 512 characters
- **Output mapping**:
  - Label `POSITIVE` → `score = +confidence`
  - Label `NEGATIVE` → `score = -confidence`
- **Method**: `analyze(news_item)` → updates `news_item.sentiment_score`, `.sentiment_label`, `.sentiment_confidence` in-place

---

## 9. Metrics Calculator: `metrics/calculator.py` (659 lines)

Data source: `yfinance` (`stock.info`, `stock.history`, `stock.balance_sheet`, `stock.income_stmt`, `stock.cashflow`)

### Technical Indicators
| Indicator | Formula |
|---|---|
| **RSI** | Standard Wilder's: `gain_avg / loss_avg` over 14 periods; `RSI = 100 - 100/(1+RS)` |
| **MACD** | `EWM(12) - EWM(26)`; signal = `EWM(9)` of MACD; histogram = MACD - signal |
| **Bollinger Bands** | `SMA(20) ± 2×STD(20)` |
| **Fibonacci Levels** | Levels at `0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0` from `(high - low)` range |
| **Max Drawdown** | `(value - cummax) / cummax`, delegates to `strategies.utils.calculate_max_drawdown` |

### Fundamental Metrics

| Metric | Formula / Source |
|---|---|
| **DCF Value** | `terminal_value = FCF × (1 + 0.03) / (0.10 - 0.03)`; `DCF_per_share = terminal_value / shares_outstanding` |
| **Graham Intrinsic Value** | `IV = EPS × (8.5 + 2g) × 4.4 / 4.5` where `g` = earnings growth rate |
| **Altman Z-Score** | `Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(MktCap/TL) + 1.0×(Revenue/TA)` |
| Z-Score thresholds | Safe: `>2.99`, Grey zone: `1.81–2.99`, Distress: `<1.81` |
| **Beneish M-Score** | `M = -4.84 + 0.92×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI` |
| M-Score threshold | Likely manipulator if `> -2.22` |
| **Piotroski F-Score** | 9 binary criteria (0 or 1 each): |
| — Profitability (4) | Net income > 0, ROA > 0, Operating cash flow > 0, Cash flow > net income |
| — Leverage (3) | Lower long-term debt ratio, Higher current ratio, No new share issuance |
| — Operating (2) | Higher gross margin, Higher asset turnover |
| F-Score thresholds | Strong: `8–9`, Moderate: `5–7`, Weak: `0–4` |

Per-ticker in-memory cache, `prefetch_metrics()` for batch pre-loading.

---

## 10. Decision Engine: `decision_engine/engine.py` (~250 lines)

### Combined Score Formula
```
combined_score = sentiment_score × 0.4 + fundamental_score × 0.3 + technical_score × 0.3
```

### Sentiment Scoring
- Raw `sentiment_score` from analyzer
- **1.2× boost** if `confidence > 0.85`
- Clamped to `[-1, 1]`

### Fundamental Scoring (averaged, clamped [-1, 1])
| Factor | Condition | Score |
|---|---|---|
| PEG ratio | < 1 | +0.5 |
| PEG ratio | < 2 | +0.2 |
| PEG ratio | > 3 | -0.3 |
| ROE | > 20% | +0.4 |
| ROE | > 15% | +0.2 |
| ROE | < 10% | -0.2 |
| EPS | > 5 | +0.3 |
| EPS | > 0 | +0.1 |
| EPS | < 0 | -0.3 |
| Intrinsic/Price ratio | > 1.2 | +0.5 |
| Intrinsic/Price ratio | > 1.0 | +0.3 |
| Intrinsic/Price ratio | < 0.8 | -0.5 |
| Intrinsic/Price ratio | < 1.0 | -0.3 |

### Technical Scoring (averaged, clamped [-1, 1])
| Factor | Condition | Score |
|---|---|---|
| RSI | < 30 | +0.5 |
| RSI | < 40 | +0.2 |
| RSI | > 70 | -0.5 |
| RSI | > 60 | -0.2 |
| MACD histogram | > 0 | +0.3 |
| MACD histogram | ≤ 0 | -0.3 |
| Bollinger position | < 0.2 | +0.4 |
| Bollinger position | < 0.4 | +0.2 |
| Bollinger position | > 0.8 | -0.4 |
| Bollinger position | > 0.6 | -0.2 |
| Max drawdown | < -30% | -0.3 |
| Max drawdown | < -20% | -0.1 |

### Decision Thresholds
| Score | Decision |
|---|---|
| ≥ 0.7 | STRONG_BUY |
| ≥ 0.4 | BUY |
| ≤ -0.7 | STRONG_SELL |
| ≤ -0.4 | SELL |
| else | HOLD |

---

## 11. Services

### 11.1 Analysis Service: `services/analysis.py` (424 lines)
`run_analysis_async(tickers)` — main Streamlit analysis flow:
1. **News scraping** (with 3-layer cache: session → scraper cache → DB freshness)
2. **Sentiment analysis** (skips already-analyzed items)
3. **Metrics calculation** (prefetch + in-memory cache)
4. **Signal generation** via `DecisionEngine`
5. Auto-sends WSB email report if SMTP configured
6. `_save_to_database()`: Prepares `signal_data`, `news_data`, `fundamental_data` → `db.save_complete_analysis()`
7. Health score normalization: Z-score → mapped to 0-100; F-score → `/9 × 100`; M-score → binary 100/0

### 11.2 Session Service: `services/session.py` (~70 lines)
`initialize_session_state()`:
- `analysis_complete = False`
- `signals = []`
- `current_page = 'main'`
- `tickers = Config.DEFAULT_TICKERS`
- `progress_messages = []`
- Initializes backtest state, cache

### 11.3 Cache Service: `services/cache.py` (~200 lines)
`SessionCache` (Singleton, thread-safe):
- Namespaces: `news`, `sentiment`, `metrics`, `signals` (keyed by ticker)
- `_CacheEntry` with `expires_at`, auto-pruning on access
- Stats tracking: `hit_count`, `miss_count`, `hit_rate()`
- TTL-aware: entries expire per-namespace TTL

---

## 12. Notifications: `notifications/manager.py` (~250 lines)

### Desktop Notifications
- Library: `plyer` (with console fallback)
- `notify_high_sentiment_news(news_item)`: fires when `confidence > 85%`
- `notify_trading_signal(signal)`: fires for `STRONG_BUY` or `STRONG_SELL`
- `NOTIFICATION_DURATION`: 10 seconds

### Email Notifications
- `send_wsb_email(signals)` (static method)
- SMTP server: `CENTURION_EMAIL_HOST` (default: `smtp-mail.outlook.com`), port `587`, STARTTLS
- Default sender: `CENTURION_EMAIL_USER` env var
- Default recipient: `s.srees@live.com`
- Format: HTML table with signal details

---

## 13. Storage

### 13.1 File Storage: `storage/manager.py` (~90 lines)
`StorageManager`:
- Saves `TradingSignal` list to Excel/CSV
- Append mode with dedup on `(ticker, source, title)`
- Default output: `daily_stock_news.xlsx`
- `Config.OUTPUT_FILE` and `Config.APPEND_MODE` configurable from UI

### 13.2 Object Storage: `storage/minio_service.py` (488 lines)
`MinIOService` (Singleton, lazy-init):

| Config | Env Var | Default |
|---|---|---|
| Endpoint | `MINIO_ENDPOINT` | `localhost:9000` |
| Access Key | `MINIO_ACCESS_KEY` | `minioadmin` |
| Secret Key | `MINIO_SECRET_KEY` | `minioadmin123` |
| Bucket | `MINIO_BUCKET` | `centurion-backtests` |
| Secure | `MINIO_SECURE` | `false` |

- **Object path**: `{run_id}/{ticker}/{strategy_name}/{filename}`
- **Metadata**: `x-amz-meta-run-id`, `x-amz-meta-strategy`, `x-amz-meta-ticker`, `x-amz-meta-chart-title`, `x-amz-meta-chart-type`
- `save_backtest_charts(run_id, charts, strategy_name)`: handles matplotlib (base64→PNG bytes) and plotly (JSON string)
- `get_backtest_images(run_id)`: reads objects + generates presigned URLs (1 hour expiry)
- `delete_run_images(run_id)`: removes all objects under prefix
- `list_runs()`: returns run_id list from object prefixes
- `list_runs_detailed()`: returns run metadata including chart_count, total_size, strategies, tickers, created_at

---

## 14. UI Layer

### 14.1 Styles: `ui/styles.py` (~300 lines)

**Color Constants (`Colors` class)**:
| Name | Hex |
|---|---|
| `PRIMARY` | `#1a1a2e` |
| `SECONDARY` | `#16213e` |
| `ACCENT` | `#0f3460` |
| `ACCENT_BLUE` | `#4361ee` |
| `ACCENT_GREEN` | `#00cc44` |
| `ACCENT_RED` | `#e63946` |
| `TEXT_LIGHT` | `#e0e0e0` |
| `TEXT_DARK` | `#1a1a2e` |
| `BG_DARK` | `#0a0a1a` |
| `BG_CARD` | `#1e1e3f` |
| `GOLD` | `#ffd700` |

**Decision colors**: `STRONG_BUY=#00ff88`, `BUY=#00cc44`, `HOLD=#ffd700`, `SELL=#ff6b6b`, `STRONG_SELL=#ff0000`

**Sentiment colors**: `POSITIVE=#00cc44`, `NEGATIVE=#e63946`, `NEUTRAL=#ffd700`

**Health colors**: `SAFE=#00cc44`, `CAUTION=#ffd700`, `DANGER=#e63946`

- CSS: Background image (`nature_bg.png` with white overlay), typography, buttons, layout, footer, data elements
- `apply_custom_styles()`: Injects all CSS into Streamlit via `st.markdown(unsafe_allow_html=True)`

### 14.2 Components: `ui/components.py` (~300 lines)
- `render_header()`: Logo + title + description + gradient bar
- `render_page_header(title, description)`: Page-level header
- `render_footer()`: Copyright + version info
- `render_navigation_buttons(current_page, back_key_suffix)`: All pages with dynamic visibility
- `render_metrics_cards(signals)`: 5-column decision count display (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- `render_features_section()`: 3-column grid of feature highlights
- `render_how_to_use_section()`: Step-by-step guide expander
- `render_score_interpretations_table()`: Z/M/F score reference table
- `get_decision_emoji(decision)`: Maps decision tags to emoji

### 14.3 Charts: `ui/charts.py` (~300 lines)
All Plotly:
- `render_decision_chart(signals)`: Pie chart of decision distribution
- `render_sentiment_chart(signals)`: Grouped bar chart (positive/negative per ticker)
- `render_score_distribution(signals)`: Scatter plot (decision_score vs ticker)
- `render_fundamental_charts(stock_metrics)`: Three bar charts side-by-side:
  - Z-Score with threshold lines at 2.99 (safe) and 1.81 (distress)
  - M-Score with threshold line at -2.22
  - F-Score with threshold lines at 8 (strong) and 5 (moderate)
- `render_fundamental_summary_metrics(stock_metrics)`: 3 metric boxes: safe/strong/clean counts

### 14.4 Tables: `ui/tables.py` (~300 lines)
- `render_simple_summary_table(signals)`: Aggregated by (stock, source) with yfinance name lookup (cached 24h)
- `render_signals_table(signals)`: Full detailed table with CSV download button
- `render_top_signals(signals)`: Expandable top-5 buy and top-5 sell
- `render_fundamental_table(stock_metrics)`: Z/M/F scores with status icons (✅/⚠️/❌)
- `render_backtest_signals_table(signals)`: Signal table with highlighted BUY/SELL column

---

## 15. UI Pages

### 15.1 Main Page: `ui/pages/main_page.py` (~200 lines)
- `render_main_page()`: header → features → how-to-use → control panel → footer
- **Control Panel**:
  - Ticker selection: 3 modes — Default Tickers, Manual Entry, Upload CSV
  - Output settings: Excel(.xlsx)/CSV(.csv), custom path, append mode toggle
  - Data sources caption: Yahoo Finance, Finviz, Investing.com, TradingView, r/WallStreetBets
  - Run Analysis button → sets `analysis_tickers`, increments `analysis_run_id`, navigates to analysis page

### 15.2 Analysis Page: `ui/pages/analysis_page.py` (~100 lines)
- `render_analysis_page()`:
  - If not complete: runs `asyncio.run(run_analysis_async(tickers))`
  - Results displayed in 4 tabs: Overview (pie + scatter), Detailed Table, Top Signals, Sentiment Charts
  - Navigation: Fundamental Analysis, Backtesting, History buttons

### 15.3 Fundamental Page: `ui/pages/fundamental_page.py` (~100 lines)
- `render_fundamental_page()`:
  - Score interpretations reference table
  - All-stocks overview table (render_fundamental_table)
  - Three charts side-by-side (Z/M/F)
  - Summary metrics

### 15.4 Backtesting Page: `ui/pages/backtesting_page.py` (870 lines)
- **Auto pre-computation**: On first visit after analysis, runs ALL non-crypto strategies against analysis tickers (default: 1 year, $10,000)
- **Cache invalidation**: Only via `analysis_run_id` counter (bumped by "Run Analysis")
- **Layout**: Config panel (left) + Results panel (right)
- **Config Panel**: Category filter → strategy selector → dynamic parameter inputs → ticker/period/capital fields → Run Backtest button
- **Results Panel**: Tab per strategy, each showing: metrics cards (total_return, sharpe, sortino, max_drawdown, trades, final_value), charts (matplotlib/plotly), data tables, signal table
- **MinIO integration**: Charts auto-saved to object storage during pre-computation
- **DB integration**: Results auto-saved to `backtest_results` table with normalized detail tables
- **Run ID format**: `run_{uuid8}_{YYYYMMDD}_{HHMMSS}`

### 15.5 Crypto Page: `ui/pages/crypto_page.py` (580 lines)
- Isolated page for crypto-only strategies
- Default tickers: `ETH, BTC, LTC`
- Data from Binance public API (auto-mapped to USDT pairs)
- Same layout pattern as backtesting page (config + results)
- Requires minimum 2 tickers
- Period options: 1y, 2y, 3y, 5y, Custom
- Separate cache: `st.session_state.crypto_cache`

### 15.6 History Page: `ui/pages/history_page.py` (599 lines)
- Requires database connection
- Date range filter: Last 7/14/30/90 days or Custom Range
- **3 tabs**:
  - **Analysis Runs**: Table of past runs with drill-down (signals + news items for selected run)
  - **Trading Signals**: Filterable by ticker, shows decision distribution chart
  - **Backtest Results**: Table with strategy comparison chart, MinIO chart viewer with presigned URL loading

### 15.7 Pages `__init__.py`: Lazy import map via `__getattr__` — page modules loaded only when first accessed

---

## 16. Strategies Framework

### 16.1 Base Strategy: `strategies/base_strategy.py` (620 lines)

**Data Structures**:
- `StrategyCategory` enum: `MOMENTUM`, `MEAN_REVERSION`, `PATTERN_RECOGNITION`, `STATISTICAL_ARBITRAGE`, `BREAKOUT`, `OPTIONS`, `OTHER`
- `ChartData`: `title`, `data` (base64 or JSON), `chart_type` ("matplotlib"/"plotly"), `description`, `ticker`
- `TableData`: `title`, `data` (list[dict]), `columns`, `description`
- `StrategyResult`: `charts`, `tables`, `metrics`, `signals` (DataFrame), `portfolio` (DataFrame), `success`, `error_message`, `execution_time`, `metadata`
- `RiskParams`: `stop_loss_pct=0.05`, `take_profit_pct=0.10`, `max_position_size=0.25`, `max_drawdown_pct=0.20`, `trailing_stop=False`, `position_sizing="fixed"`

**BaseStrategy (ABC)**:
- Class attrs: `name`, `description`, `category`, `version`, `author`, `requires_sentiment`, `min_data_points=50`
- Abstract: `run(tickers, start_date, end_date, capital, sentiment_data, risk_params, **kwargs) → StrategyResult`
- `validate_inputs()`: checks tickers non-empty, dates valid, capital positive
- `get_risk_params()`: parses dict/RiskParams
- `calculate_metrics(portfolio, signals, capital, risk_free_rate=0.02)` → dict with total_return, final_value, mean_return, std_return, sharpe_ratio, sortino_ratio, max_drawdown, total_trades
- `_calculate_sharpe()`: `excess_returns.mean() / returns.std() × √252`
- `_calculate_sortino()`: `excess_returns.mean() / downside_returns.std() × √252`
- `_calculate_max_drawdown()`: delegates to `strategies.utils.calculate_max_drawdown()`
- `_calculate_portfolio(signals, capital, risk)`: default portfolio calc — `shares = int(capital × max_position_size / max_close_price)`
- `_calculate_portfolio_long_only()`: filters to long positions only (clip lower=0)
- **Sentiment adjustment helpers** (4 methods):
  - `_sentiment_scale_indicator()`: scales indicator by `1 + score × scale_factor` when `|score| > threshold(0.5)`
  - `_sentiment_zero_positions()`: zeros all positions when `score < threshold(-0.7)`
  - `_sentiment_filter_positions()`: removes specific position directions based on sentiment thresholds
  - `_sentiment_filter_signals()`: removes buy/sell signals based on sentiment, recalculates positions (cumsum or cumsum_clip)
- `get_info()` → dict, `get_parameters()` → dict (override in subclass)

### 16.2 Registry: `strategies/registry.py` (~280 lines)
`StrategyRegistry` (class-level singleton):
- `_strategies: dict[str, Type[BaseStrategy]]`
- `register(strategy_class, name)`: normalizes name (lowercase, underscores), warns on overwrite
- `register_decorator`: `@StrategyRegistry.register_decorator` for auto-registration
- `get(name)` → class or None
- `get_or_raise(name)` → class or KeyError
- `list_all()` → `{key: strategy.get_info()}`
- `list_by_category(category)` → filtered
- `unregister(name)`, `clear()`, `count()`
- `is_initialized()` / `set_initialized()`: guards against re-loading

### 16.3 Loader: `strategies/loader.py` (~300 lines)
- `discover_strategies()`: Scans `trading_strategies/` subdirs for `.py` files (skips `_`-prefixed, strips `_bktest` suffix)
- `load_strategy_module(module_path)`: `importlib.util.spec_from_file_location()` dynamic import
- `find_strategy_classes(module)`: finds all `BaseStrategy` subclasses
- `load_all_strategies(register, include_patterns, exclude_patterns)`: main entry — discover → load → register. Skips if already initialized.
- `STRATEGY_MODULE_MAP`: explicit mapping for known strategies (8 entries: macd_oscillator, awesome_oscillator, heikin_ashi, parabolic_sar, rsi_pattern, shooting_star, support_resistance, pairs_trading)

### 16.4 Data Service: `strategies/data_service.py` (417 lines)
`DataService` (Singleton):
- `_cache: dict[str, DataFrame]`, `_cache_expiry: dict[str, datetime]`, cache duration: 1 hour
- `get_ohlcv(ticker, start, end, interval="1d")`: fetches via yfinance, cleans (ffill limit=3, remove zero-volume, numeric coerce)
- `get_multiple_ohlcv()`: iterates tickers
- `add_technical_indicators(df, indicators)`: adds any of `sma_*`, `ema_*`, `rsi`, `macd`, `bollinger`, `atr`
- Indicator impls: RSI (delegates to shared `calculate_rsi`), MACD (EWM 12/26/9), Bollinger (SMA±2σ over 20), ATR (max of 3 true ranges, rolling 14)
- `preload_data(tickers, start, end)`: batch `yf.download()` with `group_by='ticker'`

### 16.5 Utilities: `strategies/utils.py` (427 lines)
- `matplotlib_to_base64(fig, format, dpi, transparent, close_fig)`: saves to buffer → base64 → `data:{mime};base64,{str}`
- `plotly_to_json(fig)`: `fig.to_json()` parsed
- `dataframe_to_table(df, title, max_rows, round_decimals, date_format)`: → `{title, data, columns, description}`
- `create_metrics_summary(metrics, title)`: formatted metrics table with currency/pct formatters
- `calculate_rsi(prices, period=14)`: Wilder's EWM method, NaN filled with 50
- `calculate_max_drawdown(values)`: `(values - cummax) / cummax × 100`, returns negative percentage
- `calculate_trading_statistics(signals, prices)`: total_trades, winning, losing, win_rate, avg_win, avg_loss, profit_factor
- `safe_divide()`, `format_currency()`, `format_percentage()`, `clean_dataframe_for_json()`

---

## 17. Trading Strategies (Implementations)

### 17.1 `trading_strategies/__init__.py` (126 lines)
- **Lazy imports** via `__getattr__` — strategy classes loaded only on access
- `_STRATEGY_META`: 11 entries with `id`, `name`, `description`, `category`, `requires_sentiment`, optional `min_tickers` — returned by `list_strategies()` without importing any classes
- `get_strategy(name)`: returns class via lazy import
- `list_strategies()`: returns metadata list (lightweight)

### 17.2 Momentum Strategies

#### MACD Oscillator (`momentum_trading/macd_oscillator.py`)
- **Signal generation**: Short MA vs Long MA crossover
  - If `use_ema=True`: EWM(ma_short), EWM(ma_long); else SMA
  - `oscillator = ma_short - ma_long`
  - Long when `ma_short ≥ ma_long`, exit otherwise
- **Parameters**: `ma_short=10`, `ma_long=21`, `use_ema=True`
- **Sentiment**: scales oscillator magnitude via `_sentiment_scale_indicator()`
- **Charts**: Price+signals, MA+oscillator, equity curve, drawdown (4 per ticker)

#### Awesome Oscillator (`momentum_trading/awesome_oscillator.py`)
- **Formula**: `AO = SMA(median_price, 5) - SMA(median_price, 34)` where `median_price = (High + Low) / 2`
- **Signal**: Long when `AO > 0`, exit when `AO ≤ 0`
- **Parameters**: `ao_short=5`, `ao_long=34`
- **Sentiment**: scales AO magnitude via `_sentiment_scale_indicator()`
- **Charts**: Price+signals, AO histogram (color: increasing green/lime, decreasing red/salmon), MA overlay, equity curve, drawdown (5 per ticker)

#### Heikin-Ashi (`momentum_trading/heikin_ashi.py`, 555 lines)
- **HA Formulas**:
  - `HA Close = (O + H + L + C) / 4`
  - `HA Open = (prev_HA_Open + prev_HA_Close) / 2`
  - `HA High = max(High, HA_Open, HA_Close)`
  - `HA Low = min(Low, HA_Open, HA_Close)`
- **Signal**: Long when HA candle is green (`HA_Close ≥ HA_Open`)
- **Optional MA filter**: only long when `Close > SMA(ma_period)`
- **Confirmation**: require N consecutive same-color candles before entering
- **Parameters**: `confirmation_candles=1`, `use_ma_filter=False`, `ma_period=20`
- **Sentiment**: zeros all positions when score < -0.7 via `_sentiment_zero_positions()`
- **Charts**: HA candlestick (Rectangle patches), price+positions, regular vs HA price comparison, equity curve (4 per ticker)

#### Parabolic SAR (`momentum_trading/parabolic_sar.py`, 571 lines)
- **Formula**: `SAR = Prior_SAR + AF × (EP – Prior_SAR)`
  - AF starts at `af_start`, increments by `af_increment` each time EP changes, max `af_max`
  - EP = extreme point (highest high in uptrend, lowest low in downtrend)
- **Signal**: Long when `trend == 1` (uptrend), exit when `trend == -1`
- **Parameters**: `af_start=0.02`, `af_increment=0.02`, `af_max=0.2`
- **Sentiment**: zeros positions when score < -0.7
- **Charts**: Price+SAR dots (green uptrend, red downtrend), trend zones, equity curve (3 per ticker)

### 17.3 Pattern Recognition Strategies

#### RSI Pattern (`pattern_recognition/rsi_pattern.py`)
- **Signal**: Buy when `RSI < oversold`, sell when `RSI > overbought`, else neutral
- **RSI calculation**: delegates to shared `calculate_rsi()` (Wilder's EWM)
- **Parameters**: `rsi_period=14`, `oversold_threshold=30`, `overbought_threshold=70`
- **Portfolio**: long-only (`_calculate_portfolio_long_only`)
- **Sentiment**: filters long/short positions via `_sentiment_filter_positions()`
- **Charts**: Price+RSI (2-panel), RSI distribution histogram, equity curve, drawdown (4 per ticker)

#### Shooting Star (`pattern_recognition/shooting_star.py`, 602 lines)
- **Pattern detection** (8 conditions):
  1. `c1`: Open ≥ Close (red candle)
  2. `c2`: Lower wick < `lower_bound × body`
  3. `c3`: Body < `avg_body_20 × body_size`
  4. `c4`: Upper wick ≥ `2 × body`
  5. `c5`: Close ≥ prev_close (uptrend)
  6. `c6`: prev_close ≥ prev_prev_close
  7. `c7`: Next candle's high ≤ current high (confirmation)
  8. `c8`: Next candle's close ≤ current close (confirmation)
- **Signal**: Short on pattern, exit on stop_threshold (5%) or holding_period (7 days)
- **Parameters**: `lower_bound=0.2`, `body_size=0.5`, `stop_threshold=0.05`, `holding_period=7`
- **Sentiment**: filters bearish signals when sentiment > 0.5
- **Charts**: Candlestick with star markers, pattern characteristics (upper wick ratio + body ratio)

#### Support/Resistance (`pattern_recognition/support_resistance.py`, 652 lines)
- **S/R detection**: Local minima/maxima using n1 candles before + n2 candles after pivot
- **Patterns detected**: Engulfing (bullish/bearish), Star (shooting star/hammer)
- **Signal**: Buy when bullish pattern near support, sell when bearish pattern near resistance
- **Proximity check**: price within `level_proximity` (2%) of nearest S/R level
- **Parameters**: `n1=2`, `n2=2`, `back_candles=30`, `level_proximity=0.02`
- **Portfolio**: long-only, positions via `cumsum().clip(0,1)`
- **Sentiment**: filters buy/sell signals at ±0.5 thresholds

#### Bollinger Pattern (`pattern_recognition/bollinger_pattern.py`, 587 lines)
- **Pattern**: "Bottom W" reversal detection with 5 nodes: L (first top), K (first bottom touches lower band), J (middle touches mid band), M (second bottom above lower but below K), I (breakout above upper band)
- **Entry**: Buy on W pattern confirmation (price breaks upper band)
- **Exit**: When bandwidth contracts below `avg_bandwidth × 0.8`
- **Bollinger calc**: `mid = SMA(period)`, `upper/lower = mid ± std × num_std`, `%B = (Close - lower) / (upper - lower)`, `bandwidth = (upper - lower) / mid`
- **Parameters**: `bb_period=20`, `bb_std=2.0`, `pattern_period=75`, `alpha=0.01`
- **Portfolio**: long-only
- **Sentiment**: filters buy signals when score < -0.7

### 17.4 Statistical Arbitrage Strategies

#### Pairs Trading (`statistical_arbitrage/pairs_trading.py`, 600 lines)
- **Requires exactly 2 tickers**
- **Cointegration test**: Engle-Granger via `statsmodels.tsa.stattools.coint()`
- **Rolling cointegration**: tested over `bandwidth` window (default 60 days)
- **Spread**: `Asset1_Close - hedge_ratio × Asset2_Close`
- **Z-Score**: rolling standardization of spread over `bandwidth`
- **Signal rules**:
  - Z > `z_entry` (1.0): Short spread (short Asset1, long Asset2)
  - Z < `-z_entry`: Long spread (long Asset1, short Asset2)
  - |Z| < `z_exit` (0.0): Close position
- **Parameters**: `bandwidth=60`, `z_entry=1.0`, `z_exit=0.0`

#### Mean Reversion Z-Score (`statistical_arbitrage/mean_reversion.py`, 644 lines)
- **Requires ≥ 2 tickers**
- **Statistical tests run**: ADF (stationarity), Hurst Exponent (mean-reverting if < 0.5), Variance Ratio (random walk), Half-Life (speed of reversion)
- **Portfolio construction**: Johansen cointegration eigenvector weights (multi-asset) or OLS hedge ratio (2-asset)
- **Signal rules**:
  - Z-Score < −threshold (2.0): Buy
  - Z-Score > +threshold: Sell
  - Close long when Z > 0, close short when Z < 0
  - Optional per-trade stop-loss
- **Parameters**: `lookback=30` (or auto from half-life), `threshold=2.0`, `stoploss=0.05`
- **Edge utilities** (from `edge_mean_reversion.py`):
  - `perform_adf_test()`: Augmented Dickey-Fuller (stationary if p < 0.05)
  - `perform_hurst_exp_test()`: Hurst exponent (mean-reverting if < 0.5, trending if > 0.5)
  - `perform_variance_ratio_test()`: arch library or manual fallback (non-random if p < 0.05)
  - `half_life()` / `half_life_v2()`: `-ln(2) / β` from OLS of `Δy = β × y_{t-1} + ε`
  - `perform_coint_test()`: Engle-Granger cointegration (cointegrated if p < 0.05)
- **Risk utilities** (from `edge_risk_kit.py`):
  - `drawdown()`: wealth index, peaks, drawdown series
  - `summary_stats()`: annualized return, volatility, Sharpe, skew, kurtosis, Cornish-Fisher VaR, historic VaR, CVaR, max drawdown

### 17.5 Crypto Strategy

#### Crypto Mean Reversion (`crypto/mean_reversion_strategy.py`, 1194 lines)
- **Data source**: Binance public REST API via `binance_data.py`
  - Endpoint: `https://api.binance.com/api/v3/klines`
  - No API key required
  - Pagination: MAX_LIMIT=1000 candles per request, 0.25s pause between requests
  - Per-symbol CSV caching in `trading_strategies/crypto/data/`
  - Incremental cache: only downloads missing date ranges
  - Ticker mapping: base symbol (e.g. `ETH`) auto-mapped to `ETHUSDT`
- **Full pipeline** (preserved from standalone):
  - EDA: pair price plots, return correlations, seaborn pair plots
  - Statistical tests: ADF, Hurst, Variance Ratio, Half-Life
  - Cointegration: Engle-Granger pairwise + Johansen multi-asset
  - 2-asset portfolio: OLS hedge ratio
  - 3-asset portfolio: Johansen eigenvector weights
  - Wealth/drawdown analysis via `edge_risk_kit`
  - Enhanced backtesting via `backtesting.py` library (Strategy, Backtest classes)
  - Parameter optimization: max equity, min drawdown, min volatility, max Sharpe
- **Parameters**: Same Z-Score framework as equity mean reversion
- **Dependencies**: statsmodels, scipy, seaborn, backtesting.py

### 17.6 Legacy/Standalone Backtest Scripts
Each strategy has a corresponding `*_bktest.py` file (standalone scripts, not imported by framework):
- `macd_oscillator_bktest.py`, `awesome_oscillator_bktest.py`, `heikin-ashi_bktest.py`, `parabolic_sar_bktest.py`
- `rsi_pattern_recognize_bktest.py`, `shooting_star_bktest.py`, `support_resistance_bktest.py`, `bollinger_band_bktest.py`
- `pairs_trading_bktest.py`

### 17.7 Shared Backtest Utilities: `trading_strategies/backtest_utils.py` (200 lines)
- `mdd(series)`: Maximum drawdown (iterative, returns negative float)
- `candlestick(df, ax, highlight, ...)`: matplotlib candlestick chart with fill_between, optional highlight overlay
- `portfolio(data, capital0=10000, positions=100)`: basic portfolio value from signals
- `profit(port)`: portfolio asset value plot with long/short markers

---

## 18. Registered Strategy Catalog

| ID | Name | Category | Parameters |
|---|---|---|---|
| `macd` | MACD Oscillator | momentum | `ma_short=10`, `ma_long=21`, `use_ema=True` |
| `awesome_oscillator` | Awesome Oscillator | momentum | `ao_short=5`, `ao_long=34` |
| `heikin_ashi` | Heikin-Ashi | momentum | `confirmation_candles=1`, `use_ma_filter=False`, `ma_period=20` |
| `parabolic_sar` | Parabolic SAR | momentum | `af_start=0.02`, `af_increment=0.02`, `af_max=0.2` |
| `rsi_pattern` | RSI Pattern | pattern_recognition | `rsi_period=14`, `oversold_threshold=30`, `overbought_threshold=70` |
| `shooting_star` | Shooting Star | pattern_recognition | `lower_bound=0.2`, `body_size=0.5`, `stop_threshold=0.05`, `holding_period=7` |
| `support_resistance` | Support/Resistance | pattern_recognition | `n1=2`, `n2=2`, `back_candles=30`, `level_proximity=0.02` |
| `bollinger_pattern` | Bollinger Pattern | pattern_recognition | `bb_period=20`, `bb_std=2.0`, `pattern_period=75`, `alpha=0.01` |
| `pairs_trading` | Pairs Trading | statistical_arbitrage | `bandwidth=60`, `z_entry=1.0`, `z_exit=0.0` (min_tickers: 2) |
| `mean_reversion` | Mean Reversion (Z-Score) | statistical_arbitrage | `lookback=30`, `threshold=2.0`, `stoploss=0.05` (min_tickers: 2) |
| `crypto_mean_reversion` | Crypto Mean Reversion (Z-Score) | crypto | Same Z-Score params (min_tickers: 2) |

---

## 19. Deployment: `deployment/`

- `Dockerfile`: Container build
- `docker-compose.yml`: Multi-service orchestration
- `deploy.ps1` / `deploy.sh`: General deployment scripts
- `deploy-azure.ps1` / `deploy-gcp.ps1`: Cloud-specific
- `DEPLOYMENT.md` / `DOCKER_QUICKSTART.md`: Documentation

---

## 20. File Count Summary

| Directory | Files | Purpose |
|---|---|---|
| Root | 10 | Entry points, config, models, utils |
| `auth/` | 3 | Authentication + credentials |
| `database/` | 4 + 7 repos | ORM, connection, service, repositories |
| `scrapers/` | 8 | News scrapers + cache + aggregator |
| `sentiment/` | 2 | DistilBERT analyzer |
| `metrics/` | 2 | Financial metrics calculator |
| `decision_engine/` | 2 | Scoring engine |
| `services/` | 4 | Analysis, session, cache services |
| `notifications/` | 2 | Desktop + email notifications |
| `storage/` | 3 | File + MinIO object storage |
| `ui/` | 4 + 7 pages | Streamlit UI components + pages |
| `strategies/` | 6 | Strategy framework (base, registry, loader, data, utils) |
| `trading_strategies/` | ~25 | Strategy implementations + backtest scripts + utilities |
| `deployment/` | 7 | Docker + cloud deployment |
