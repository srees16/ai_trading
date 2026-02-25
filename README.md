# Centurion Capital LLC — Enterprise AI Trading Platform

A comprehensive Python-based enterprise trading platform combining multi-source news scraping, AI-powered sentiment analysis, fundamental & technical analysis, strategy backtesting, and persistent data storage. Features an interactive Streamlit web interface, PostgreSQL database persistence, and MinIO object storage for backtest chart images.

## 🚀 Key Features

### Core Analysis Engine
- **Multi-Source News Scraping**: Aggregates news from Yahoo Finance, Finviz, Investing.com, TradingView, and more
- **AI-Powered Sentiment Analysis**: Uses DistilBERT transformer model for accurate sentiment detection
- **Comprehensive Stock Metrics**:
  - Fundamentals: PEG ratio, ROE, EPS, Free Cash Flow, DCF value, Intrinsic value
  - Technicals: RSI, MACD, Fibonacci retracement, Bollinger Bands, Maximum drawdown
  - Scoring: Altman Z-Score, Beneish M-Score, Piotroski F-Score
- **Intelligent Decision Engine**: Weighted combination of sentiment, fundamentals, and technicals to generate trading decisions (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- **Real-Time Alerts**: Desktop popup notifications for high-confidence trading signals
- **Async Architecture**: Concurrent scraping for optimal performance

### Strategy Backtesting
- **11 Built-in Strategies** across three categories:
  - **Momentum**: MACD Oscillator, Parabolic SAR, Heikin-Ashi, Awesome Oscillator
  - **Pattern Recognition**: RSI Pattern, Bollinger Band Pattern, Shooting Star, Support & Resistance
  - **Statistical Arbitrage**: Pairs Trading, Mean Reversion (Stocks), Crypto Mean Reversion (Z-Score)
- **Crypto Mean Reversion Pipeline**: Full statistical arbitrage pipeline for cryptocurrency pairs via the Binance public API — includes EDA, cointegration tests (Engle-Granger & Johansen), 2- and 3-asset portfolio construction, backtesting with parameter optimisation (max equity, min drawdown, min volatility, max Sharpe), and wealth/drawdown analysis
- **Multi-Ticker Support**: Run backtests across multiple tickers simultaneously
- **Per-Ticker Performance Tabs**: Detailed metrics breakdown per ticker
- **Strategy Caching**: Pre-computes and caches results for instant strategy switching
- **Configurable Parameters**: Adjust capital, date range, and strategy-specific settings

### Database Persistence (PostgreSQL)
- **Automatic Logging**: All analysis runs, signals, news items, and backtest results are persisted
- **7 Database Tables**: `analysis_runs`, `stock_signals`, `news_items`, `fundamental_metrics`, `backtest_results`, `trading_signals`, `strategy_parameters`
- **Repository Pattern**: Clean data access layer via SQLAlchemy ORM
- **History Page**: Browse past analyses, signals, and backtests filtered by date range

### Object Storage (MinIO)
- **Chart Image Persistence**: All backtest charts (matplotlib & plotly) saved to S3-compatible MinIO
- **Organised by Run**: Images stored under `<run_id>/<ticker>/<strategy>/<filename>` paths
- **History Integration**: Browse and view stored charts from the History page
- **Metadata Tags**: Each image carries strategy name, chart type, and title as S3 metadata

### Interactive Web Interface
- **Enterprise Branding**: Centurion Capital LLC logo and styling throughout
- **Modular UI Architecture**: Separate page modules for main, analysis, fundamental, backtesting, and history
- **Consistent Navigation**: Uniform button labels across all pages (🏠 Main, 📈 Stock Analysis, 📊 Fundamental Analysis, 🔬 Backtest Strategy, 📋 History)
- **CSV Upload**: Upload custom ticker lists in various formats
- **Visual Analytics**: Interactive charts, pie charts, scatter plots, and bar graphs
- **Authentication**: YAML-based credential management

## 📁 Project Structure

```
centurion_core/
├── app.py                        # Streamlit application router
├── main.py                       # Core orchestration script
├── config.py                     # Configuration settings
├── models.py                     # Data models and interfaces
├── utils.py                      # CSV parsing and utilities
├── setup_database.py             # Database schema initialization
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (DB + MinIO)
├── .streamlit/
│   └── config.toml               # Streamlit server config (port 9090)
├── sample_tickers.csv            # Example ticker list
├── run_streamlit.bat             # Windows quick-launch script
│
├── ui/                           # Modular UI layer
│   ├── components.py             # Reusable components (header, footer, nav)
│   ├── charts.py                 # Chart rendering utilities
│   ├── tables.py                 # Table rendering utilities
│   ├── styles.py                 # CSS styling
│   └── pages/
│       ├── main_page.py          # Dashboard & control panel
│       ├── analysis_page.py      # Stock analysis results display
│       ├── fundamental_page.py   # Fundamental analysis drill-down
│       ├── backtesting_page.py   # Strategy backtesting with MinIO integration
│       └── history_page.py       # Historical results browser (DB + MinIO)
│
├── auth/                         # Authentication
│   ├── authenticator.py          # Login/session management
│   └── credentials.yaml          # User credentials
│
├── database/                     # PostgreSQL persistence layer
│   ├── connection.py             # SQLAlchemy engine & session management
│   ├── models.py                 # ORM models (7 tables)
│   ├── service.py                # High-level database service (singleton)
│   └── repositories/
│       ├── base.py               # Base repository class
│       ├── analysis.py           # AnalysisRepository
│       ├── signals.py            # SignalRepository
│       ├── news.py               # NewsRepository
│       ├── fundamentals.py       # FundamentalsRepository
│       └── backtests.py          # BacktestRepository
│
├── storage/                      # Object storage
│   ├── manager.py                # Excel/CSV file export
│   └── minio_service.py          # MinIO S3 client (singleton)
│
├── scrapers/                     # News scraping modules
│   ├── yahoo_finance.py
│   ├── finviz.py
│   ├── investing.py
│   ├── tradingview.py
│   └── aggregator.py             # Concurrent scraping coordinator
│
├── sentiment/                    # AI sentiment analysis
│   └── analyzer.py               # DistilBERT implementation
│
├── metrics/                      # Financial metrics
│   └── calculator.py             # Fundamentals & technicals
│
├── decision_engine/              # Trading logic
│   └── engine.py                 # Weighted scoring algorithm
│
├── strategies/                   # Strategy framework
│   ├── base_strategy.py          # BaseStrategy, StrategyResult, ChartData
│   ├── registry.py               # Strategy auto-discovery
│   ├── loader.py                 # Dynamic loading
│   ├── data_service.py           # Market data service
│   └── utils.py                  # Chart conversion utilities (base64, plotly JSON)
│
├── trading_strategies/           # Strategy implementations
│   ├── momentum_trading/
│   │   ├── macd_oscillator.py          # + macd_oscillator_bktest.py
│   │   ├── parabolic_sar.py            # + parabolic_sar_bktest.py
│   │   ├── heikin_ashi.py              # + heikin-ashi_bktest.py
│   │   └── awesome_oscillator.py       # + awesome_oscillator_bktest.py
│   ├── pattern_recognition/
│   │   ├── rsi_pattern.py              # + rsi_pattern_recognize_bktest.py
│   │   ├── bollinger_pattern.py
│   │   ├── shooting_star.py            # + shooting_star_bktest.py
│   │   └── support_resistance.py       # + support_resistance_bktest.py
│   └── statistical_arbitrage/
│       ├── pairs_trading.py            # + pairs_trading_bktest.py
│       ├── mean_reversion.py           # Stock mean reversion adapter
│       ├── mean_reversion_strategy.py  # Crypto mean reversion (Binance API)
│       ├── binance_data.py             # Binance public REST API client + CSV cache
│       ├── edge_mean_reversion.py      # ADF, Hurst, Variance Ratio, cointegration tests
│       └── edge_risk_kit.py            # Drawdown & summary statistics helpers
│
├── notifications/                # Desktop alerts
│   └── manager.py
│
├── services/                     # Business logic services
│   ├── analysis.py               # Analysis orchestration
│   └── session.py                # Session state management
│
└── deployment/                   # Deployment configs
    ├── docker-compose.yml        # App + MinIO containers
    ├── Dockerfile
    ├── deploy.ps1 / deploy.sh
    └── DEPLOYMENT.md
```

## 🛠️ Installation

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| PostgreSQL | 14+ | Analysis & backtest persistence |
| Docker | 20+ | MinIO object storage |
| pip | Latest | Package management |

### 1. Clone & Install Dependencies

```powershell
cd centurion_core

# Activate the virtual environment (create it first if needed)
# python -m venv mywinenv
.\mywinenv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **Note**: First run downloads the DistilBERT model (~250 MB).

### 2. Configure Environment Variables

Copy the example and edit:

```powershell
# Create .env in the project root (see .env for all available settings)
```

Or create `.env` in the project root with the following:

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

### 3. Set Up PostgreSQL

#### Option A — Fresh Install (Windows)

1. Download and install [PostgreSQL](https://www.postgresql.org/download/windows/).
2. During installation, set a superuser password (e.g., `superadmin1`).

#### Create the Database & User

Open a terminal and run:

```powershell
# Connect as superuser (adjust path to match your PostgreSQL version)
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres

# Inside psql:
CREATE USER admin WITH PASSWORD 'admin123';
CREATE DATABASE centurion_trading OWNER admin;
GRANT ALL PRIVILEGES ON DATABASE centurion_trading TO admin;
\q
```

#### Verify Connection

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U admin -d centurion_trading -c "SELECT version();"
```

#### Initialize Tables

```powershell
python setup_database.py
```

This creates all 7 tables:

| Table | Purpose |
|---|---|
| `analysis_runs` | Tracks each analysis execution (status, timing, parameters) |
| `stock_signals` | Trading signals with decision, score, and reasoning |
| `news_items` | Scraped news with sentiment labels and confidence |
| `fundamental_metrics` | Per-ticker fundamental data snapshots |
| `backtest_results` | Strategy backtest outcomes (return, Sharpe, drawdown) |
| `trading_signals` | Real-time strategy signals |
| `strategy_parameters` | Saved strategy configurations |

> **TimescaleDB** (optional): If the extension is available, hypertables are created automatically for time-series queries. It is not required — standard PostgreSQL works fine.

### 4. Set Up MinIO (Object Storage for Charts)

MinIO stores chart images generated during backtesting so you can browse them later from the History page.

#### Start MinIO via Docker Compose

```powershell
cd deployment
docker compose up -d minio
```

This pulls the `minio/minio:latest` image and starts a container exposing:

| Port | Purpose |
|---|---|
| `9000` | S3-compatible API |
| `9001` | Web Console |

#### Verify MinIO Is Running

```powershell
docker ps --filter name=centurion-minio
```

#### Access the Web Console

Open **http://localhost:9001** in your browser.

| Field | Value |
|---|---|
| Username | `minioadmin` |
| Password | `minioadmin123` |

The `centurion-backtests` bucket is created automatically on first use.

#### Test From Python

```python
from storage.minio_service import get_minio_service
svc = get_minio_service()
print("Available:", svc.is_available)  # True
```

### 5. Launch the Application

```powershell
streamlit run app.py
```

Opens at **http://localhost:9090** (port configured in `.streamlit/config.toml`).

## 📊 Usage Guide

### Quick Start

1. Launch the app → you land on the **Main** page.
2. Select tickers (default list, manual entry, or CSV upload).
3. Click **🚀 Run Analysis** → results appear on the **Stock Analysis** page.
4. Navigate to **📊 Fundamental Analysis** for detailed fundamental drill-down.
5. Navigate to **🔬 Backtest Strategy** to test any of the 11 strategies.
6. Navigate to **📋 History** to review past runs, signals, and stored charts.

### Strategy Backtesting

1. From the main page (or any sub-page), click **🔬 Backtest Strategy**.
2. Select a strategy from the dropdown.
3. Enter tickers (comma-separated) or use pre-filled tickers from analysis.
4. Adjust period and initial capital.
5. Click **Run Backtest**.
6. Results include:
   - Per-ticker performance tabs with key metrics.
   - Interactive charts (matplotlib & plotly).
   - Results auto-saved to **PostgreSQL** (`backtest_results` table).
   - Charts auto-saved to **MinIO** (`centurion-backtests` bucket).
7. Switch strategies instantly — cached results load without re-computation.

#### Crypto Mean Reversion Strategy

The **Crypto Mean Reversion (Z-Score)** strategy runs a full statistical arbitrage pipeline against live Binance data:

1. Select the strategy from the dropdown.
2. Enter crypto tickers (e.g., `ETH, BTC, LTC`) — these are automatically mapped to USDT pairs.
3. The pipeline runs: EDA → statistical tests → 2-asset portfolio → 3-asset portfolio → backtesting.
4. With **optimisation enabled** (default), four optimisation targets are tested: max equity, min drawdown, min volatility, and max Sharpe ratio.
5. All charts (8 matplotlib PNGs) and backtest HTML plots are saved to MinIO; results are persisted to PostgreSQL.
6. Data is fetched from the Binance public REST API (no API key required) and cached locally as CSV.

### History Page

The **📋 History** page has three tabs:

| Tab | Content |
|---|---|
| **Analysis Runs** | All past analysis executions with drill-down to signals and news |
| **Trading Signals** | Signal history filterable by ticker |
| **Backtest Results** | Past backtest outcomes with strategy comparison chart + stored chart images from MinIO |

Use the date range filter at the top to narrow results.

### Navigation

All pages share consistent navigation buttons:

| Button | Action |
|---|---|
| 🏠 **Main** | Return to the main dashboard |
| 📈 **Stock Analysis** | View analysis results (visible after running analysis) |
| 📊 **Fundamental Analysis** | Open fundamental metrics page |
| 🔬 **Backtest Strategy** | Open the backtesting page |
| 📋 **History** | Open historical results browser |

### CSV Upload Format

Your CSV file can use any of these formats:

```csv
Ticker
AAPL
MSFT
GOOGL
```

```csv
Ticker,Company
AAPL,Apple Inc.
MSFT,Microsoft Corporation
```

Recognised headers: `Ticker`, `Symbol`, `Stock`, `Tickers`, `Symbols`, `Stocks`.

### Decision Algorithm

```
Combined Score = (Sentiment × 0.4) + (Fundamentals × 0.3) + (Technicals × 0.3)

Score ≥  0.7  → STRONG_BUY
Score ≥  0.4  → BUY
Score ≤ -0.7  → STRONG_SELL
Score ≤ -0.4  → SELL
Otherwise     → HOLD
```

## 🗄️ Database Operations Reference

### Service Layer API

```python
from database.service import get_database_service

db = get_database_service()

# Check connectivity
db.is_available  # True / False

# Use a session
with db.session_scope() as session:
    from database.repositories import AnalysisRepository, BacktestRepository
    repo = AnalysisRepository(session)
    runs = repo.get_recent_runs(days=7)
```

### Key Repository Methods

| Repository | Method | Description |
|---|---|---|
| `AnalysisRepository` | `start_run()` / `complete_run()` | Lifecycle management |
| `SignalRepository` | `save_signal()` | Persist a trading signal |
| `NewsRepository` | `save_news_item()` | Persist scraped news |
| `FundamentalsRepository` | `save_metrics()` | Persist fundamental data |
| `BacktestRepository` | `save_result()` / `get_recent_backtests()` | Backtest CRUD |

### Manual Table Reset

```sql
-- Connect to centurion_trading as admin
TRUNCATE analysis_runs, stock_signals, news_items,
         fundamental_metrics, backtest_results,
         trading_signals, strategy_parameters
CASCADE;
```

## 🪣 MinIO Operations Reference

### Python API

```python
from storage.minio_service import get_minio_service

minio = get_minio_service()

# Save an image
path = minio.save_backtest_image(
    run_id="run_b080a824_20260218_163000",
    image_data=png_bytes,
    filename="equity_curve.png",
    strategy_name="MACD Oscillator",
    ticker="AAPL",
    chart_title="Equity Curve",
)

# Retrieve all images for a run
images = minio.get_backtest_images(run_id="run_b080a824_20260218_163000")
for img in images:
    print(img["chart_title"], img["chart_type"], img["size"])

# List all stored runs (basic)
runs = minio.list_runs()

# List runs with full metadata (size, chart count, tickers, strategies)
details = minio.list_runs_detailed()

# Delete all images for a run
deleted = minio.delete_run_images(run_id="run_b080a824_20260218_163000")
```

### Docker Compose Commands

```powershell
# Start MinIO
cd deployment
docker compose up -d minio

# Stop MinIO
docker compose down minio

# View logs
docker logs centurion-minio

# Remove all data (destructive)
docker compose down -v
```

### Storage Path Pattern

```
centurion-backtests/
  └── <run_id>/                          # e.g. run_b080a824_20260218_163000
       └── <TICKER>/                     # e.g. AAPL
            └── <strategy_name>/         # e.g. macd_oscillator
                 ├── chart_0.png         (matplotlib)
                 ├── chart_1.json        (plotly)
                 └── ...
```

## 🐳 Docker Deployment

The `deployment/docker-compose.yml` defines two services:

| Service | Image | Ports | Purpose |
|---|---|---|---|
| `algo-trading` | Custom build | 8501 | Streamlit app |
| `minio` | `minio/minio:latest` | 9000, 9001 | Object storage |

```powershell
# Start everything
cd deployment
docker compose up -d

# Start only MinIO (for local development)
docker compose up -d minio
```

See [DEPLOYMENT.md](deployment/DEPLOYMENT.md) for full cloud deployment instructions (Azure, GCP).

## 🔧 Troubleshooting

### Database Issues

**"no password supplied"**
→ Ensure `.env` has `CENTURION_DB_PASSWORD` set and `python-dotenv` is installed.

**"relation analysis_runs does not exist"**
→ Run `python setup_database.py` to create tables.

**"can't subtract offset-naive and offset-aware datetimes"**
→ Already fixed — all timestamps use `datetime.now(timezone.utc)`.

**TimescaleDB warnings in logs**
→ Harmless — TimescaleDB is optional. The app works with standard PostgreSQL.

### MinIO Issues

**Charts not appearing in MinIO after backtest**
→ Ensure Docker container is running: `docker ps --filter name=centurion-minio`.
→ Check `.env` has `MINIO_ENABLED=true`.
→ Verify connectivity: `python -c "from storage.minio_service import get_minio_service; print(get_minio_service().is_available)"`.

**"minio module not found"**
→ Run `pip install minio`.

**Connection refused on port 9000**
→ Start MinIO: `cd deployment && docker compose up -d minio`.

### Streamlit Issues

**Port already in use**
```powershell
streamlit run app.py --server.port 8502
```

**First run is slow**
→ DistilBERT model downloads (~250 MB) on first launch. Subsequent runs are fast.

### General

**Import errors**
```powershell
pip install -r requirements.txt --upgrade
```

**Virtual environment activation (Windows)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\mywinenv\Scripts\Activate.ps1
```

## 📈 Dependencies

| Category | Packages |
|---|---|
| **Web Framework** | streamlit, plotly, streamlit-aggrid |
| **Data** | pandas, numpy, openpyxl |
| **Financial Data** | yfinance |
| **Crypto Data** | Binance public REST API (no key required) |
| **Scraping** | aiohttp, beautifulsoup4, lxml, requests, selenium |
| **AI/ML** | transformers, torch, scikit-learn |
| **Analysis** | matplotlib, statsmodels, backtesting (0.6+), arch |
| **Database** | sqlalchemy ≥ 2.0, psycopg2-binary ≥ 2.9, python-dotenv ≥ 1.0 |
| **Object Storage** | minio ≥ 7.2 |
| **Auth** | pyyaml ≥ 6.0 |
| **Notifications** | plyer |

## ⚠️ Disclaimer

This software is provided for **educational and informational purposes only**. It does not constitute financial advice, investment recommendations, or professional trading guidance. Stock trading involves substantial risk of loss. Always consult qualified financial advisors before making investment decisions. Use at your own risk.

---

**Ready to get started? Run `streamlit run app.py` and begin analysing! 🚀📈**

*Last Updated: February 2026*
