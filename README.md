# Centurion Capital LLC — Enterprise AI Trading Platform

A Python-based enterprise trading platform built on a **Carver systematic trading framework** (23 forecast sources, FDM combination, volatility-targeted position sizing with AFML meta-labeling). Combines multi-source news scraping, AI-powered sentiment analysis, fundamental & technical analysis, strategy backtesting, HMM regime detection, 7-layer drawdown protection, RL confidence modifier, and automated live Indian market trading via Zerodha Kite Connect. Includes a RAG-powered document intelligence pipeline for research and a **walk-forward signal weight optimizer** (R21a) with Kaggle cloud compute support. Built with a **Next.js 14 frontend** (React, TanStack Query, Tailwind CSS) and a **FastAPI backend**. Backed by PostgreSQL/Neon persistence, MinIO/Cloudflare R2 object storage, Upstash Redis caching, ChromaDB vector search, multi-provider LLM integration (Claude / OpenAI / Ollama), Sentry error tracking, and Better Stack log aggregation. Deployable on HF Spaces + Vercel with GitHub Actions CI/CD.

---

## Quick Start

---

### Step 1 — Clone & install dependencies

```powershell/bash
git clone -b c.core/iterative https://github.com/srees16/centurion_core.git
cd centurion_core
python -m venv myenv 
myenv\Scripts\activate (macOS/Linux: source myenv/bin/activate)
pip install -r requirements.txt
```
> Install Next.js frontend dependencies
```
cd ../centurion_core-fe
npm install
cd ../centurion_core
```
---

### Step 2 — Set environment variables

**Windows PowerShell:**
```powershell
$env:ZERODHA_API_KEY='YOUR_API_KEY'; $env:ZERODHA_API_SECRET='YOUR_API_SECRET'; $env:ZERODHA_USER_ID='YOUR_ZERODHA_ID'; $env:ZERODHA_PASSWORD='YOUR_ZERODHA_PASSWORD'; $env:ZERODHA_TOTP_SECRET='YOUR_BASE32_TOTP_SECRET'; $env:ANTHROPIC_API_KEY='YOUR_ANTHROPIC_API_KEY'; $env:CENTURION_EMAIL_USER='YOUR_GMAIL_ID'; $env:CENTURION_EMAIL_PASS='YOUR_GMAIL_APP_PASSWORD'; $env:CENTURION_DATABASE_URL='postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require'; $env:UPSTASH_REDIS_URL='rediss://default:token@host.upstash.io:6379'; $env:SENTRY_DSN='https://YOUR_KEY@YOUR_ORG.ingest.sentry.io/YOUR_PROJECT_ID'; $env:LOGTAIL_TOKEN='YOUR_LOGTAIL_SOURCE_TOKEN'; $env:CENTURION_RAG_LLM_PROVIDER='claude'; $env:CENTURION_RAG_CLAUDE_MODEL='claude-opus-4-20250514'; $env:CENTURION_RAG_CLAUDE_MAX_TOKENS='1024'; $env:CENTURION_RAG_CLAUDE_TEMPERATURE='0.2'; $env:CENTURION_EMAIL_HOST='smtp.gmail.com'; $env:CENTURION_EMAIL_PORT='587'; $env:API_PORT='9001'; $env:CENTURION_DB_HOST='localhost'; $env:CENTURION_DB_PORT='9003'; $env:CENTURION_DB_NAME='centurion_rag'; $env:CENTURION_DB_USER='postgres'; $env:CENTURION_DB_PASSWORD='superadmin1'; $env:KITE_DB_HOST='localhost'; $env:KITE_DB_PORT='9003'; $env:KITE_DB_NAME='livestocks_ind'; $env:KITE_DB_USER='postgres'; $env:KITE_DB_PASSWORD='superadmin1'; $env:KITE_POOL_MAXSIZE='40'; $env:MINIO_ENDPOINT='localhost:9004'; $env:MINIO_ACCESS_KEY='minioadmin'; $env:MINIO_SECRET_KEY='minioadmin123'; $env:MINIO_SECURE='false'; $env:MINIO_BUCKET='centurion-backtests'; $env:MINIO_ENABLED='true'; $env:MINIO_REGION='auto'; $env:CENTURION_DEFAULT_ADMIN_PASSWORD='admin123'; $env:CENTURION_DEFAULT_ANALYST_PASSWORD='analyst123'; $env:CENTURION_RAG_LLM_URL='http://localhost:11434'; $env:RAG_MODEL='qwen2.5:3b'; $env:CENTURION_RAG_LLM_FIRST_TOKEN_TIMEOUT='300'; $env:CENTURION_RAG_LLM_CHUNK_TIMEOUT='30'; $env:CENTURION_RAG_LLM_NUM_CTX='4096'; $env:CENTURION_RAG_LLM_NUM_PREDICT='500'; $env:CENTURION_RAG_LLM_MAX_TOKENS='500'; $env:CENTURION_RAG_LLM_TEMPERATURE='0.2'; $env:CENTURION_RAG_CHROMA_DIR='./data/chroma_db'; $env:CENTURION_RAG_EMBEDDING_MODEL='BAAI/bge-base-en-v1.5'; $env:CENTURION_RAG_CONTEXT_TOKEN_BUDGET='2000'; $env:CENTURION_RAG_MAX_CONTEXT_CHUNKS='8'; $env:CENTURION_RAG_TOP_K='15'; $env:CENTURION_RAG_SIMILARITY_THRESHOLD='0.70'; $env:CENTURION_RAG_QUERY_BUDGET='300'; $env:CENTURION_RAG_QUERY_REWRITE='false'; $env:CENTURION_RAG_STREAMING='true'; $env:CENTURION_RAG_CACHE_ENABLED='false'; $env:CENTURION_RAG_FAQ_ENABLED='false'; $env:RAG_FAST_MODE='false'; $env:SENTRY_TRACES_SAMPLE_RATE='0.2'; $env:SENTRY_ENVIRONMENT='development'
```

**macOS / Linux:**
```bash
export ZERODHA_API_KEY="YOUR_API_KEY" && export ZERODHA_API_SECRET="YOUR_API_SECRET" && export ZERODHA_USER_ID="YOUR_ZERODHA_ID" && export ZERODHA_PASSWORD="YOUR_ZERODHA_PASSWORD" && export ZERODHA_TOTP_SECRET="YOUR_BASE32_TOTP_SECRET" && export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY" && export CENTURION_EMAIL_USER="YOUR_GMAIL_ID" && export CENTURION_EMAIL_PASS="YOUR_GMAIL_APP_PASSWORD" && export CENTURION_DATABASE_URL="postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require" && export UPSTASH_REDIS_URL="rediss://default:token@host.upstash.io:6379" && export SENTRY_DSN="https://YOUR_KEY@YOUR_ORG.ingest.sentry.io/YOUR_PROJECT_ID" && export LOGTAIL_TOKEN="YOUR_LOGTAIL_SOURCE_TOKEN" && export CENTURION_RAG_LLM_PROVIDER="claude" && export CENTURION_RAG_CLAUDE_MODEL="claude-opus-4-20250514" && export CENTURION_RAG_CLAUDE_MAX_TOKENS="1024" && export CENTURION_RAG_CLAUDE_TEMPERATURE="0.2" && export CENTURION_EMAIL_HOST="smtp.gmail.com" && export CENTURION_EMAIL_PORT="587" && export API_PORT="9001" && export CENTURION_DB_HOST="localhost" && export CENTURION_DB_PORT="9003" && export CENTURION_DB_NAME="centurion_rag" && export CENTURION_DB_USER="postgres" && export CENTURION_DB_PASSWORD="superadmin1" && export KITE_DB_HOST="localhost" && export KITE_DB_PORT="9003" && export KITE_DB_NAME="livestocks_ind" && export KITE_DB_USER="postgres" && export KITE_DB_PASSWORD="superadmin1" && export KITE_POOL_MAXSIZE="40" && export MINIO_ENDPOINT="localhost:9004" && export MINIO_ACCESS_KEY="minioadmin" && export MINIO_SECRET_KEY="minioadmin123" && export MINIO_SECURE="false" && export MINIO_BUCKET="centurion-backtests" && export MINIO_ENABLED="true" && export MINIO_REGION="auto" && export CENTURION_DEFAULT_ADMIN_PASSWORD="admin123" && export CENTURION_DEFAULT_ANALYST_PASSWORD="analyst123" && export CENTURION_RAG_LLM_URL="http://localhost:11434" && export RAG_MODEL="qwen2.5:3b" && export CENTURION_RAG_LLM_FIRST_TOKEN_TIMEOUT="300" && export CENTURION_RAG_LLM_CHUNK_TIMEOUT="30" && export CENTURION_RAG_LLM_NUM_CTX="4096" && export CENTURION_RAG_LLM_NUM_PREDICT="500" && export CENTURION_RAG_LLM_MAX_TOKENS="500" && export CENTURION_RAG_LLM_TEMPERATURE="0.2" && export CENTURION_RAG_CHROMA_DIR="./data/chroma_db" && export CENTURION_RAG_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5" && export CENTURION_RAG_CONTEXT_TOKEN_BUDGET="2000" && export CENTURION_RAG_MAX_CONTEXT_CHUNKS="8" && export CENTURION_RAG_TOP_K="15" && export CENTURION_RAG_SIMILARITY_THRESHOLD="0.70" && export CENTURION_RAG_QUERY_BUDGET="300" && export CENTURION_RAG_QUERY_REWRITE="false" && export CENTURION_RAG_STREAMING="true" && export CENTURION_RAG_CACHE_ENABLED="false" && export CENTURION_RAG_FAQ_ENABLED="false" && export RAG_FAST_MODE="false" && export SENTRY_TRACES_SAMPLE_RATE="0.2" && export SENTRY_ENVIRONMENT="development"
```

> **Tip:** Instead of setting env vars inline, you can copy `.env.example` to `.env` in the project root. The app loads it via `python-dotenv` automatically. See **Section 11, Step 6** or **Section 16.8** for the complete `.env` reference.

---

### Step 3 — Start PostgreSQL + create databases (Docker)

**Windows PowerShell:**
```powershell
docker run -d --name centurion-postgres -p 9003:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=superadmin1 -e POSTGRES_DB=centurion_rag timescale/timescaledb:latest-pg15; Start-Sleep -Seconds 9; docker exec centurion-postgres psql -U postgres -c "CREATE DATABASE centurion_trading;"; docker exec centurion-postgres psql -U postgres -c "CREATE DATABASE livestocks_ind;"
```

**macOS / Linux:**
```bash
docker run -d --name centurion-postgres -p 9003:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=superadmin1 -e POSTGRES_DB=centurion_rag timescale/timescaledb:latest-pg15 && sleep 9 && docker exec centurion-postgres psql -U postgres -c "CREATE DATABASE centurion_trading;" && docker exec centurion-postgres psql -U postgres -c "CREATE DATABASE livestocks_ind;"
```

This creates three databases: `centurion_rag` (analysis results, backtesting, RAG pipeline — default), `centurion_trading` (strategy metrics, Financial ML and Test & Tune chapter outputs), `livestocks_ind` (Kite/Zerodha live trading).

---

### Step 4 — Initialize database tables

Run in the same terminal (env vars from Step 2 are still active):
```powershell
python setup_database.py
```
Expected output: `✓ Database tables created successfully`

This creates all 14 tables in the `centurion_rag` database: `analysis_runs`, `news_items`, `stock_signals`, `fundamental_metrics`, `backtest_results`, `backtest_trades`, `backtest_equity_points`, `backtest_daily_returns`, `strategy_performance_summary`, `user_watchlists`, `alert_configurations`, `raw_scraped_news`, `data_freshness`, and `order_records`.
> The `livestocks_ind` tables (`stocks`, `index_groups`, `index_stocks`, `tick_data`) are auto-created at runtime when the Kite dashboard or webhook service starts.

---

### Step 5 — Start MinIO (Docker) — for backtest, Financial ML & Test-and-Tune charts

```powershell
docker run -d --name centurion-minio -p 9004:9000 -p 9002:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin123 minio/minio:latest server /data --console-address ":9001"
```

The `centurion-backtests` bucket is auto-created on first use. It stores backtest strategy charts, Financial ML figures.

---

### Step 6 (Optional) — Install Ollama for local RAG

**Windows:**
```
winget install Ollama.Ollama (or download from https://ollama.ai/download)
```
**macOS:**
```
curl -fsSL https://ollama.com/install.sh | sh
```
Then pull the model:
```
ollama pull qwen2.5:3b
```
---

### Step 7 — Terminal 1: Launch FastAPI backend

Run in the **same terminal** (env vars from Step 2 must still be active):
```
python run_api.py --reload
```
Backend API at: **http://localhost:9001** — API docs at **http://localhost:9001/docs**

---

### Step 8 — Terminal 2: Launch Next.js frontend

Open a new terminal:
> cd centurion_core-fe
```
npm run dev
```
Opens at: **http://localhost:3000** — login with `admin` / `admin123`

MinIO console at: **http://localhost:9002/login** — login with `minioadmin` / `minioadmin123`

---

Jump to **Section 15: Troubleshooting** or **Section 12: Installation** for detailed setup.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Carver Systematic Trading Framework](#2-carver-systematic-trading-framework)
3. [Core Analysis Engine](#3-core-analysis-engine)
4. [Strategy Backtesting](#4-strategy-backtesting)
5. [Live Trading — Zerodha Kite Connect](#5-live-trading--zerodha-kite-connect)
6. [RAG Document Intelligence](#6-rag-document-intelligence)
7. [Database Layer](#7-database-layer)
8. [Object Storage (MinIO / Cloudflare R2)](#8-object-storage-minio--cloudflare-r2)
9. [Interactive Web Interface](#9-interactive-web-interface)
10. [Financial ML & Test-and-Tune](#10-financial-ml--test-and-tune)
11. [Project Structure](#11-project-structure)
12. [Installation (Detailed)](#12-installation)
13. [Usage Guide](#13-usage-guide)
14. [API Reference](#14-api-reference)
15. [Troubleshooting](#15-troubleshooting)
16. [Dependencies](#16-dependencies)
17. [Cloud Infrastructure & Observability](#17-cloud-infrastructure--observability)

---

## Changelog

### April 2026

**R21a Walk-Forward Optimizer** — New `optimizer/optimize_weights_r21a.py` uses scipy differential evolution (population=60, maxiter=150) to find optimal signal weights for the 11-source Carver backtester. Features: checkpoint every 5 generations with resume support, Kaggle cloud compute via `cloud/run_cloud_kaggle.py` (free tier: 4 CPU, 29 GB RAM, 12-hr sessions). Current best at Gen 50: Sharpe 1.780, CAGR 56.7%, MaxDD 22.4%.

**Cloud Runners** — New `cloud/` folder with Kaggle, Colab, and Modal runners for offloading heavy compute. Dataset management via Kaggle CLI with `--dir-mode zip` for folder uploads.

**CLI Runners** — New `runners/` folder consolidating entry points: `run_backtest.py`, `run_r21a.py`, `run_extract_forecasts.py`, `run_r21a_pipeline.py`, `run_contra_v4.py`.

**Paper Trading Frontend** — Trade Monitor page (`/ind-stocks/trade-monitor`) with Paper Validation tab showing cumulative performance metrics (Sharpe, Sortino, Calmar, CAGR, Max DD, Win Rate), equity curve, daily P&L, weekly checkpoints, signal audit, and pass/fail verdict. Daily Detail tab for per-day drill-down. Automated via GitHub Actions.

**Streamlit Removal** — Removed legacy Streamlit UI (`app.py`, `ui/` folder, `auth/authenticator.py`). Next.js 14 is now the sole frontend.

**Signal Quality Evaluator** — New `services/signal_quality_evaluator.py` provides regime-conditioned signal analysis with CAGR estimation, stress testing, and auto-generated documentation.

**Aronson EBTA Validation** — New `services/aronson_validator.py` implements statistical validation: detrended returns, signal t-statistics, Benjamini-Hochberg FDR, White's Reality Check, and Deflated Sharpe Ratio.

**Full Pipeline Backtest Fixes** — Fixed 3 incorrect import names in `services/full_pipeline_backtest.py` (`compute_fii_forecast`, `generate_event_forecasts`, `compute_sentiment_batch`) and enabled 6 previously omitted offline sources (penfold_trend, ehlers_dsp, intermarket, acceleration, carver_value, skew_signal) bringing active backtest sources from 11 to 17.

**New Services** — `services/hrp_allocator.py` (Hierarchical Risk Parity), `services/deflated_sharpe.py` (Bailey-López de Prado DSR).

**Cleanup** — Removed session-generated test scaffolds (3 files, 58 tests), dead code (`_fetch_sentry.py`, `services/portfolio_correlation.py`, unused `CIRCUIT_BREAKER_TIERS` config), and 9 audit-session docs.

### March 2026

#### Forecast Engine: 11 → 23 Sources

Expanded from 11 to 23 independent forecast signals. New additions:

| # | Source | Weight | Description |
|---|--------|--------|-------------|
| 12 | EWMAC (8, 32) | 7% | Fastest swing crossover |
| 13 | Breakout | 0% | Channel breakout (disabled — poor NSE backtest) |
| 14 | Cross Momentum | 4% | Cross-sectional relative momentum |
| 15 | Pairs Arb | 2% | Statistical pairs mean-reversion |
| 16 | Event Driven | 4% | Corporate event catalyst scoring |
| 17 | Penfold Trend | 7% | Bryce Gilmore / Penfold adaptive trend |
| 18 | Ehlers DSP | 8% | John Ehlers digital signal processing (MESA, SNR) |
| 19 | Intermarket | 7% | Ruggiero-style cross-asset signals |
| 20 | Acceleration | 5% | Rate-of-change of EWMAC(16,64) |
| 21 | Carver Value | 2% | Long-term valuation mean-reversion |
| 22 | Skew Signal | 3% | Options skew premium extraction |
| 23 | Sentiment | 2% | News/social sentiment NLP composite |

### Meta-Labeling (AFML Ch.3)

New secondary classifier predicts whether primary forecasts will be profitable:
- **Model**: RandomForest (300 trees, max_depth=5), walk-forward 252d train / 63d test
- **Features**: 20 (expanded from 12) — added FII flow proxy, OI change, VIX term structure, breadth momentum, return_60d, vol-of-vol, skew, volume trend
- **Gate**: Blocks signals with meta-probability < 0.50; scales remaining by confidence
- **Scheduler**: Job 18 retrains semi-monthly at 02:00 IST
- **File**: `services/meta_labeling.py`

### Configuration Changes

| Parameter | Old | New | Rationale |
|-----------|-----|-----|----------|
| `VOL_TARGET` | 20% | 75% | Matched to 7× F&O leverage capacity |
| `MAX_LEVERAGE` | 1× | 7× | NRML F&O margin-based (Bull cap) |
| `IDM` | 1.0 | 2.0 | Instrument Diversification Multiplier for 12 positions |
| `MAX_POSITIONS` | 6 | 12 | Broader diversification |
| `OPTIONS_ENABLED` | False | True | CSP + covered calls active |
| `RL_ENABLED` | False | True | RL agent as ±15% confidence modifier |
| `PEAD weight` | 4% | 6% | Boosted post-earnings drift signal |
| `ML_MIN_PROB` | 0.55 | 0.50 | Reduced over-aggressive filtering |

### Regime-Adaptive Weights

5-regime conditional profiles in `services/regime_strategy_mix.py`:
- **Bear/Range/Crisis**: Counter-cyclical signals boosted (PEAD 13-15%, mean-reversion 13-15%, sentiment 3-4%)
- **Bull**: Trend signals dominate (Penfold 12%, Ehlers 12%, momentum 10%)
- Trend-following signals (EWMAC, momentum) reduced in non-trending regimes
- All 5 profiles verified sum = 1.000

### Drawdown Protection: 6-Tier → 7-Layer

| Drawdown | Action | Leverage Cap |
|----------|--------|--------------|
| 0–15% | Full size | Bull: 7×, Range: 5× |
| 15–25% | Quadratic scale-down | Bear: 2× |
| 25–30% | Minimal exposure | Crisis: 0.5× |
| >30% | **Full halt** | 0× |

### Pipeline Flow

```
Forecast (23 sources) → RL Modifier (±15%) → Meta-Label Gate (prob>0.50)
  → Cost Filter → Vol-Targeted Sizing → Regime Leverage Cap → DD Scale → Execute
```

### Backtest Validation (Apr 2021 – Mar 2026, 5-Year Walk-Forward)

Full pipeline backtest on 14 NIFTY50 stocks, 17 offline-capable forecast sources, 75% vol target, 7× leverage:

| Metric | Value |
|--------|-------|
| **Annual Return (CAGR)** | **+45.9%** |
| **Total Return** | +330.7% (500K → 2.15M) |
| **Sharpe Ratio** | 1.064 |
| **Sortino Ratio** | 1.456 |
| **Calmar Ratio** | 1.004 |
| **Max Drawdown** | 45.7% |
| **Total Trades** | 2,185 |
| **Avg Positions** | 8.2 |

**Regime-Conditioned Performance:**

| Regime | CAGR | Sharpe | Max DD |
|--------|------|--------|--------|
| BULL (41%) | +61.0% | 1.31 | 33.4% |
| SIDEWAYS (40%) | +44.0% | 1.05 | 29.7% |
| BEAR (19%) | +22.2% | 0.65 | 32.5% |

**CAGR Estimates (with overfitting adjustments):**
- Ideal: +45.9% | Realistic: +45.6% | Conservative: +8.9%
- 90% Bootstrap CI: [+4.8%, +122.0%]
- Aronson EBTA: 9/16 signals with t ≥ 2.0; Trimmed Sharpe 1.589

### Signal Quality Evaluator (April 2026)

New `services/signal_quality_evaluator.py` provides regime-conditioned signal analysis:
- **Regime segmentation**: HMM + ADX + trend slope → BULL / BEAR / SIDEWAYS classification
- **Signal metrics**: Per-source hit rate, Sharpe, profit factor, expectancy by regime
- **Backtest**: Delegates to production `full_pipeline_backtest.py` (17 sources, vol-targeted)
- **CAGR estimation**: Ideal / realistic / conservative with block bootstrap CI
- **Stress testing**: High-vol, extreme bear, low-confidence, first-year, last-year scenarios
- **Auto-generated docs**: `docs/signal_quality_by_regime.md`, `regime_performance.md`, `cagr_estimation.md`, `signal_insights.md`

### Aronson EBTA Statistical Validation (April 2026)

New `services/aronson_validator.py` implements Evidence-Based Technical Analysis (Aronson, 2007):
- **Detrended returns**: Remove market beta before evaluating signal performance
- **Signal t-statistics**: Per-signal statistical significance testing
- **Benjamini-Hochberg**: FDR-controlled p-value adjustment for multiple comparisons
- **White's Reality Check**: Bootstrap data-mining bias estimation
- **Deflated Sharpe Ratio**: Bailey-López de Prado DSR (corrects for trial multiplicity)
- **Walk-forward degradation**: OOS/IS ratio with automatic overfit detection

---

## 1. Architecture Overview

The application follows a modular, deferred-import architecture with a **Carver-inspired systematic trading pipeline** at its core:

```
Next.js 14 Frontend (primary — port 3000)
  ├── React 18, TypeScript, Tailwind CSS, TanStack Query v5
  ├── JWT auth (Zustand store) + next-themes dark/light mode
  ├── API proxy rewrites → FastAPI backend (port 9001)
  └── Pages: US/IND Stocks, Financial ML, Test & Tune, RAG Engine, Settings

FastAPI Backend (port 9001)
  ├── /api/v1/* — 50+ REST + SSE endpoints
  ├── Auth: itsdangerous signed tokens (8h TTL)
  └── Delegates to: scrapers, sentiment, metrics, forecast_combiner, rag_pipeline
```

### Signal-to-Execution Pipeline

```
┌─ STAGE 1: UNIVERSE ──────────────────────────────────────────────────┐
│ NSE Universe Download (NIFTY50 + NEXT50) via Kite API / yfinance     │
└──────────────────────────────────────────────────────────────────────┘
         ↓
┌─ STAGE 2: SCREENING ─────────────────────────────────────────────────┐
│ 3-stage NSE screener: liquidity → volatility → technical composite   │
│ (RSI + MACD + Bollinger + volume surge + price range)                │
└──────────────────────────────────────────────────────────────────────┘
         ↓
┌─ STAGE 3: SIGNAL FILTER ─────────────────────────────────────────────┐
│ IntegratedScorer 2-layer evaluation → accept BUY / STRONG_BUY only   │
│ Enrich with live LTP, filter bid-ask < 0.2%, order < 5% ADV         │
└──────────────────────────────────────────────────────────────────────┘
         ↓
┌─ STAGE 4: CARVER FORECAST ENGINE ────────────────────────────────────┐
│ 23 forecast sources → FDM combination → single forecast (±20)        │
│ RL confidence modifier (±15%) → Meta-label gate (prob>0.50)          │
│ Cost speed limit + strategy decay filter + HMM regime blend          │
│ Volatility-targeted position sizing (75% annual vol target, 7× lev)  │
│ 7-layer drawdown protection + regime-adaptive leverage caps          │
└──────────────────────────────────────────────────────────────────────┘
         ↓
┌─ STAGE 5: EXECUTION ─────────────────────────────────────────────────┐
│ Kite order placement (BUY CNC) + SL-M + TP limit orders             │
│ TradeMonitor lifecycle: trailing stop, crash recovery, corp actions   │
└──────────────────────────────────────────────────────────────────────┘
```

### IntegratedScorer — 2-Layer Evaluation Pipeline

```
IntegratedScorer (services/integrated_scorer.py)
  Layer 1 — Core Analysis (45%)
    Fundamental score (P/E, EPS growth, debt ratios)
    Technical score (RSI, MACD, moving averages)
    Macro score (GDP, inflation, FII flows)
    Delivery conviction multiplier (NSE ≥ 60%)
    Earnings momentum boost (post-earnings drift)

  Layer 2 — Strategy Consensus + Robustness (55%)
    10+ strategies run in parallel (ThreadPoolExecutor)
    Sharpe-weighted consensus vote, horizon-aware multipliers
    Walk-forward validation (4 rolling folds, 252d train / 63d test)
    Degradation ratio check (OOS/IS Sharpe; reject if < 0.5)
    CSCV bootstrap & MC permutation tests
  ──────────────────────────────────────────────────────────────────────
  Output → StockVerdict (STRONG_BUY ≥ 0.55 | BUY ≥ 0.30 | HOLD | SELL ≤ -0.30 | STRONG_SELL ≤ -0.55)
```

### Background Scheduler

```
scheduler.py (APScheduler)
  ├── Pre-market (09:20 IST)        DB pre-warm → full pipeline: scrape → analyse → screen → auto-execute
  ├── Intraday (10:30, 12:30, 14:30) Score refresh + re-screen for new signals
  ├── Walk-forward audit (Sat 06:00)  Weekly walk-forward validation of all strategies
  ├── Reconciliation (Sat 07:00)      3-leg: backtest ↔ paper ↔ live parity check
  ├── Nightly backup (23:00)          SQLite databases → R2/MinIO object storage
  └── Auto-auth                       Kite TOTP auto-fill via pyotp (zero-touch)
```

### Services Layer

```
services/
  ├── ForecastCombiner     23 forecast sources → FDM combination (±20 cap, ~1.35 multiplier)
  ├── VolatilityTarget     75% annual vol target, 7× leverage, IDM=2.0, rolling capital rebalancing
  ├── MetaLabeling         AFML Ch.3 triple-barrier meta-labeling (20 features, walk-forward RF)
  ├── RegimeStrategyMix    5-regime conditional weight profiles (bull/bear/range/high-vol/crisis)
  ├── RegimeHMM            3-state Gaussian HMM (Bull/Bear/Sideways); log-space forward-backward
  ├── RegimeDetector       5-state fallback regime (VIX, NIFTY returns, ADX); adaptive thresholds
  ├── StrategyDecay        63-day rolling Sharpe monitor; auto-scale or blacklist degraded strategies
  ├── OptionsOverlay       Covered calls + cash-secured puts (IV rank, delta-based strike selection)
  ├── WalkForward          Rolling 1Y train / 1Q test; degradation ratio (OOS/IS < 0.5 = overfit)
  ├── CorporateActions     NSE SPLIT/BONUS/DIVIDEND/RIGHTS; adjusts OHLCV, positions, SL/TP
  ├── DeliveryVolume       NSE delivery % analysis (≥ 60% = institutional conviction, +12 pts)
  ├── EarningsMomentum     Post-earnings drift (5-day momentum); score boost +0.12 → +0.02 decay
  ├── SectorRotation       NIFTY sector 1-month momentum; top 3 bonus, bottom 3 penalty
  ├── SurvivorshipFilter   Detects delisted/suspended/dead stocks (4 methods, 1-hour cache)
  ├── FundamentalFreshness Intra-quarter freshness via bulk deals, promoter pledges, MF holdings
  ├── IntegratedScorer     2-layer eval: Core 45% → Strategy + Robustness 55%
  ├── SignalQualityEval    Regime-conditioned signal analysis, CAGR estimation, stress testing
  ├── AronsonValidator     EBTA statistical validation (detrend, t-stats, BH, White's RC, DSR)
  ├── HRPAllocator         Hierarchical Risk Parity (López de Prado) portfolio allocation
  └── DeflatedSharpe       Bailey-López de Prado deflated Sharpe ratio (trial multiplicity correction)
```

### Infrastructure Layer

```
infrastructure/
  ├── EventBus         In-process pub/sub with JSONL replay log; topic routing, correlation IDs
  ├── FaultIsolation   SupervisedWorker (crash recovery) + CircuitBreaker (cascading failure prevention)
  ├── LatencyTracker   Microsecond SLA tracking (100-200ms target); p50/p95/p99 sliding window
  ├── ModelRegistry    Lazy-loading ML model registry (FinBERT, transformers); thread-safe singleton
  ├── ReplayEngine     Deterministic event replay from JSONL logs for live → backtest reproducibility
  ├── TimeSeriesStore  TimescaleDB (live) / in-memory ring buffer (replay); unified tick/OHLCV API
  ├── ExecutionContext Dual-mode context (live / paper / backtest); same code path for all modes
  ├── AnalysisPipeline 8-stage institutional pipeline: Raw → Clean → Feature → Alpha → Combine → Optimize → Execute → Post-Trade
  ├── LoggingConfig    JSON structured logging with correlation IDs; Better Stack (Logtail) cloud shipping
  ├── CacheService     Dual-layer L1 (in-memory) + L2 (Upstash Redis); lazy URL resolution; /health reporting
  ├── Sentry           Error tracking + performance tracing (FastAPI/Starlette/Logging integrations)
  └── BackupService    Nightly SQLite backup to R2/MinIO (scheduler_cache, trade_monitor, chroma)
```

### Architectural Layers

```
layers/
  ├── AlphaResearch    Coordinates all alpha sources; emits alpha.signal events; customisable weights
  ├── ExecutionEngine  Routes orders: Kite (IND) / DriveWealth (US) / PaperBroker based on context
  ├── MarketData       Unified data feed (OHLCV, ticks, fundamentals, news); .NS/.BO → Kite + yfinance
  ├── Monitoring       Health checks, latency dashboards, audit trail via EventBus subscription
  ├── Portfolio        Allocation tracking, P&L computation, rebalancing logic with events
  └── RiskEngine       Pre-trade + post-trade risk checks (max position, drawdown circuit breaker)
```

### Dual Strategy System

| System | Location | Base Class | Data Source | Output |
|--------|----------|------------|-------------|--------|
| **Framework** | `strategies/` + `trading_strategies/` | `BaseStrategy` (ABC) | `DataService` (yfinance, cached) | `StrategyResult` (charts, tables, metrics, signals) |
| **Standalone** | `*_bktest.py` files | None | Direct yfinance or CSV | matplotlib plots, printed stats |

---

## 2. Carver Systematic Trading Framework

The core alpha engine implements a **Robert Carver–inspired systematic trading pipeline** (*Systematic Trading*, *Leveraged Trading*). All position sizing, forecast generation, and risk management run through this framework for Indian equities via Zerodha Kite Connect.

### 23 Forecast Sources

Every screened stock generates a combined forecast from 23 independent signal sources, each capped at ±20:

| # | Source | Weight | Description |
|---|--------|--------|-------------|
| 1 | **EWMAC (8, 32)** | 7% | Fastest swing crossover |
| 2 | **EWMAC (16, 64)** | 7% | Fast swing — 16-day EMA minus 64-day EMA |
| 3 | **EWMAC (32, 128)** | 6% | Medium-term trend confirmation |
| 4 | **EWMAC (64, 256)** | 6% | Positional trend — 64-day EMA minus 256-day EMA |
| 5 | **Carry Rule** | 1% | Dividend yield minus funding cost spread |
| 6 | **Momentum (20d)** | 8% | 20-day price momentum |
| 7 | **Mean Reversion** | 3% | Bollinger / Keltner oversold–overbought |
| 8 | **Screener Score** | 4% | Technical + fundamental composite overlay |
| 9 | **PEAD** | 6% | Post-earnings announcement drift |
| 10 | **FII Flow Signal** | 3% | FII inflow rate convexity (z-score adaptive) |
| 11 | **Options OI Signal** | 2% | Open interest distribution skew |
| 12 | **Decision Engine** | 3% | Multi-layer integrated verdict |
| 13 | **Breakout** | 0% | Channel breakout (disabled — poor NSE fit) |
| 14 | **Cross Momentum** | 4% | Cross-sectional relative momentum |
| 15 | **Pairs Arb** | 2% | Statistical pairs mean-reversion |
| 16 | **Event Driven** | 4% | Corporate event catalyst scoring |
| 17 | **Penfold Trend** | 7% | Bryce Gilmore / Penfold adaptive trend |
| 18 | **Ehlers DSP** | 8% | John Ehlers MESA adaptive cycle + SNR filter |
| 19 | **Intermarket** | 7% | Ruggiero-style cross-asset signals |
| 20 | **Acceleration** | 5% | Rate-of-change of EWMAC(16,64) |
| 21 | **Carver Value** | 2% | Long-term valuation mean-reversion |
| 22 | **Skew Signal** | 3% | Options skew premium extraction |
| 23 | **Sentiment** | 2% | News/social sentiment NLP composite |

**Combination:**
- Forecasts are combined via a **Forecast Diversification Multiplier (FDM)** ~1.35 (max 2.0), computed from the inter-forecast correlation matrix
- Target average forecast magnitude: ~10
- Trend vs. carry correlation: 0.25 (decorrelated)
- Mean reversion vs. EWMAC: -0.15 to -0.30 (negatively correlated — diversification benefit)

### Volatility-Targeted Position Sizing

All positions are sized to a **75% annual volatility target** with IDM=2.0 (IND) or 20% / IDM=1.5 (US):

$$\text{Qty} = \frac{\text{Capital} \times \text{VolTarget} \times \text{Weight} \times \text{IDM} \times \frac{|\text{Forecast}|}{10}}{\text{InstrumentVol} \times \text{Price}}$$

- **Capital**: ₹500K (IND), $10K (US) | **Max Positions**: 12 (IND), 15 (US)
- **Leverage Caps**: Bull 7×, Range 5×, Bear 2×, Crisis 0.5×
- Recalculated daily with rolling capital

### 7-Layer Drawdown Protection

| Drawdown | Risk Level | Position Scale | Leverage Cap |
|----------|-----------|----------------|--------------|
| 0–10% | HEALTHY | 100% | Full (regime cap) |
| 10–15% | WARNING | Quadratic scale | Reduced |
| 15–25% | CRITICAL | ~50–75% | Bear: 2× |
| 25–30% | EXTREME | ~25% | Crisis: 0.5× |
| >30% | **HALTED** | **0%** | **0× — all orders blocked** |

### Risk Management Stack

| Control | Value | Purpose |
|---------|-------|---------|
| Max risk per trade | 1–3% of capital | Kelly-criterion bounds |
| Max open trades | 12 positions (IND) | Concentration limit |
| Max per sector | 30% of capital | Diversification |
| Max trades/sector | 3 open positions | Sector correlation cap |
| Min R:R ratio | 2.5:1 | Risk/reward threshold |
| VIX caution (>20) | Scale to 60% | Regime overlay |
| VIX panic (>25) | Block BUY orders | Circuit breaker |
| ADX choppy (<20) | Scale to 50% | Trend-quality filter |
| Meta-label gate | Block if prob < 0.50 | False-signal filter |
| Portfolio correlation | Reject if > 0.60 | Crowded-trade protection |

### HMM Regime Detection

A **3-state Gaussian Hidden Markov Model** provides regime-conditioned parameter adaptation:

| State | Name | Characteristics | Position Scale |
|-------|------|----------------|----------------|
| S₀ | **BULL** | μ > 0, σ low, breadth positive | 1.0× |
| S₁ | **BEAR** | μ < 0, σ high, breadth weak | 0.5× |
| S₂ | **SIDEWAYS** | μ ≈ 0, σ medium, range-bound | 0.7× |

- **Features (4D):** NIFTY daily log-returns, India VIX (normalised), market breadth (A/D ratio), delivery volume %
- **Training:** EM (Baum-Welch) on 5 years of data; log-space forward-backward (prevents underflow for T > 200)
- **Prediction:** 5-day ahead state probabilities via transition matrix
- Falls back to 5-state rule-based `RegimeDetector` when HMM confidence < 0.6

### Strategy Decay Monitor

Rolling 63-day Sharpe is compared against walk-forward historical:

| Status | Decay Ratio | Allocation | Action |
|--------|------------|------------|--------|
| HEALTHY | > 0.50× historical | 100% | Full weight |
| DEGRADED | 0.25–0.50× | 50% | Halve, monitor |
| DEAD | < 0.25× | 0% | Zero, re-calibrate |
| INVERTED | Sharpe < 0 | 0% | Blacklist |

### Options Overlay (Enabled)

Two systematic strategies for premium harvesting (`OPTIONS_ENABLED=True`):

| Strategy | Condition | Strike | Expiry | Roll Trigger |
|----------|-----------|--------|--------|-------------|
| **Covered Call** | IV rank > 50, forecast weakening | 30-delta OTM | 30–45 DTE | 50% max profit or 14 DTE |
| **Cash-Secured Put** | IV rank > 40, positive forecast | 25-delta OTM | 30–45 DTE | 50% max profit or 14 DTE |

### Walk-Forward Validation

- **Training:** 252 days (1Y in-sample) → **Test:** 63 days (1Q out-of-sample)
- **Folds:** 4 rolling quarterly windows
- **Degradation ratio:** OOS Sharpe / IS Sharpe — reject if < 0.5 (overfit)
- **Transaction cost:** 0.40% round-trip (STT + exchange + GST + slippage) deducted per OOS trade
- **Cost speed limit:** Rejects forecasts where turnover cost exceeds vol-target benefit

---

## 3. Core Analysis Engine

### News Scraping

Five concurrent US scrapers + eleven Indian scrapers with 3-layer caching (session → scraper cache → DB freshness):

**US Market:**

| Source | Method | Limit |
|--------|--------|-------|
| Yahoo Finance | `yfinance` library (`Ticker.news`) | 10/ticker |
| Finviz | HTTP scraping (optional Selenium for Elite) | 10/ticker |
| Investing.com | HTTP with custom headers | 10/ticker |
| TradingView | JSON API (`news-headlines.tradingview.com`) | 10/ticker |
| r/WallStreetBets | Reddit public JSON API (8 flairs) | 50/flair |

**Indian Market:**

| Source | Method | Limit |
|--------|--------|-------|
| MoneyControl | HTTP scraping | 10/ticker |
| Economic Times | HTTP scraping | 10/ticker |
| LiveMint | HTTP scraping | 10/ticker |
| Business Standard | HTTP scraping | 10/ticker |
| Hindu BusinessLine | HTTP scraping | 10/ticker |
| Zerodha Pulse | HTTP scraping | 10/ticker |
| NDTV Profit | HTTP scraping | 10/ticker |
| Google News India | HTTP scraping | 10/ticker |
| FII/DII Flows | NSE data API | Daily |
| Circuit Detector | NSE bhavcopy | Daily |
| Market Breadth | NSE advance/decline | Daily |

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

The decision engine provides the **base verdict** that feeds into the Carver forecast pipeline (see [Section 2](#2-carver-systematic-trading-framework)). It combines three signal components into a preliminary score:

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

This verdict is one of the 11 forecast sources (weight: 5%) in the Carver forecast combiner. The combined forecast (±20) then drives volatility-targeted position sizing and all downstream risk filters.

---

## 4. Strategy Backtesting

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

## 5. Live Trading — Zerodha Kite Connect

Real-time Indian equity monitoring, order management, option chain analysis, and **Carver-pipeline automated trading**.

### Components

| Module | Purpose |
|--------|---------|
| `zerodha_live.py` | Main dashboard — live quotes, order book, positions, holdings, RSI scanner |
| `auth/kite_auth.py` | OAuth flow with Selenium auto-login + **automated TOTP** via `pyotp` (zero-touch 2FA when `ZERODHA_TOTP_SECRET` is set; falls back to visible browser for manual entry) |
| `auth/kite_session.py` | Reusable authenticated `KiteConnect` session |
| `core/config.py` | API credentials, DB config, index groups (NIFTY50, BANKNIFTY, NIFTYIT, NIFTYENERGY) |
| `core/database_service.py` | PostgreSQL connection pool for `livestocks_ind` database |
| `core/selenium_service.py` | Chrome/Edge WebDriver lifecycle management (headless mode via `--headless=new`) |
| `nse/screener.py` | **3-stage NSE screener** — liquidity filter → volatility filter → technical composite score (RSI + MACD + Bollinger + volume surge + price range) |
| `nse/nse_universe.py` | NSE symbol list download (NIFTY50, BANKNIFTY, full NSE) |
| `options/option_chain.py` | Concurrent option chain with OI, Greeks, and IV (ThreadPoolExecutor, 20 workers) |
| `trading/order_service.py` | Market/Limit/SL/SL-M orders, CNC/MIS/NRML products, DAY/IOC validity — **auto-persists every order to DB** (`order_records` table) + **sends email confirmation** + **circuit breaker** (3 failures → 120s halt) |
| `trading/auto_executor.py` | End-to-end Carver pipeline: screen → score → forecast → vol-target size → risk-check → order → monitor |
| `trading/risk_manager.py` | Vol-targeted position sizing, 6-tier drawdown protection, ATR-based SL/TP, VIX/ADX regime scaling, sector limits (30% cap, 3 trades/sector), portfolio correlation filter (reject > 0.60) |
| `trading/trade_monitor.py` | Post-trade SL/TP lifecycle — SL-M + limit TP after entry fill, **trailing stop** (5% profit lock → 3% trail), **crash recovery** (SQLite WAL persistence, auto-restore on restart), corporate action adjustments, forced exit after max hold period |
| `trading/paper_trader.py` | Virtual broker — simulates fills with live Kite LTP + slippage model, persists P&L & trades to SQLite |
| `trading/rsi_strategy.py` | Live RSI scanner — BUY (RSI<30 + reversal), SELL (RSI>70 + reversal), auto-order placement |

### Exit Management Lifecycle

The `TradeMonitor` manages the full post-entry lifecycle:

```
Entry fill confirmed → place SL-M + limit TP orders
  ├── SL triggered      → close trade, cancel orphaned TP, realise loss
  ├── TP filled          → close trade, cancel orphaned SL, realise gain
  ├── Trailing stop      → ratchet SL up after 5% profit (3% trail distance)
  ├── Forced exit        → close at market after max hold (10d swing / 30d positional)
  ├── Corporate action   → adjust quantity + SL/TP for split/bonus
  ├── TP expiry (DAY)    → re-place next trading day
  └── Crash recovery     → SQLite WAL persistence, auto-restore on restart
```

**Stop-Loss Methods (configurable):** MA50-based, swing-low (10d), ATR-based (2.0× ATR), or tightest-of-three. Min SL: 5%, Max SL: 8% + 1.5σ gap buffer for overnight NSE risk.

### Real-time Streaming Architecture

Push-based tick distribution via Kite WebSocket (KiteTicker) with an internal event dispatcher:

| Component | Purpose |
|-----------|--------|
| `webhooks/ticker.py` | `KiteWebSocketService` — manages KiteTicker connection, batch-flushes ticks every 0.5 s |
| `webhooks/dispatcher.py` | `WebhookDispatcher` — singleton fan-out to subscribers via ThreadPoolExecutor |
| `webhooks/handlers.py` | `DBTickHandler` (PostgreSQL), `UITickCache`, `NSEMarketStatusMonitor`, `SessionWatchdog` |
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
- **Automated TOTP 2FA** — when `ZERODHA_TOTP_SECRET` is set, Kite login is fully automated via `pyotp`: headless Chrome auto-fills credentials + TOTP, captures redirect token. Falls back to visible browser for manual entry if auto-fill fails
- **Order database persistence** — every order (BUY/SELL, MARKET/LIMIT/AMO, success/failure) is automatically saved to the `order_records` PostgreSQL table with fill_price, filled_at, and status
- **Email order confirmations** — styled HTML email sent via SMTP for every placed order (requires `CENTURION_EMAIL_*` env vars)
- **Circuit breaker** — 3 consecutive API failures → order placement halted for 120 seconds; auto-recovers via half-open test; manual reset available via `reset_circuit_breaker()`
- **Crash recovery** — TradeMonitor persists all open trade state to SQLite (WAL mode); on container/process restart, active trades are automatically restored and monitoring resumes
- **Paper trading engine** — virtual broker that simulates fills using live Kite LTP + configurable slippage; persists trades and P&L to SQLite for paper ↔ live reconciliation
- **3-leg reconciliation** — weekly automated comparison: backtest ↔ paper, paper ↔ live, backtest ↔ live (Sharpe, win-rate, return drift detection)
- Auto-refresh every 30 seconds via `@st.fragment(run_every=...)`
- Market status pill indicators from NSE API (pre-open, live, post-market)
- Batch quote fetching (200 symbols/batch)
- Option chain: expiry discovery (45 days + monthly), Sensibull-style colouring, ATM highlighting, PCR metric
- Price alerts: `price_above`, `price_below`, `change_pct_above`, `change_pct_below`, `volume_above` with desktop notifications
- All-combinations pairs trading: C(n,2) pair analysis when >2 tickers provided
- **Portfolio Analyzer** — sector weights, allocation drift analysis from live Kite holdings
- **SELL Pipeline** — automated exit for SELL/STRONG_SELL verdicts on existing holdings

---

## 6. RAG Document Intelligence

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

## 7. Database Layer

### PostgreSQL Schema (14 tables)

Supports both local PostgreSQL + TimescaleDB and cloud-hosted **Neon** serverless PostgreSQL.

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `analysis_runs` | Tracks each analysis execution | status, tickers, duration, user_id |
| `news_items` | Scraped news with sentiment | ticker, source, sentiment_label, content_hash (SHA-256 dedup) |
| `stock_signals` | Trading signals | decision, decision_score, reasoning, technical indicators |
| `fundamental_metrics` | Per-ticker fundamental snapshots | PE, PEG, ROE, Z-Score, M-Score, F-Score |
| `order_records` | Every Kite order (BUY/SELL, success/failure) | symbol, side, quantity, fill_price, filled_at, status, order_id, exchange |
| `trade_journal` | Live trade journal with entry/exit tracking | symbol, side, strategy, entry_price, exit_price, pnl, holding_period, notes |
| `backtest_results` | Strategy backtest outcomes | total_return, sharpe_ratio, max_drawdown, equity_curve (JSONB) |
| `backtest_trades` | Individual trade records | entry/exit price, PnL, holding period |
| `backtest_equity_points` | Equity curve data points | portfolio_value, drawdown, benchmark |
| `backtest_daily_returns` | Daily return series | daily_return, cumulative_return |
| `strategy_performance_summary` | Materialised gold layer | avg metrics across all backtests per strategy |
| `user_watchlists` | Custom ticker lists | tickers (ARRAY), is_default |
| `alert_configurations` | Price/signal alerts | conditions (JSONB), notification_channels |
| `raw_scraped_news` | Bronze/raw layer | raw content, is_processed flag |
| `data_freshness` | Cache staleness tracking | last_fetched_at, consecutive_errors |

**TimescaleDB** (optional, local only): Hypertables on `stock_signals`, `fundamental_metrics`, `news_items` with 7-day chunk interval. Disabled for Neon (`DB_ENABLE_TIMESCALEDB=false`).

### Neon Serverless PostgreSQL (Cloud)

The database connection layer supports direct `DATABASE_URL` connection strings for Neon:

| Feature | Detail |
|---------|--------|
| **Connection** | `CENTURION_DATABASE_URL` env var (Neon pooled string) |
| **SSL** | Auto-forced `sslmode=require` for Neon endpoints |
| **Pool** | Smaller defaults (5 pool, 5 overflow) for serverless connection limits |
| **Recycle** | 300s `pool_recycle` to handle Neon's connection timeout |
| **Pre-warming** | `DatabaseManager.pre_warm()` wakes Neon auto-suspended compute before market hours |
| **URL rewrite** | `postgres://` → `postgresql+psycopg2://` handled automatically |

### Service Layer

`DatabaseService` (singleton) provides a unified API:
- Analysis lifecycle: `start_analysis_run()` `complete_analysis_run()` / `fail_analysis_run()`
- Persistence: `save_signals()`, `save_news_items()` (SHA-256 dedup), `save_fundamental_metrics()` (upsert)
- Order persistence: `save_single_order()` — auto-called by `order_service.place_order()` for every order (maps fill_price, filled_at, status)
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

## 8. Object Storage (MinIO / Cloudflare R2)

S3-compatible storage for backtest chart images. Supports **MinIO** (local development) and **Cloudflare R2** (production — free tier: 10 GB, zero egress fees):

- **Path pattern**: `centurion-backtests/<run_id>/<TICKER>/<strategy_name>/<filename>`
- **Metadata tags**: `x-amz-meta-run-id`, `x-amz-meta-strategy`, `x-amz-meta-ticker`, `x-amz-meta-chart-title`
- **Formats**: matplotlib (base64 PNG), plotly (JSON), backtesting.py (HTML)
- **Presigned URLs**: 1-hour expiry for History page viewing
- **R2 auto-detection**: Endpoints containing `r2.cloudflarestorage.com` auto-force HTTPS and pass `region=auto`
- **Backup prefix**: `backups/<YYYY-MM-DD>/` for nightly SQLite database backups

```python
from storage.minio_service import get_minio_service

minio = get_minio_service()
minio.save_backtest_image(run_id, image_data, filename, strategy_name, ticker, chart_title)
minio.upload_file("/path/to/file.sqlite3", "backups/2026-03-23/cache.sqlite3")
images = minio.get_backtest_images(run_id)  # with presigned URLs
details = minio.list_runs_detailed()         # metadata: size, chart count, strategies
```

---

## 9. Interactive Web Interface

### Pages

| Page | Route | Description |
|------|-------|-------------|
| **Main** | `main` | Ticker selection (default / manual / CSV upload), output settings, Run Analysis button |
| **Stock Analysis** | `analysis` | Multi-colour CSS spinner during analysis 4-tab results (Overview, Detailed Table, Top Signals, Sentiment) |
| **Fundamental** | `fundamental` | Z/M/F score interpretations, all-stocks table, three charts side-by-side |
| **Backtesting** | `backtesting` | Auto pre-computes all strategies on first visit; config panel + per-strategy result tabs with charts |
| **US Holdings** | `us_holdings` | US portfolio holdings view |
| **Crypto** | `crypto` | Isolated crypto strategies (default: ETH, BTC, LTC); Binance data, separate cache |
| **History** | `history` | 3 tabs: Analysis Runs (drill-down), Trading Signals (filterable), Backtest Results (with MinIO charts) |
| **RAG** | `rag` | PDF upload, query input with KB source selector, streaming response, code applicator |
| **Financial ML** | `finance_ml` | 19 AFML chapter analyses — data structures, labeling, feature importance, HRP, CSCV, and more |
| **Test & Tune** | `testune_ts` | 7 chapter analyses from *Testing and Tuning Market Trading Systems* (Timothy Masters) |
| **Indian Main** | `ind_main` | Indian equities analysis dashboard with auto-order execution for STRONG_BUY signals |
| **NSE Screener** | `screener` | 3-stage NSE screener → IntegratedScorer verdicts → risk-managed order placement (auto or manual) |
| **Indian Equities** | `ind_kite` | Live quotes, order book, positions, holdings, option chain, RSI scanner (Kite Connect) |
| **Verdict** | `verdict` | IntegratedScorer 5-layer verdict with composite scores, layer breakdowns, reasoning (IND & US) |
| **Options** | `options` | Concurrent option chain with OI, Greeks, IV, Sensibull-style colouring |

### Next.js Frontend (Primary UI)

A modern React-based frontend built with Next.js 14, Tailwind CSS, and TanStack Query (React Query v5). Connects to the FastAPI backend at `http://localhost:9001`.

| Feature | Description |
|---------|-------------|
| **Authentication** | JWT token-based login with signed session cookies (8-hour TTL, 30-min inactivity timeout) |
| **User Menu** | Header dropdown showing username, avatar initials, dark/light mode toggle, Settings link, and logout |
| **Dark / Light Mode** | `next-themes` provider with system detection; toggle available in header menu and Settings page |
| **Settings Page** | `/settings` — profile info, appearance theme picker (Light / Dark / System), change password form |
| **RAG Engine** | PDF upload (drag-and-drop, direct to backend), async background ingestion with polling status, SSE streaming query with token-by-token LLM response, knowledge base with document metadata |
| **Sidebar** | Collapsible sidebar with US/IND stock tabs, module navigation (Financial ML, Test & Tune, Crypto, RAG Engine) |
| **Ticker Ribbon** | Scrolling LTP ribbon with TTL-cached backend prices |
| **Lazy Loading** | `loading.tsx` skeleton + `dynamic()` imports with `ssr: false` for heavy components |
| **Financial ML** | Ticker input (Default / Manual / CSV), calendar popover date pickers, chapter selection with descriptions, spinner progress indicator, collapsible chapter results |
| **Verdict Pages** | 5-layer IntegratedScorer verdict for both US and IND stocks with composite scores, layer breakdowns, and reasoning |
| **Calendar Popover** | `react-day-picker` v9 date pickers with Radix Popover, styled for dark theme |

**Tech stack:** Next.js 14, React 18, TypeScript, Tailwind CSS, Radix UI primitives, TanStack Query v5, Zustand (auth state), `next-themes`, `react-day-picker` v9, `date-fns`, Lucide icons.

### Authentication
- YAML-based credentials (`auth/credentials.yaml`)
- Bcrypt password hashing (with legacy SHA-256 support via `hmac.compare_digest`)
- JWT tokens via `itsdangerous` signed serializer (8-hour TTL)
- Session timeout: 30 min inactivity, 8 hours absolute
- Password change via Settings page (`POST /api/v1/auth/change-password`)
- Default users: `admin`/`admin123`, `analyst`/`analyst123`

### Styling
- Enterprise CSS: dark gradient theme with Centurion branding (dark mode default)
- Light mode support via Tailwind CSS `dark:` variants and CSS custom properties
- Decision colours: STRONG_BUY `#00ff88`, BUY `#00cc44`, HOLD `#ffd700`, SELL `#ff6b6b`, STRONG_SELL `#ff0000`
- Background image overlay, custom buttons, consistent footer

---

## 10. Financial ML & Test-and-Tune

Two book-based quantitative research modules share the same UI pattern — tabbed chapter analyses, async background pre-computation, MinIO figure persistence, and PostgreSQL result storage.

### Financial ML (AFML)

Based on *Advances in Financial Machine Learning* by Marcos López de Prado. 19 chapter scripts in `financial_ML/applied/` covering:

| Tab | Chapters |
|-----|----------|
| Data Structures | Financial Data Structures, Triple-Barrier Labeling, Sample Weights |
| Features | Fractional Differentiation, Feature Importance, Structural Breaks, Entropy, Microstructure |
| Modeling | Ensemble Methods, Cross-Validation, Hyper-Parameter Tuning, Bet Sizing |
| Backtesting | Dangers of Backtesting, Synthetic Backtesting, Backtest Statistics, Strategy Risk |
| Portfolio | ML Asset Allocation (HRP) |
| Computation | Multiprocessing & Vectorization, Brute Force & Quantum |

Next.js page: `frontend/app/(dashboard)/financial-ml/page.tsx` — Route: `/financial-ml` — with ticker input (Default / Manual / CSV), calendar popover date pickers, chapter selection, spinner progress, and collapsible results

### Test & Tune (TTMTS)

Based on *Testing and Tuning Market Trading Systems* by Timothy Masters (2018). 7 chapter scripts in `testune_trade_sys/applied/` covering:

| Tab | Chapters |
|-----|----------|
| Foundations | Introduction (returns, future leak, percent wins), Pre-Optimization Issues (stationarity, entropy) |
| Optimization | Optimization Issues (elastic-net, differential evolution), Post-Optimization Issues (StocBias, sensitivity) |
| Performance Estimation | Unbiased Performance (walk-forward, CSCV), Trade-Based Analysis (BCa bootstrap, drawdown bounds) |
| Statistical Testing | Permutation Tests (return/price/bar permutation, walk-forward permutation) |

C++ algorithms from the book are converted to Python (NumPy/SciPy). Each chapter has a companion reading in `testune_trade_sys/readings/`.

Next.js page: `frontend/app/(dashboard)/test-tune/page.tsx` — Route: `/test-tune`

### Shared Architecture

Both modules follow the same pattern:
- `sample_data.py` — Data generators with yfinance caching to `_cache/` (parquet)
- `applied/chNN_*.py` — Chapter scripts with algorithm functions and a `main()` entry point
- `readings/chNN_*.md` — Companion documentation
- Chapter execution via `importlib`, matplotlib figure capture (PNG bytes), MinIO + PostgreSQL persistence

---

## 11. Project Structure

```
centurion_core/
├── main.py                       # Core orchestration (AlgoTradingSystem)
├── config.py                     # Configuration (~140 settings, CENTURION_* env vars)
├── models.py                     # Data models (NewsItem, StockMetrics, TradingSignal)
├── utils.py                      # CSV parsing and ticker validation
├── scheduler.py                  # APScheduler — 5 jobs (pre-market, intraday, walk-forward, reconciliation, backup)
├── run_api.py                    # FastAPI server launcher (port 9001)
├── setup_database.py             # Database schema initialisation
├── requirements.txt              # Python dependencies
├── sample_tickers.csv            # Example ticker list
├── .env.example                  # Complete environment variable reference
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD: HF Spaces backend + Vercel frontend deployment
│
├── optimizer/                    # Walk-forward signal weight optimisation
│   ├── optimize_weights_r21a.py  # R21a — differential evolution on 11 signals, checkpoint resume
│   └── analyze_r21a.py           # Post-optimisation analysis and reporting
│
├── runners/                      # CLI entry points for standalone tasks
│   ├── run_backtest.py           # Full pipeline backtest runner
│   ├── run_r21a.py               # R21a validation with real engine
│   ├── run_extract_forecasts.py  # Extract per-source daily forecasts to .pkl
│   ├── run_r21a_pipeline.py      # End-to-end R21a pipeline (extract → optimise → validate)
│   └── run_contra_v4.py          # Contra regime strategy runner
│
├── cloud/                        # Cloud compute runners (Kaggle, Colab, Modal)
│   ├── run_cloud_kaggle.py       # Kaggle free-tier runner (4 CPU, 29 GB RAM, 12-hr sessions)
│   ├── run_cloud_colab.py        # Google Colab runner
│   ├── run_cloud_modal.py        # Modal serverless runner
│   └── run_kaggle.py             # Kaggle dataset + notebook management
│
├── auth/                         # Authentication
│   ├── shared_session.py         # Cross-app SSO token signing (itsdangerous)
│   └── credentials.yaml          # User credentials (bcrypt hashed)
│
├── database/                     # PostgreSQL persistence layer (local + Neon serverless)
│   ├── connection.py             # SQLAlchemy engine (QueuePool, SSL, DATABASE_URL, Neon auto-detect)
│   ├── models.py                 # ORM models (15 tables incl. TradeJournal)
│   ├── service.py                # DatabaseService singleton
│   └── repositories/             # Repository pattern (6 repos + base)
│
├── scrapers/                     # News scraping modules
│   ├── us_aggregator.py          # US market concurrent coordinator (Semaphore, 3-layer cache)
│   ├── ind_aggregator.py         # Indian market news aggregator
│   ├── broader_sentiment.py      # Macro / broader market sentiment
│   ├── morningstar.py            # Morningstar data scraper
│   ├── cache.py                  # Rate limiter + content deduplicator
│   ├── us_news/                  # US news source scrapers (Yahoo, Finviz, Investing, TradingView, WSB)
│   ├── ind_news/                 # Indian news source scrapers
│   └── macro/                    # FII/DII tracker, India Fear & Greed, RBI/NSDL, macro indicators
│
├── sentiment/                    # AI sentiment analysis
│   └── analyzer.py               # DistilBERT implementation
│
├── metrics/                      # Financial metrics
│   └── calculator.py             # Fundamentals + technicals (yfinance)
│
├── decision_engine/              # Trading logic
│   └── engine.py                 # Weighted scoring with regime-adaptive thresholds
│
├── infrastructure/               # Platform infrastructure & reliability
│   ├── event_bus.py              # In-process pub/sub with JSONL replay; correlation IDs, priorities
│   ├── fault_isolation.py        # SupervisedWorker + CircuitBreaker (cascading failure prevention)
│   ├── latency_tracker.py        # Microsecond SLA tracking (p50/p95/p99 sliding window)
│   ├── model_registry.py         # Lazy-loading ML model registry (thread-safe singleton)
│   ├── replay_engine.py          # Deterministic event replay from JSONL logs
│   ├── timeseries_store.py       # TimescaleDB (live) / in-memory ring buffer (replay)
│   ├── execution_context.py      # Dual-mode context (live / paper / backtest)
│   ├── analysis_pipeline.py      # 8-stage institutional pipeline (Raw → Post-Trade)
│   ├── logging_config.py         # JSON structured logging with correlation IDs + Better Stack (Logtail)
│   ├── cache.py                  # Dual-layer cache: L1 in-memory + L2 Upstash Redis (lazy URL resolution)
│   └── backup_service.py         # Nightly SQLite backup to R2/MinIO
│
├── layers/                       # Architectural abstraction layers
│   ├── alpha_research.py         # Coordinates all alpha sources; emits alpha.signal events
│   ├── execution_engine.py       # Order routing: Kite (IND) / DriveWealth (US) / PaperBroker
│   ├── market_data.py            # Unified data feed (OHLCV, ticks, fundamentals, news)
│   ├── monitoring.py             # Health checks, latency dashboards, audit trail
│   ├── portfolio.py              # Allocation tracking, P&L, rebalancing
│   └── risk_engine.py            # Pre-trade + post-trade risk checks, drawdown circuit breaker
│
├── services/                     # Business logic & analysis services
│   ├── analysis.py               # Analysis orchestration (async)
│   ├── integrated_scorer.py      # 2-layer evaluation pipeline (core 45% + strategy/robustness 55%)
│   ├── forecast_combiner.py      # Carver 11-source forecast combination with FDM (~1.35)
│   ├── volatility_target.py      # 20% annual vol target, Half-Kelly sizing, rolling capital
│   ├── regime_hmm.py             # 3-state Gaussian HMM (Bull/Bear/Sideways); log-space forward-backward
│   ├── regime_detector.py        # 5-state fallback regime (VIX, NIFTY returns, ADX); adaptive thresholds
│   ├── strategy_decay.py         # 63-day rolling Sharpe monitor; auto-scale or blacklist degraded strategies
│   ├── options_overlay.py        # Covered calls + cash-secured puts (IV rank, delta-based strike selection)
│   ├── walk_forward.py           # Rolling walk-forward validation (1Y train / 1Q test)
│   ├── corporate_actions.py      # NSE SPLIT/BONUS/DIVIDEND/RIGHTS handler
│   ├── delivery_volume.py        # NSE delivery % analysis (≥60% = institutional conviction)
│   ├── earnings_momentum.py      # Post-earnings drift detector (5-day momentum boost)
│   ├── sector_rotation.py        # NIFTY sector momentum ranking (12 sectors)
│   ├── survivorship_filter.py    # Delisted/suspended/dead stock detector (4 methods)
│   ├── fundamental_freshness.py  # Intra-quarter freshness (bulk deals, promoter pledges, MF holdings)
│   ├── portfolio_analyzer.py     # Kite holdings analysis (sector weights, allocation drift)
│   ├── cache.py                  # SessionCache (TTL-aware, thread-safe)
│   └── drivewealth.py            # DriveWealth API client for US brokerage
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
├── financial_ML/                 # AFML chapter analyses (López de Prado)
│   ├── sample_data.py            # Data generators + yfinance caching (_cache/)
│   ├── applied/                  # 19 chapter scripts (ch02–ch21)
│   ├── readings/                 # Companion markdown docs
│   ├── _cache/                   # Parquet price cache (git-ignored)
│   └── _output/                  # Analysis outputs (git-ignored)
│
├── testune_trade_sys/            # Test & Tune chapter analyses (Timothy Masters)
│   ├── sample_data.py            # Data generators + yfinance caching (_cache/)
│   ├── applied/                  # 7 chapter scripts (ch01–ch07)
│   ├── readings/                 # Companion markdown docs
│   ├── _cache/                   # Parquet price cache (git-ignored)
│   └── _output/                  # Analysis outputs (git-ignored)
│
├── kite_connect/                 # Zerodha live trading (Indian markets)
│   ├── auth/                     # OAuth + Selenium auto-login + TOTP auto-fill (pyotp)
│   ├── core/                     # Config, PostgreSQL, Selenium (headless)
│   ├── nse/                      # NSE universe download + 3-stage screener
│   ├── options/                  # Concurrent option chain + Greeks
│   ├── trading/                  # Order service (circuit breaker), auto-executor, risk manager,
│   │   │                         # trade monitor (crash recovery), paper trader, RSI strategy
│   │   ├── order_service.py      # Idempotent retry + circuit breaker (3 failures → 120s halt)
│   │   ├── trade_monitor.py      # SL/TP lifecycle, trailing stops, SQLite crash recovery
│   │   ├── paper_trader.py       # Virtual broker — live LTP + slippage, SQLite persistence
│   │   ├── auto_executor.py      # Screen → signal-filter → risk → order → monitor pipeline
│   │   ├── risk_manager.py       # Position sizing, ATR SL/TP, regime scaling, sector limits
│   │   └── rsi_strategy.py       # Live RSI scanner with auto-order placement
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
│   ├── rag_page.py               # RAG page entry point
│   ├── core/                     # Query pipeline core
│   │   ├── query_engine.py       # 10-stage pipeline (~968 lines)
│   │   ├── reranker.py           # Cross-encoder re-ranking (code_mode support)
│   │   ├── hybrid_search.py      # BM25 + vector RRF fusion
│   │   ├── query_rewriter.py     # LLM query expansion + HyDE
│   │   ├── query_classifier.py   # Query intent classification
│   │   ├── semantic_cache.py     # Embedding-based answer cache (in-memory)
│   │   ├── retriever.py          # Unified retrieval interface
│   │   ├── context_builder.py    # Token-budget context assembly
│   │   └── fastpath.py           # Fast-path optimisations
│   ├── storage/                  # Vector & embedding storage
│   │   ├── vector_store.py       # ChromaDB HNSW cosine wrapper + DualIndexStore
│   │   ├── embeddings.py         # sentence-transformers (BGE-base-en-v1.5)
│   │   └── triplet_export.py     # Fine-tuning triplet generator
│   ├── ingestion/                # Document ingestion
│   │   ├── pdf_ingestion.py      # Structure-aware PDF chunking
│   │   ├── chunking.py           # Token-based chunking with code extraction
│   │   ├── tiered_retrieval.py   # FAQ tier (similarity >= 0.90)
│   │   └── background_ingest.py  # Background ingestion worker
│   ├── llm/                      # LLM integration
│   │   ├── llm_service.py        # Ollama / Claude / OpenAI abstraction
│   │   ├── evaluation.py         # IR metrics + LLM-as-Judge
│   │   └── code_applier.py       # RAG → strategy code applicator
│   ├── ui/                       # RAG UI widgets
│   │   └── ui_components.py      # Upload, query, response UI
│   └── utils/                    # Pipeline utilities
│       ├── token_counter.py      # tiktoken / heuristic counter
│       ├── perf_trace.py         # Pipeline stage timing
│       ├── retrieval_evaluator.py # Retrieval quality evaluation
│       └── time_budget.py        # Query time budget management
│
├── notifications/                # Desktop + email alerts
│   └── manager.py                # plyer popups + SMTP HTML email (order confirmations, WSB reports)
│
├── storage/                      # Object storage
│   ├── manager.py                # Excel/CSV file export
│   └── minio_service.py          # S3 client (MinIO local / Cloudflare R2 production)
│
├── api/                          # FastAPI REST API layer
│   ├── main.py                   # App factory, auth-gated /docs, Sentry init
│   ├── auth.py                   # Token signing, login/logout
│   ├── dependencies.py           # Dependency injection (DB, Kite, RAG)
│   ├── schemas/                  # Pydantic v2 request/response models
│   │   ├── common.py             # Shared: SuccessResponse, Pagination
│   │   ├── us_stocks.py          # Analysis, news, signals, backtest
│   │   ├── ind_stocks.py         # Kite auth, quotes, orders, options
│   │   ├── rag.py                # Ingest, query, evaluation
│   │   ├── crypto.py             # Prices, backtest, strategies
│   │   └── streaming.py          # SSE, WebSocket, Postback, OHLC, Alerts
│   └── routers/                  # Route modules (50+ endpoints)
│       ├── health.py             # GET /health (includes cache health: backend type, key count)
│       ├── us_stocks.py          # 9 endpoints
│       ├── ind_stocks.py         # 11 endpoints
│       ├── rag.py                # 10 endpoints
│       ├── v1_gateway.py         # 50+ /api/v1/* endpoints (primary Next.js gateway)
│       ├── crypto.py             # 4 endpoints
│       └── streaming.py          # 9 endpoints (SSE, WS, postback, OHLC, alerts, status)
│
└── deployment/                   # Deployment configs
    ├── Dockerfile                # Production: Python 3.11-slim, FastAPI + Scheduler
    ├── start.sh                  # Dual-process entrypoint (graceful shutdown + backup-on-exit)
    ├── docker-compose.yml        # FastAPI + MinIO containers
    ├── deploy.ps1 / deploy.sh    # General deployment
    ├── deploy-azure.ps1          # Azure deployment
    ├── deploy-gcp.ps1            # GCP deployment
    ├── DEPLOYMENT.md             # Cloud deployment guide
    └── DOCKER_QUICKSTART.md      # Docker quick start guide
```

---

## 12. Installation

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
        Write-Host " Port $port is in use" -ForegroundColor Yellow
    } else {
        Write-Host " Port $port is available" -ForegroundColor Green
    }
}
```

If ports are in use, either:
1. Kill the process: `Get-Process -Name processname | Stop-Process`
2. Or update `.env` to use different ports

---

### Step 1: Clone Repository & Setup Python Environment

```powershell
# Clone the repository (dev branch)
git clone -b dev https://github.com/srees16/centurion_core.git
cd centurion_core

# Create and activate virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Verify Python version
python --version  # should be 3.10+

# Install Python dependencies (installs DistilBERT ~250MB on first run)
pip install --upgrade pip
pip install -r requirements.txt

# Install Next.js frontend dependencies
cd frontend
npm install
cd ..
```

**Expected output:**
```
Successfully installed psycopg2-binary==2.9.X ...
```

---

### Step 2: Set Up PostgreSQL Database

Choose **Option A (Docker)** or **Option B (Local PostgreSQL)**.

#### **Option A: PostgreSQL via Docker** (Recommended)

```powershell
docker run -d --name centurion-postgres -p 9003:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=superadmin1 -e POSTGRES_DB=centurion_rag timescale/timescaledb:latest-pg15; Start-Sleep -Seconds 9; docker exec centurion-postgres psql -U postgres -c "CREATE DATABASE centurion_trading;"; docker exec centurion-postgres psql -U postgres -c "CREATE DATABASE livestocks_ind;"; docker ps | findstr centurion-postgres
```

#### **Option B: Local PostgreSQL Installation**

```powershell
# Verify PostgreSQL is installed and running in Windows Service Status:
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
 Database connection successful
 Database tables created successfully
 Database service layer ready
 Database setup completed successfully!
```

**If this fails:**
- Check PostgreSQL is running: `docker ps | findstr centurion-postgres`
- Verify port 9003 is listening: `Test-NetConnection -ComputerName localhost -Port 9003`
- Check password matches in `.env`

---

### Step 4: Set Up MinIO Object Storage (for Backtest Charts)

```powershell
docker run -d --name centurion-minio -p 9004:9000 -p 9002:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin123 minio/minio:latest server /data --console-address ":9001"
Start-Sleep -Seconds 5
docker ps | findstr centurion-minio
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
winget install Ollama.Ollama
ollama pull qwen2.5:3b

# Verify Ollama is running (should listen on port 11434)
Test-NetConnection -ComputerName localhost -Port 11434
```

---

### Step 6: Configure Environment Variables

If you already set environment variables in Step 2 of the Quick Start, you can skip this step. Otherwise, create a `.env` file in the `centurion_core/` root directory:

```ini
# ═══════════════════════════════════════════════════════════════════
# CRITICAL: Copy this entire block to .env (replace YOUR_*_HERE)
# ═══════════════════════════════════════════════════════════════════

# ─── FastAPI Backend ──────────────────────────────────────────────
API_PORT=9001

# ─── PostgreSQL (Analysis, Backtesting, RAG) ───────────────────────
CENTURION_DB_HOST=localhost
CENTURION_DB_PORT=9003
CENTURION_DB_NAME=centurion_rag
CENTURION_DB_USER=postgres
CENTURION_DB_PASSWORD=superadmin1

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
ZERODHA_API_KEY=YOUR_KEY_HERE
ZERODHA_API_SECRET=YOUR_SECRET_HERE
ZERODHA_USER_ID=YOUR_USER_HERE
ZERODHA_PASSWORD=YOUR_PASSWORD_HERE
ZERODHA_TOTP_SECRET=YOUR_BASE32_TOTP_SECRET

# ─── Email Notifications (Order Confirmations) ─────────────────────
# Gmail: enable 2-Step Verification → https://myaccount.google.com/apppasswords
CENTURION_EMAIL_HOST=smtp.gmail.com
CENTURION_EMAIL_PORT=587
CENTURION_EMAIL_USER=YOUR_GMAIL_HERE
CENTURION_EMAIL_PASS=YOUR_GMAIL_APP_PASSWORD

# ─── KiteConnect Connection Pool ──────────────────────────────────
KITE_POOL_MAXSIZE=40

# ─── RAG Document Pipeline ────────────────────────────────────────
CENTURION_RAG_LLM_URL=http://localhost:11434
RAG_MODEL=qwen2.5:3b
CENTURION_RAG_CHROMA_DIR=./data/chroma_db
CENTURION_RAG_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
CENTURION_RAG_CONTEXT_TOKEN_BUDGET=2000
CENTURION_RAG_MAX_CONTEXT_CHUNKS=8
CENTURION_RAG_TOP_K=15
CENTURION_RAG_SIMILARITY_THRESHOLD=0.70
CENTURION_RAG_LLM_NUM_CTX=4096
CENTURION_RAG_LLM_NUM_PREDICT=500
CENTURION_RAG_LLM_MAX_TOKENS=500
CENTURION_RAG_LLM_TEMPERATURE=0.2
CENTURION_RAG_LLM_FIRST_TOKEN_TIMEOUT=300
CENTURION_RAG_LLM_CHUNK_TIMEOUT=30
CENTURION_RAG_QUERY_BUDGET=300
CENTURION_RAG_QUERY_REWRITE=false
CENTURION_RAG_STREAMING=true
CENTURION_RAG_FAQ_ENABLED=false
RAG_FAST_MODE=false
CENTURION_RAG_CACHE_ENABLED=false

# ─── Authentication ───────────────────────────────────────────────
CENTURION_DEFAULT_ADMIN_PASSWORD=admin123
CENTURION_DEFAULT_ANALYST_PASSWORD=analyst123

# ─── Cloud LLM (Claude — default provider) ─────────────────────────
CENTURION_RAG_LLM_PROVIDER=claude
ANTHROPIC_API_KEY=YOUR_KEY_HERE
CENTURION_RAG_CLAUDE_MODEL=claude-opus-4-20250514
CENTURION_RAG_CLAUDE_MAX_TOKENS=1024
CENTURION_RAG_CLAUDE_TEMPERATURE=0.2

# ─── Optional: OpenAI (uncomment to use instead of Claude) ──────────
# CENTURION_RAG_LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_key_here

# ─── Neon PostgreSQL (cloud — overrides local DB settings) ──────────
# CENTURION_DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require

# ─── Upstash Redis (cloud caching — falls back to in-memory if unset)
# UPSTASH_REDIS_URL=rediss://default:token@host.upstash.io:6379

# ─── Cloudflare R2 (production object storage — overrides local MinIO)
# MINIO_ENDPOINT=<account_id>.r2.cloudflarestorage.com
# MINIO_ACCESS_KEY=<r2_access_key>
# MINIO_SECRET_KEY=<r2_secret_key>
# MINIO_SECURE=true
# MINIO_REGION=auto

# ─── Sentry (error tracking) ──────────────────────────────────────
# SENTRY_DSN=https://<key>@<org>.ingest.sentry.io/<project_id>
# SENTRY_TRACES_SAMPLE_RATE=0.2
# SENTRY_ENVIRONMENT=development

# ─── Better Stack / Logtail (log aggregation) ─────────────────────
# LOGTAIL_TOKEN=<your_source_token>
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
    print(' PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f' PostgreSQL error: {e}')
"

# Test MinIO connection
python -c "
from minio import Minio
try:
    client = Minio('localhost:9004', access_key='minioadmin', secret_key='minioadmin123', secure=False)
    client.bucket_exists('centurion-backtests')
    print(' MinIO connection successful')
except Exception as e:
    print(f' MinIO error: {e}')
"

# Test Ollama (if using RAG)
# Test-NetConnection -ComputerName localhost -Port 11434
```

---

### Step 8: Launch the Application

**Terminal 1 — FastAPI Backend:**

```powershell
cd centurion_core
.\myenv\Scripts\Activate.ps1
python run_api.py
```

Backend API at: **http://localhost:9001** — API docs at **http://localhost:9001/docs**

**Terminal 2 — Next.js Frontend:**

```powershell
cd centurion_core-fe
npm run dev
```

Opens at: **http://localhost:3000** — login with `admin` / `admin123`

---

### Step 9: Login & Verify Application

1. Open http://localhost:3000 in your browser
2. Login with default credentials:
   - **Username**: `admin`
   - **Password**: `admin123`
3. Navigate to **Main** page — ensure no error messages appear
4. Try a quick analysis with 2-3 tickers (e.g., AAPL, MSFT, GOOGL)
5. Check **History** **Analysis Runs** to verify database persistence

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
| `ModuleNotFoundError: No module named 'X'` | Dependencies not installed | Run `pip install -r requirements.txt` |
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

The platform supports deployment on free-tier cloud services:

| Service | Role | Free Tier |
|---------|------|-----------|
| **HF Spaces** | FastAPI backend + APScheduler | 2 vCPU, 16 GB RAM, 50 GB disk |
| **Vercel** | Next.js frontend | Unlimited static, 100 GB bandwidth |
| **Neon** | PostgreSQL (serverless) | 0.5 GB storage, auto-suspend |
| **Cloudflare R2** | Object storage (S3-compatible) | 10 GB, zero egress |
| **Upstash Redis** | Caching layer | 10K commands/day |
| **GitHub Actions** | CI/CD pipeline | 2K minutes/month |

**Deployment files:**

| File | Purpose |
|------|---------|
| `deployment/Dockerfile.hf` | Production container for HF Spaces: Python 3.11-slim, FastAPI + Scheduler |
| `deployment/start.sh` | Dual-process entrypoint with graceful shutdown + backup-on-exit |
| `deployment/docker-compose.yml` | Local Docker Compose (FastAPI + MinIO) |
| `frontend/vercel.json` | Vercel deployment config with API rewrites |
| `.github/workflows/deploy.yml` | CI/CD: lint → deploy backend (HF Spaces) + frontend (Vercel) |
| `.env.example` | Complete environment variable reference for all services |

**Docker commands:**

**Windows PowerShell:**
```powershell
cd deployment; docker compose up -d
```

**macOS / Linux:**
```bash
cd deployment && docker-compose up -d
```

For detailed setup, see [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md) and [deployment/DOCKER_QUICKSTART.md](deployment/DOCKER_QUICKSTART.md).

---

## 13. Usage Guide

### Quick Start

1. Launch the app log in land on the **Main** page.
2. Select tickers (default list, manual entry, or CSV upload).
3. Click **Run Analysis** results appear on the **Stock Analysis** page.
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
3. The pipeline runs: EDA statistical tests portfolio construction backtesting optimisation.
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

Tab-based sub-navigation per market section:

**IND Stocks:** Main → Fly Kite → Fundamentals → Screener → Verdict → Backtest → Options → History

**US Stocks:** Main → Fundamentals → Verdict → Backtest → Holdings → History

**Modules:** Financial ML, Test & Tune, Crypto, RAG Engine, Settings

---

## 14. API Reference

### REST API (FastAPI)

A full REST API serves the Next.js frontend on a separate port (default `9001`).

**Interactive docs** — **http://localhost:9001/docs** (Swagger UI) and **http://localhost:9001/redoc** (ReDoc) are available after authenticating. On first visit you are redirected to a login page; use the same credentials as the frontend (e.g. `admin` / `admin123`). A signed session cookie (8-hour TTL) keeps you logged in.

| Module | Prefix | Endpoints | Examples |
|--------|--------|-----------|----------|
| Health | `/api/health` | 1 | DB, RAG, Kite status check |
| V1 Gateway | `/api/v1` | 50+ | `/analysis/run`, `/analysis/metrics`, `/macro/snapshot`, `/kite/auth`, `/fml/run`, `/verdict/run` |
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

### MinIO / R2 Service

```python
from storage.minio_service import get_minio_service

minio = get_minio_service()
path = minio.save_backtest_image(run_id, png_bytes, "equity_curve.png", "MACD Oscillator", "AAPL", "Equity Curve")
minio.upload_file("/path/to/file.sqlite3", "backups/2026-03-23/cache.sqlite3")
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

**Windows PowerShell:**
```powershell
# Start everything
cd deployment; docker compose up -d

# Start only MinIO
docker compose up -d minio

# View logs
docker logs centurion-minio

# Stop
docker compose down

# Remove all data (destructive)
docker compose down -v
```

**macOS / Linux:**
```bash
# Start everything
cd deployment && docker-compose up -d

# Start only MinIO
docker-compose up -d minio

# View logs
docker logs centurion-minio

# Stop
docker-compose down

# Remove all data (destructive)
docker-compose down -v
```

---

## 15. Troubleshooting

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
| Port in use | Change port in `.env` or kill the conflicting process |
| Slow first run | DistilBERT model download (~250 MB); subsequent runs are fast |

---

## 16. Dependencies

| Category | Packages |
|---|---|
| **Web Framework** | **Next.js 14** (React 18, Tailwind CSS, TanStack Query v5, react-day-picker v9, date-fns), plotly |
| **Data** | pandas, numpy, openpyxl |
| **Financial Data** | yfinance |
| **Crypto Data** | Binance public REST API (no key required) |
| **Live Trading** | kiteconnect (Zerodha Kite Connect SDK), pyotp (TOTP auto-fill) |
| **Scraping** | aiohttp, beautifulsoup4, lxml, requests, selenium, webdriver-manager |
| **AI/ML** | transformers, torch, scikit-learn |
| **LLM Providers** | anthropic, openai (Ollama via HTTP) |
| **RAG / Embeddings** | chromadb, sentence-transformers, PyMuPDF, tiktoken |
| **Analysis** | matplotlib, statsmodels, backtesting (0.6+), arch, scipy, seaborn |
| **Database** | sqlalchemy ≥ 2.0, psycopg2-binary ≥ 2.9, python-dotenv ≥ 1.0 |
| **Object Storage** | minio ≥ 7.2 (MinIO local / Cloudflare R2 production) |
| **Caching** | redis (Upstash Redis in production, in-memory fallback) |
| **Error Tracking** | sentry-sdk[fastapi] (Sentry — error capture + performance tracing) |
| **Log Aggregation** | logtail-python (Better Stack — cloud log shipping) |
| **Auth** | pyyaml ≥ 6.0, itsdangerous, bcrypt |
| **Notifications** | plyer |
| **API** | fastapi, uvicorn[standard], python-multipart |

---

## 17. Cloud Infrastructure & Observability

Production deployment uses a fully managed cloud stack with zero self-hosted servers.

### Production Architecture

```
User → Vercel (Next.js frontend)
         │
         ├── API calls → HF Spaces (FastAPI backend)
         │                  ├── Neon PostgreSQL (database)
         │                  ├── Upstash Redis (caching)
         │                  ├── Cloudflare R2 (object storage)
         │                  ├── Sentry (error tracking)
         │                  └── Better Stack (log aggregation)
         │
         └── Static assets → Vercel CDN
```

### Service Summary

| Service | Purpose | Free Tier | Dashboard |
|---------|---------|-----------|----------|
| **HF Spaces** | FastAPI backend hosting | Community GPU / CPU | huggingface.co/spaces |
| **Vercel** | Next.js frontend hosting | 100 GB bandwidth/month | vercel.com/dashboard |
| **Neon** | Serverless PostgreSQL | 0.5 GB storage, 1 project | console.neon.tech |
| **Upstash** | Serverless Redis (caching) | 10K commands/day | console.upstash.com |
| **Cloudflare R2** | S3-compatible object storage | 10 GB, zero egress | dash.cloudflare.com |
| **Sentry** | Error tracking + performance | 5K errors/month | sentry.io |
| **Better Stack** | Cloud log aggregation | 1 GB logs/month | logs.betterstack.com |

### Access URLs

| Service | URL | Notes |
|---------|-----|-------|
| **Frontend (prod)** | https://centurion-core-fe.vercel.app | Next.js — login with `admin` / `admin123` |
| **Frontend (local)** | http://localhost:3000 | `npm run dev` from `centurion_core-fe/` |
| **Backend API (prod)** | https://srees16-centurion-core.hf.space | HF Spaces — FastAPI |
| **Backend API (local)** | http://localhost:9001 | `python run_api.py` |
| **API Docs (local)** | http://localhost:9001/docs | Swagger UI (auth-gated) |
| **Neon Console** | https://console.neon.tech | Database management, SQL editor, branching |
| **Upstash Console** | https://console.upstash.com | Redis data browser, CLI, usage metrics |
| **Cloudflare Dashboard** | https://dash.cloudflare.com | R2 bucket browser, API tokens, usage |
| **Sentry Dashboard** | https://sentry.io | Issues, performance traces, releases |
| **Better Stack Live Tail** | https://logs.betterstack.com | Real-time log stream, search, alerts |
| **HF Spaces Settings** | https://huggingface.co/spaces/srees16/centurion_core/settings | Repository secrets, hardware, visibility |
| **Vercel Settings** | https://vercel.com/dashboard | Deployments, domains, env vars |
| **MinIO Console (local)** | http://localhost:9002 | `minioadmin` / `minioadmin123` |
| **Zerodha Kite** | https://kite.zerodha.com | Live trading, API app management |

---

### 17.1 HF Spaces (Backend)

The FastAPI backend runs on Hugging Face Spaces using a Docker container.

| Setting | Value |
|---------|-------|
| **Runtime** | Docker (Python 3.11-slim) |
| **Entrypoint** | `deployment/start.sh` — dual-process (FastAPI + APScheduler) |
| **Port** | 7860 (HF Spaces default) |
| **Secrets** | All `.env` variables added as HF Space secrets |

**Deployment**: Push to `main` branch → GitHub Actions builds and deploys to HF Spaces automatically.

**Secrets to configure** (HF Space → Settings → Repository secrets):

| Secret | Description |
|--------|-------------|
| `CENTURION_DATABASE_URL` | Neon PostgreSQL connection string |
| `UPSTASH_REDIS_URL` | Upstash Redis connection string |
| `MINIO_ENDPOINT` | Cloudflare R2 endpoint |
| `MINIO_ACCESS_KEY` | R2 access key ID |
| `MINIO_SECRET_KEY` | R2 secret access key |
| `SENTRY_DSN` | Sentry project DSN |
| `SENTRY_ENVIRONMENT` | `production` |
| `LOGTAIL_TOKEN` | Better Stack source token |
| `ANTHROPIC_API_KEY` | Claude API key |
| `ZERODHA_*` | Zerodha Kite Connect credentials (6 vars) |
| `CENTURION_EMAIL_*` | Gmail SMTP credentials (4 vars) |

---

### 17.2 Vercel (Frontend)

The Next.js 14 frontend deploys to Vercel with API rewrites to the HF Spaces backend.

| Setting | Value |
|---------|-------|
| **Framework** | Next.js 14 (auto-detected) |
| **Build** | `next build` |
| **Config** | `vercel.json` — API rewrites to HF Spaces URL |
| **Auth** | JWT token via signed cookies (8-hour TTL) |

**`vercel.json` rewrites**: All `/api/*` requests are proxied to the HF Spaces backend URL, keeping the frontend decoupled from the backend host.

---

### 17.3 Neon PostgreSQL (Database)

Serverless PostgreSQL with auto-suspend and connection pooling.

| Setting | Value |
|---------|-------|
| **Connection** | `CENTURION_DATABASE_URL` env var (pooled endpoint) |
| **SSL** | Auto-forced `sslmode=require` for Neon endpoints |
| **Pool** | 5 connections + 5 overflow (serverless-optimised) |
| **Recycle** | 300s `pool_recycle` for Neon connection timeout |
| **Pre-warming** | `DatabaseManager.pre_warm()` wakes suspended compute before market hours |
| **URL rewrite** | `postgres://` → `postgresql+psycopg2://` handled automatically |
| **Tables** | 14 tables in `centurion_rag` database |

---

### 17.4 Upstash Redis (Caching)

Dual-layer caching: L1 in-memory dict + L2 Upstash Redis for cross-restart persistence.

| Setting | Value |
|---------|-------|
| **Connection** | `UPSTASH_REDIS_URL` env var (TLS `rediss://`) |
| **Module** | `infrastructure/cache.py` — `CacheService` singleton |
| **L1** | In-memory Python dict (fastest, per-process) |
| **L2** | Upstash Redis (persistent, shared across restarts) |
| **Resolution** | Lazy `_resolve_url()` — defers env lookup until first use (dotenv compat) |
| **Health** | Reported in `GET /health` response (`cache.backend`, `cache.key_count`) |

**Caches using dual-layer (L1 + L2)**:

| Cache | Redis Key Pattern | TTL |
|-------|-------------------|-----|
| Layer 1 scorer | `l1:{symbol}` | 15 minutes |
| Ticker prices | `price:{symbol}` | 5 minutes |
| Sector rotation | `sector:rotation` | 1 hour |
| Fundamental freshness | `fresh:{symbol}` | 12 hours |

---

### 17.5 Cloudflare R2 (Object Storage)

S3-compatible storage for backtest charts, Financial ML figures, and nightly SQLite backups.

| Setting | Value |
|---------|-------|
| **Endpoint** | `MINIO_ENDPOINT` (R2 auto-detected via `r2.cloudflarestorage.com`) |
| **Auth** | `MINIO_ACCESS_KEY` + `MINIO_SECRET_KEY` (R2 API tokens) |
| **Bucket** | `centurion-backtests` (auto-created on first use) |
| **TLS** | `MINIO_SECURE=true` (auto-forced for R2) |
| **Region** | `MINIO_REGION=auto` |
| **Module** | `services/storage/minio_service.py` |
| **Availability** | `is_available()` uses `bucket_exists()` fallback (R2 denies `list_buckets()`) |
| **Presigned URLs** | 1-hour expiry for chart viewing in History page |

**Storage layout**:
```
centurion-backtests/
├── <run_id>/<TICKER>/<strategy>/<chart>.png    # Backtest charts
├── <run_id>/financial_ml/<chapter>/<fig>.png    # Financial ML figures
└── backups/<YYYY-MM-DD>/<db>.sqlite3            # Nightly SQLite backups
```

---

### 17.6 Sentry (Error Tracking)

Automatic error capture and performance tracing for the FastAPI backend.

| Setting | Value |
|---------|-------|
| **Package** | `sentry-sdk[fastapi]` |
| **Init** | `api/main.py` — `_init_sentry()` called at module import |
| **Integrations** | `FastApiIntegration`, `StarletteIntegration`, `LoggingIntegration` |
| **Breadcrumbs** | From `INFO` level and above |
| **Events** | From `ERROR` level and above |
| **Traces** | `SENTRY_TRACES_SAMPLE_RATE=0.2` (20% of requests) |
| **Graceful** | Skips silently if `sentry-sdk` not installed or `SENTRY_DSN` not set |

**Environment variables**:

| Variable | Description | Example |
|----------|-------------|--------|
| `SENTRY_DSN` | Project DSN from sentry.io | `https://key@org.ingest.sentry.io/id` |
| `SENTRY_TRACES_SAMPLE_RATE` | Performance trace sampling (0.0–1.0) | `0.2` |
| `SENTRY_ENVIRONMENT` | Environment tag | `production` / `development` |

---

### 17.7 Better Stack (Log Aggregation)

Cloud log shipping via Logtail — all structured JSON logs are forwarded to Better Stack for search, alerting, and dashboards.

| Setting | Value |
|---------|-------|
| **Package** | `logtail-python` |
| **Init** | `infrastructure/logging_config.py` — `setup_logging()` |
| **Handler** | `LogtailHandler` attached to root logger when `LOGTAIL_TOKEN` is set |
| **Format** | Structured JSON (correlation IDs, timestamps, module names) |
| **Graceful** | Skips silently if `logtail` not installed or `LOGTAIL_TOKEN` not set |
| **Live tail** | Real-time log stream at logs.betterstack.com |

**Environment variables**:

| Variable | Description | Example |
|----------|-------------|--------|
| `LOGTAIL_TOKEN` | Source token from Better Stack | `tBx3cfn8ihznc4A3hru5mUJT` |

---

### 17.8 Complete Cloud `.env` Reference

All cloud-specific environment variables (add to `.env` locally and as HF Spaces secrets for production):

```ini
# ─── Neon PostgreSQL ───────────────────────────────────────────────
CENTURION_DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require

# ─── Upstash Redis ─────────────────────────────────────────────────
UPSTASH_REDIS_URL=rediss://default:token@host.upstash.io:6379

# ─── Cloudflare R2 ─────────────────────────────────────────────────
MINIO_ENDPOINT=<account_id>.r2.cloudflarestorage.com
MINIO_ACCESS_KEY=<r2_access_key>
MINIO_SECRET_KEY=<r2_secret_key>
MINIO_SECURE=true
MINIO_BUCKET=centurion-backtests
MINIO_ENABLED=true
MINIO_REGION=auto

# ─── Sentry ────────────────────────────────────────────────────────
SENTRY_DSN=https://<key>@<org>.ingest.sentry.io/<project_id>
SENTRY_TRACES_SAMPLE_RATE=0.2
SENTRY_ENVIRONMENT=production

# ─── Better Stack / Logtail ───────────────────────────────────────
LOGTAIL_TOKEN=<your_source_token>
```

---

## Disclaimer

This software is provided for **educational and informational purposes only**. It does not constitute financial advice, investment recommendations, or professional trading guidance. Stock trading involves substantial risk of loss. Always consult qualified financial advisors before making investment decisions. Use at your own risk.

---

**Ready to get started?**

```bash
# Terminal 1: Backend
python run_api.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open **http://localhost:3000** and start analysing!
