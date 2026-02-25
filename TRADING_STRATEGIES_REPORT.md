# Centurion Core — Complete Trading Strategies Report

> **Generated**: Exhaustive documentation of every strategy file across  
> `trading_strategies/` and `strategies/` directories.  
> Every formula, parameter, threshold, signal logic, entry/exit condition,  
> and backtest mechanic is documented below.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Strategies Framework (`strategies/`)](#2-strategies-framework)
3. [Momentum Trading](#3-momentum-trading)
4. [Pattern Recognition](#4-pattern-recognition)
5. [Statistical Arbitrage](#5-statistical-arbitrage)
6. [Crypto Strategies](#6-crypto-strategies)
7. [FX Intraday](#7-fx-intraday)
8. [Derivatives / Options](#8-derivatives--options)
9. [Portfolio Analysis](#9-portfolio-analysis)
10. [Risk Modelling](#10-risk-modelling)
11. [Shared Utilities](#11-shared-utilities)
12. [Strategy Registry & Loader](#12-strategy-registry--loader)

---

## 1. Architecture Overview

The codebase runs a **dual architecture**:

| System | Location | Base Class | Data Source | Output |
|--------|----------|------------|-------------|--------|
| **New Framework** | `strategies/` + individual `*Strategy` classes | `BaseStrategy` (ABC) | `DataService` (yfinance wrapper, singleton, cached) | `StrategyResult` (charts, tables, metrics, signals, portfolio) |
| **Legacy Standalone** | `*_bktest.py` files | None — standalone scripts | Direct `yfinance` or CSV | matplotlib plots, printed stats |

### Dual System Flow

```
                 ┌─────────────────────┐
                 │ trading_strategies/  │
                 │   __init__.py        │  ← Lazy import registry (_LAZY_IMPORTS)
                 │                      │     11 strategies keyed by string ID
                 ├──────────────────────┤
                 │ momentum_trading/    │  4 BaseStrategy classes + 4 _bktest.py
                 │ pattern_recognition/ │  4 BaseStrategy classes + 5 _bktest.py
                 │ statistical_arbitrage│  2 BaseStrategy classes + 1 _bktest.py
                 │ crypto/             │  1 BaseStrategy class
                 │ fx_intraday/        │  2 standalone _bktest.py
                 │ derivatives/        │  2 standalone scripts
                 │ portfolio_analysis/ │  1 standalone script
                 │ risk_modelling/     │  1 standalone script
                 └─────────────────────┘

                 ┌─────────────────────┐
                 │ strategies/          │  ← Framework package
                 │  base_strategy.py    │  BaseStrategy ABC + dataclasses
                 │  registry.py         │  StrategyRegistry singleton
                 │  loader.py           │  Dynamic discovery + import
                 │  data_service.py     │  DataService (yfinance + cache)
                 │  utils.py            │  matplotlib_to_base64, RSI, MDD
                 └─────────────────────┘
```

---

## 2. Strategies Framework

### 2.1 BaseStrategy ABC (`strategies/base_strategy.py`, 620 lines)

**Dataclasses:**

| Dataclass | Fields |
|-----------|--------|
| `ChartData` | `title: str`, `data: str\|dict`, `chart_type: str = "matplotlib"`, `description: str`, `ticker: str` |
| `TableData` | `title: str`, `data: list[dict]`, `columns: list[str]`, `description: str` |
| `StrategyResult` | `charts: list[ChartData]`, `tables: list[TableData]`, `metrics: dict`, `signals: Optional[DataFrame]`, `portfolio: Optional[DataFrame]`, `success: bool`, `error_message: str`, `execution_time: float`, `metadata: dict` |
| `RiskParams` | `stop_loss_pct=0.05`, `take_profit_pct=0.10`, `max_position_size=0.25`, `max_drawdown_pct=0.20`, `trailing_stop=False`, `position_sizing="fixed"` |

**StrategyCategory Enum:**
`MOMENTUM`, `MEAN_REVERSION`, `PATTERN_RECOGNITION`, `STATISTICAL_ARBITRAGE`, `BREAKOUT`, `OPTIONS`, `OTHER`

**Abstract Method:**
```python
def run(self, tickers: list[str], start_date: str, end_date: str,
        capital: float, sentiment_data: Optional[dict] = None,
        risk_params: Optional[RiskParams|dict] = None, **kwargs) -> StrategyResult
```

**Class Attributes (override in subclasses):**
- `name`, `description`, `category`, `version = "1.0.0"`, `author = "Centurion Capital"`
- `requires_sentiment: bool = False`, `min_data_points: int = 50`

**Built-in Metric Calculators:**

| Method | Formula |
|--------|---------|
| `_calculate_sharpe(returns, rfr)` | $\text{Sharpe} = \frac{\bar{r} - r_f/252}{\sigma_r} \times \sqrt{252}$ |
| `_calculate_sortino(returns, rfr)` | $\text{Sortino} = \frac{\bar{r} - r_f/252}{\sigma_{\text{downside}}} \times \sqrt{252}$ |
| `_calculate_max_drawdown(values)` | $\text{MDD} = \min\left(\frac{V_t - \text{peak}_t}{\text{peak}_t}\right) \times 100$ |

**Portfolio Calculators:**

| Method | Logic |
|--------|-------|
| `_calculate_portfolio(signals, capital, risk)` | `shares = int(capital * max_position_size / max_close)`. `holdings = positions * Close * shares`. `cash = capital - cumsum(signals * Close * shares)`. `total_value = holdings + cash` |
| `_calculate_portfolio_long_only(...)` | Same but clips `positions` and `signals` to `>= 0` |

**Sentiment Adjustment Helpers:**

| Method | Effect |
|--------|--------|
| `_sentiment_scale_indicator(signals, column, sentiment, threshold=0.5, scale_factor=0.2)` | Multiplies indicator by `(1 + score * 0.2)` when `|score| > 0.5` |
| `_sentiment_zero_positions(signals, sentiment, threshold=-0.7)` | Zeros ALL positions when `score < -0.7` |
| `_sentiment_filter_positions(signals, sentiment, neg_threshold=-0.7, pos_threshold=0.7)` | Removes long (1) positions when negative, short (-1) when positive |
| `_sentiment_filter_signals(signals, sentiment, neg_threshold, pos_threshold, recalc_method)` | Removes buy/sell signals then recalculates positions via `cumsum().clip(0,1)` |

### 2.2 DataService (`strategies/data_service.py`, 417 lines)

**Singleton** pattern via `__new__`. In-memory cache with 1-hour expiry.

| Method | Details |
|--------|---------|
| `get_ohlcv(ticker, start, end, interval="1d")` | Downloads via `yf.download(..., auto_adjust=True)`. Cleans NaN/zero-volume rows. Forward-fills gaps (limit=3). |
| `get_multiple_ohlcv(tickers, ...)` | Iterates `get_ohlcv` per ticker |
| `preload_data(tickers, ...)` | Batch `yf.download(tickers, group_by='ticker')` for cache warming |
| `add_technical_indicators(df, indicators)` | SMA, EMA, RSI, MACD, Bollinger, ATR |

**Technical Indicator Formulas in DataService:**

| Indicator | Formula |
|-----------|---------|
| `sma_{n}` | `Close.rolling(n).mean()` |
| `ema_{n}` | `Close.ewm(span=n, adjust=False).mean()` |
| `rsi` | Wilder's EWM method (see `calculate_rsi` in utils below) |
| `macd` | `EMA(12) - EMA(26)`, signal = `EMA(macd, 9)` |
| `bollinger` | `mid = SMA(20)`, `upper = mid + 2σ`, `lower = mid - 2σ` |
| `atr` | $\text{ATR} = \text{rolling}_{14}\left(\max(H-L, |H-C_{-1}|, |L-C_{-1}|)\right)$ |

### 2.3 StrategyRegistry (`strategies/registry.py`, ~250 lines)

- `_strategies: dict[str, Type[BaseStrategy]]` — class-level dictionary
- `_normalize_name(name)`: lowercases, replaces spaces/hyphens with underscores
- `register(strategy_class, name=None)`: validates BaseStrategy inheritance, warns on overwrite
- `register_decorator`: class decorator version
- `get(name)` / `get_or_raise(name)`: lookup by normalized name
- `list_all()`, `list_by_category(category)`, `list_names()`, `count()`

### 2.4 Strategy Loader (`strategies/loader.py`, 335 lines)

- `STRATEGY_MODULE_MAP`: hardcoded dict mapping 8 strategy names to module paths
- `discover_strategies()`: scans `trading_strategies/` subdirs, finds `.py` files, strips `_bktest` suffix
- `load_strategy_module(path)`: uses `importlib.util.spec_from_file_location`
- `find_strategy_classes(module)`: finds all `BaseStrategy` subclasses in module
- `load_all_strategies(register=True)`: main entry — discover → load → register. Sets `_initialized` flag.
- `load_strategy_by_name(name)`: on-demand single strategy loading
- `get_strategy_categories()`: returns `{category: [strategy_names]}`

### 2.5 Shared Utilities (`strategies/utils.py`, 427 lines)

**RSI Calculation (shared across DataService and RSI strategy):**
```python
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    avg_gains = gains.ewm(com=period-1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
```

**Max Drawdown:**
```python
def calculate_max_drawdown(values):
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return float(drawdown.min() * 100)
```

**Trading Statistics Calculator:**
- Tracks position = 0/1, entry_price on buy signal, computes trade returns on sell
- Returns: `total_trades`, `winning_trades`, `losing_trades`, `win_rate`, `avg_win`, `avg_loss`, `profit_factor`

**Conversion Utilities:**
- `matplotlib_to_base64(fig, format="png", dpi=100)` → `data:image/png;base64,...`
- `plotly_to_json(fig)` → JSON dict
- `dataframe_to_table(df, title, max_rows, round_decimals=4)` → `{title, data, columns, description}`
- `create_metrics_summary(metrics, title)` → formatted table with human-readable names + formatters

---

## 3. Momentum Trading

### 3.1 Awesome Oscillator Strategy

**Files:** `awesome_oscillator.py` (470 lines, BaseStrategy), `awesome_oscillator_bktest.py` (standalone)

#### New Framework (`AwesomeOscillatorStrategy`)

| Attribute | Value |
|-----------|-------|
| Class | `AwesomeOscillatorStrategy(BaseStrategy)` |
| Category | `StrategyCategory.MOMENTUM` |
| `requires_sentiment` | `True` |
| `min_data_points` | `50` |

**Parameters:**

| Parameter | Default | Min | Max | Description |
|-----------|---------|-----|-----|-------------|
| `ao_short` | `5` | 2 | 20 | Short-period SMA window |
| `ao_long` | `34` | 20 | 100 | Long-period SMA window |

**Formulas:**
$$\text{median\_price} = \frac{H + L}{2}$$
$$\text{AO} = \text{SMA}(\text{median\_price}, 5) - \text{SMA}(\text{median\_price}, 34)$$

**Signal Logic:**
- `positions = 1` when `AO > 0`, else `0`
- `signals = positions.diff()`

**Entry:** Long when AO crosses above zero  
**Exit:** Close long when AO crosses below zero

**Sentiment Integration:** Calls `_sentiment_scale_indicator(signals, 'ao', sentiment)` to scale AO magnitude.

**Charts Generated (5):**
1. Price & Signals (buy/sell markers)
2. AO Histogram (green/red bars)
3. Moving Averages (short + long SMA)
4. Equity Curve (portfolio total value)
5. Drawdown Chart

#### Legacy Backtest (`awesome_oscillator_bktest.py`)

**Comparison:** AO vs MACD using identical periods.

| Parameter | AO | MACD |
|-----------|-----|------|
| Short period | 5 | 5 |
| Long period | 34 | 34 |
| Price input | `(High + Low) / 2` → SMA | `Close` → EWM |
| Data | yfinance AAPL 2016-01-01 to 2018-01-01, sliced at index 300 |
| Capital | $5,000 |
| Position size | 100 shares |

**Saucer Pattern (additional signal):**
- **Bearish Saucer:** AO negative + 2 consecutive green bars + 1 red bar → Short
- **Bullish Saucer:** AO positive + 2 consecutive red bars + 1 green bar → Long
- Saucer signals take priority over MA crossover; `cumsum` prevents duplicate signals

**Metrics:** Sharpe ratio (risk-free = 0), Maximum Drawdown

---

### 3.2 MACD Oscillator Strategy

**Files:** `macd_oscillator.py` (440 lines, BaseStrategy), `macd_oscillator_bktest.py` (standalone)

#### New Framework (`MACDOscillatorStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.MOMENTUM` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Min | Max | Description |
|-----------|---------|-----|-----|-------------|
| `ma_short` | `10` | 2 | 50 | Short MA period |
| `ma_long` | `21` | 10 | 200 | Long MA period |
| `use_ema` | `True` | — | — | EMA vs SMA toggle |

**Formulas (EMA mode):**
$$\text{ma\_short} = \text{EWM}(\text{Close}, \text{span}=10)$$
$$\text{ma\_long} = \text{EWM}(\text{Close}, \text{span}=21)$$
$$\text{oscillator} = \text{ma\_short} - \text{ma\_long}$$

**Formulas (SMA mode):**
$$\text{ma\_short} = \text{SMA}(\text{Close}, 10)$$
$$\text{ma\_long} = \text{SMA}(\text{Close}, 21)$$

**Signal Logic:**
- `positions = 1` when `ma_short >= ma_long`, else `0`
- `signals = positions.diff()`

**Sentiment:** Scales oscillator magnitude via `_sentiment_scale_indicator`.

#### Legacy Backtest (`macd_oscillator_bktest.py`)

- SMA-based (not EWM), `ma1=10, ma2=21`
- Original MACD: "12 and 26 → now 10 and 21" (adjusted for 5-day trading week)
- Data: yfinance AAPL 2016-01-01 to 2018-01-01, sliced at 300

---

### 3.3 Heikin-Ashi Strategy

**Files:** `heikin_ashi.py` (555 lines, BaseStrategy), `heikin-ashi_bktest.py` (standalone)

#### New Framework (`HeikinAshiStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.MOMENTUM` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Min | Max |
|-----------|---------|-----|-----|
| `confirmation_candles` | `1` | 1 | 5 |
| `use_ma_filter` | `False` | — | — |
| `ma_period` | `20` | 5 | 100 |

**Heikin-Ashi Formulas:**
$$\text{HA\_Close} = \frac{O + H + L + C}{4}$$
$$\text{HA\_Open} = \frac{\text{prev\_HA\_Open} + \text{prev\_HA\_Close}}{2}$$
$$\text{HA\_High} = \max(H, \text{HA\_Open}, \text{HA\_Close})$$
$$\text{HA\_Low} = \min(L, \text{HA\_Open}, \text{HA\_Close})$$
$$\text{ha\_color} = \begin{cases} 1 & \text{if HA\_Close} > \text{HA\_Open (bullish)} \\ 0 & \text{otherwise} \end{cases}$$

**Signal Logic (basic):**
- `positions = 1` when `ha_color == 1` (green)

**With MA Filter:**
- `positions = 1` when `ha_color == 1 AND Close > SMA(Close, ma_period)`

**With Confirmation:**
- Requires `X` consecutive candles of same color before signal triggers

**Sentiment:** `_sentiment_zero_positions` zeros positions when score < -0.7

#### Legacy Backtest (`heikin-ashi_bktest.py`)

| Parameter | Value |
|-----------|-------|
| `stls` (max long positions) | `3` |
| Data | yfinance NVDA 2015-04-01 to 2018-02-15, slicer=700 |

**Long Entry Trigger:**
- `HA_Open > HA_Close` (red candle) AND `HA_Open == HA_High` (no upper wick)
- Current body > previous body AND previous was also red

**Exit Trigger:**
- `HA_Open < HA_Close` (green) AND `HA_Open == HA_Low` (no lower wick) AND previous was green
- **Clears ALL positions at once**

**Statistics Calculated:**

| Metric | Formula |
|--------|---------|
| CAGR | $(V_{final}/V_{initial})^{252/n} - 1$ |
| Sharpe | $\bar{r} / \sigma_r$ |
| MDD | Running peak drawdown |
| Calmar | CAGR / MDD |
| Omega Ratio | $\frac{\int_0^{+\infty}(1-F(x))dx}{\int_{-\infty}^{0} F(x)dx}$ using Student-t CDF + `scipy.integrate.quad` |
| Sortino | $\bar{r} / \sigma_{\text{downside}}$ using Student-t CDF integration |

---

### 3.4 Parabolic SAR Strategy

**Files:** `parabolic_sar.py` (600 lines, BaseStrategy), `parabolic_sar_bktest.py` (standalone)

#### New Framework (`ParabolicSARStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.MOMENTUM` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Min | Max |
|-----------|---------|-----|-----|
| `af_start` | `0.02` | 0.01 | 0.1 |
| `af_increment` | `0.02` | 0.01 | 0.05 |
| `af_max` | `0.2` | 0.1 | 0.5 |

**Parabolic SAR Formula:**
$$\text{SAR}_{t+1} = \text{SAR}_t + \text{AF} \times (\text{EP} - \text{SAR}_t)$$

Where:
- **AF** (Acceleration Factor) starts at `0.02`, increments by `0.02` each new extreme, caps at `0.2`
- **EP** (Extreme Point) = highest high in uptrend, lowest low in downtrend

**Trend Reversal Rules:**

| Condition | Action |
|-----------|--------|
| Uptrend AND `Low[i] < PSAR[i]` | Switch to downtrend. PSAR = max(EP, High[i], High[i-1]). Reset AF. |
| Downtrend AND `High[i] > PSAR[i]` | Switch to uptrend. PSAR = min(EP, Low[i], Low[i-1]). Reset AF. |

**SAR Constraints:**
- In uptrend: SAR ≤ min(last two lows)
- In downtrend: SAR ≥ max(last two highs)

**Signal Logic:**
- `positions = 1` when `trend == 1` (uptrend), else `0`
- `signals = positions.diff()`

**Sentiment:** Scales the SAR distance values.

#### Legacy Backtest (`parabolic_sar_bktest.py`)

- `initial_af=0.02, step_af=0.02, end_af=0.2`
- Uses a trend counter (accumulates >1 for sustained trends)
- Data: yfinance EA 2016-01-01 to 2018-01-01, slicer=450

---

## 4. Pattern Recognition

### 4.1 Bollinger Band Pattern Strategy

**Files:** `bollinger_pattern.py` (~600 lines, BaseStrategy), `bollinger_band_bktest.py` (standalone)

#### New Framework (`BollingerPatternStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.PATTERN_RECOGNITION` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Min | Max |
|-----------|---------|-----|-----|
| `bb_period` | `20` | 5 | 100 |
| `bb_std` | `2.0` | 0.5 | 4.0 |
| `pattern_period` | `75` | 20 | 200 |
| `alpha` | `0.01` | — | — |

**Bollinger Band Formulas:**
$$\text{mid\_band} = \text{SMA}(\text{Close}, 20)$$
$$\text{upper\_band} = \text{mid} + 2\sigma$$
$$\text{lower\_band} = \text{mid} - 2\sigma$$
$$\text{bandwidth} = \frac{\text{upper} - \text{lower}}{\text{mid}}$$
$$\%B = \frac{\text{Close} - \text{lower}}{\text{upper} - \text{lower}}$$

**Bottom W Pattern Detection (5 nodes: I, J, K, L, M):**

Scans backward from current position over `pattern_period=75` days:

| Node | Condition | Description |
|------|-----------|-------------|
| **K** (cond 1) | `|Close - lower_band| < alpha * mid_band` | First bottom — price near lower band |
| **L** | Price above mid band | First peak (between K and I) |
| **J** (cond 2) | `|Close - mid_band| < alpha * mid_band` AND `|mid - current_upper| < alpha * mid_band` | Pullback to mid band |
| **M** (cond 3) | `Close > lower_band` AND `Close < threshold` AND `M_price > K_price` | Second bottom — higher than first (W shape) |
| **I** (cond 4) | `Close > upper_band` | Breakout above upper band → **BUY** |

**Exit Condition:**
$$\text{Exit when bandwidth} < \beta = \text{avg\_bandwidth} \times 0.8$$

Uses `_calculate_portfolio_long_only` from BaseStrategy.

#### Legacy Backtest (`bollinger_band_bktest.py`)

- Uses `alpha=5.0, beta=3.0` (absolute thresholds, not relative like new version)
- Processes `Close` as `'price'` column
- Data: yfinance AAPL 2020-01-01 to 2022-12-31

---

### 4.2 RSI Pattern Strategy

**Files:** `rsi_pattern.py` (~600 lines, BaseStrategy), `rsi_pattern_recognize_bktest.py` (standalone)

#### New Framework (`RSIPatternStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.PATTERN_RECOGNITION` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Min | Max |
|-----------|---------|-----|-----|
| `rsi_period` | `14` | 5 | 50 |
| `oversold_threshold` | `30` | 10 | 50 |
| `overbought_threshold` | `70` | 50 | 90 |

**RSI Formula:**
Uses `calculate_rsi` from `strategies.utils` (Wilder's EWM method, see Section 2.5).

**Signal Logic:**
$$\text{positions} = \begin{cases} 1 & \text{if RSI} < 30 \\ -1 & \text{if RSI} > 70 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{signals} = \text{positions.diff()}$$

Uses `_calculate_portfolio_long_only` from BaseStrategy.

**Sentiment:** Adjusts thresholds or filters positions via sentiment helpers.

#### Legacy Backtest (`rsi_pattern_recognize_bktest.py`)

**Two strategies implemented:**

**Strategy 1 — Simple Overbought/Oversold:**
- RSI via Smoothed Moving Average (SMMA) — authentic Wilder method
- Buy when RSI < 30, sell when RSI > 70

**Strategy 2 — Head-and-Shoulder Pattern on RSI:**

| Parameter | Value |
|-----------|-------|
| `period` | 25 |
| `delta` | 0.2 |
| `head` multiplier | 1.1 |
| `shoulder` multiplier | 1.1 |

**Pattern Nodes (7 points on RSI curve):**
- `i` → start point
- `j` → head (maximum RSI in window)
- `k` → trough between j and i
- `l` → left of j
- `m` → left of l
- `n` → shoulder (between m and l)
- `o` → right shoulder (between k and i)

**Exit Conditions:**
- `exit_rsi = 4` — exit when RSI increases by 4 points
- `exit_days = 5` — exit after maximum 5 holding days

Data: yfinance MSFT 2016-01-01 to 2018-01-01

---

### 4.3 Shooting Star Strategy

**Files:** `shooting_star.py` (602 lines, BaseStrategy), `shooting_star_bktest.py` (standalone)

#### New Framework (`ShootingStarStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.PATTERN_RECOGNITION` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lower_bound` | `0.2` | Max lower wick as fraction of body |
| `body_size` | `0.5` | Max body size relative to average body |
| `stop_threshold` | `0.05` | 5% stop loss/take profit trigger |
| `holding_period` | `7` | Max days before forced exit |

**8 Pattern Conditions (ALL must be true):**

| # | Condition | Formula |
|---|-----------|---------|
| c1 | Red candle | `Open >= Close` |
| c2 | Small lower wick | `lower_wick < 0.2 × body` |
| c3 | Small body | `body < avg_body × 0.5` |
| c4 | Long upper wick | `upper_wick >= 2 × body` |
| c5 | Previous day up | `Close >= prev_Close` |
| c6 | Two days ago up | `prev_Close >= prev_prev_Close` |
| c7 | Next day confirms (lower high) | `next_High <= current_High` |
| c8 | Next day confirms (lower close) | `next_Close <= current_Close` |

Where:
$$\text{body} = |O - C|, \quad \text{upper\_wick} = H - \max(O,C), \quad \text{lower\_wick} = \min(O,C) - L$$

**Entry:** **Short** when all 8 conditions met  
**Exit:** `|price_change| > 0.05` (5%) OR `counter >= 7` days

**Portfolio Logic:** Custom short position P&L tracking (not using BaseStrategy portfolio methods).

#### Legacy Backtest (`shooting_star_bktest.py`)

- Same 8 conditions
- Data: yfinance VOD.L (Vodafone) 2000-01-01 to 2021-11-04

---

### 4.4 Support & Resistance Strategy

**Files:** `support_resistance.py` (652 lines, BaseStrategy), `support_resistance_bktest.py` (standalone)

#### New Framework (`SupportResistanceStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.PATTERN_RECOGNITION` |
| `requires_sentiment` | `True` |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n1` | `2` | Candles before pivot for S/R detection |
| `n2` | `2` | Candles after pivot for S/R detection |
| `back_candles` | `30` | Lookback window for finding levels |
| `level_proximity` | `0.02` | 2% proximity threshold to S/R level |

**Support Detection Algorithm:**
- `n1` candles with **decreasing lows** before the pivot
- `n2` candles with **increasing lows** after the pivot

**Resistance Detection Algorithm:**
- `n1` candles with **increasing highs** before the pivot
- `n2` candles with **decreasing highs** after the pivot

**Candlestick Pattern Types:**

| Pattern | Type Code | Conditions |
|---------|-----------|------------|
| Bullish Engulfing | `2` | Current close > prev open, current open < prev close, green candle |
| Bearish Engulfing | `1` | Current close < prev open, current open > prev close, red candle |
| Shooting Star | `1` | Small body, lower wick < body, upper wick ≥ 2× body |
| Hammer | `2` | Small body, upper wick < body, lower wick ≥ 2× body |

**Signal Generation:**
$$\text{signal} = \begin{cases} \text{SELL} & \text{Bearish engulfing/star near resistance} \\ \text{BUY} & \text{Bullish engulfing/hammer near support} \end{cases}$$

**Proximity Check:**
$$|\text{price} - \text{closest\_level}| \leq \text{limit}$$

#### Legacy Backtest (`support_resistance_bktest.py`)

Uses **`backtesting.py` library** (not matplotlib-based):
- `cash = 10,000`
- `commission = 0.002` (0.2%)
- `exclusive_orders = True`
- **Stop Loss:** `sl = Close × 0.90` (10%)
- **Take Profit:** `tp = Close × 1.08` (8%)
- Data: yfinance RGTI 2021-01-01 to 2025-01-01

---

## 5. Statistical Arbitrage

### 5.1 Pairs Trading Strategy

**Files:** `pairs_trading.py` (~700 lines, BaseStrategy), `pairs_trading_bktest.py` (standalone)

#### New Framework (`PairsTradingStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.STATISTICAL_ARBITRAGE` |
| `requires_sentiment` | `False` |
| Tickers required | **Exactly 2** |

**Parameters:**

| Parameter | Default | Min | Max | Description |
|-----------|---------|-----|-----|-------------|
| `bandwidth` | `60` | 30 | 500 | Rolling window for Z-score |
| `z_entry` | `1.0` | 0.5 | 3.0 | Z-score threshold for entry |
| `z_exit` | `0.0` | — | — | Z-score threshold for exit |

**Cointegration Test:** Engle-Granger two-step method
1. OLS regression: `Y = β × X + ε`
2. ADF test on residuals `ε` (p ≤ 0.05 → cointegrated)
3. **Fallback:** Simple correlation > 0.7 with linear regression

**Spread & Z-Score:**
$$\text{spread} = Y - \beta \times X$$
$$Z = \frac{\text{spread} - \text{SMA}(\text{spread}, 60)}{\text{rolling\_std}(\text{spread}, 60)}$$

**Trading Signals:**

| Condition | Action |
|-----------|--------|
| $Z > z_{\text{entry}}$ (+1.0) | Short asset2, Long asset1 |
| $Z < -z_{\text{entry}}$ (-1.0) | Long asset2, Short asset1 |
| $|Z| < z_{\text{exit}}$ (0.0) | Close all positions |

**Portfolio:** Capital split 50/50. Returns = weighted combination of both asset returns.

#### Legacy Backtest (`pairs_trading_bktest.py`)

- `bandwidth = 250`
- Z thresholds: ±1 sigma
- `capital0 = 20,000`
- `shares = capital0 // max(price)` per asset
- Data: yfinance NVDA vs AMD 2013-01-01 to 2014-12-31

---

### 5.2 Mean Reversion Strategy

**Files:** `mean_reversion.py` (644 lines, BaseStrategy), `edge_mean_reversion.py` (~300 lines), `edge_risk_kit.py` (~300 lines)

#### New Framework (`MeanReversionStrategy`)

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.STATISTICAL_ARBITRAGE` |
| `requires_sentiment` | `False` |
| Tickers required | 2 or more |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | `30` | Rolling window (0 = auto from half-life) |
| `threshold` | `2.0` | Z-score entry threshold |
| `stoploss` | `0.05` | Stop-loss percentage per trade |

**Portfolio Construction:**

| # Assets | Method |
|----------|--------|
| 2 | OLS hedge ratio: `Y = β × X + ε`, portfolio = `Y - β × X` |
| 3+ | Johansen eigenvector weights: `portfolio = Σ(prices × coeff)` via `coint_johansen(prices, det_order=0, k_ar_diff=1)` |

**Statistical Tests Run (from `edge_mean_reversion`):**

| Test | Function | Threshold |
|------|----------|-----------|
| ADF Test | `adf(ts)` — manual ADF via `mackinnonp` | p ≤ 0.05 → stationary |
| Hurst Exponent | Lag variance method | H < 0.5 → mean reverting |
| Variance Ratio | `variance_ratio(ts, lag=2)` | p < 0.05 → not random walk |
| Half-Life | $-\frac{\ln(2)}{\beta_0}$ from OLS of diff vs lag | Used for lookback auto-calc |
| Cointegration | `coint(x, y)` from statsmodels | p ≤ 0.05 → cointegrated |

**Signal Generation:**

| Condition | Action |
|-----------|--------|
| $Z < -\text{threshold}$ | Buy (position = 1) |
| $Z > +\text{threshold}$ | Sell short (position = -1) |
| Position = 1 AND $Z > 0$ | Exit long |
| Position = -1 AND $Z < 0$ | Exit short |
| Per-trade loss > stoploss | Force exit |

**Risk Analytics (from `edge_risk_kit`):**

| Function | Formula |
|----------|---------|
| `drawdown(r)` | Wealth index from `(1+r).cumprod()`, tracks peaks and drawdowns |
| `annualize_rets(r, periods)` | $(1 + \bar{r})^{\text{periods}} - 1$ |
| `annualize_vol(r, periods)` | $\sigma_r \times \sqrt{\text{periods}}$ |
| `sharpe_ratio(r, rfr, periods)` | $\frac{r_a - r_f}{\sigma_a}$ |
| `summary_stats(r, rfr=0.03, periods=12)` | Annualized return, vol, Sharpe, skewness, kurtosis, max drawdown |

---

## 6. Crypto Strategies

### 6.1 Crypto Mean Reversion Strategy

**Files:** `crypto/mean_reversion_strategy.py` (1194 lines, BaseStrategy), `crypto/binance_data.py` (375 lines), `crypto/__init__.py`

#### `CryptoMeanReversionStrategy(BaseStrategy)`

| Attribute | Value |
|-----------|-------|
| Category | `StrategyCategory.MEAN_REVERSION` |
| `requires_sentiment` | `False` |
| `min_data_points` | `100` |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | `30` | Z-score rolling window |
| `threshold` | `2.0` | Z-score entry threshold |
| `stoploss` | `0.001` | Stop-loss per trade |
| `cash` | `10000` | Backtest initial cash |
| `commission` | `0.002` | Trading commission (0.2%) |
| `run_optimisation` | `True` | Whether to run parameter optimization |

**Data Source:** Binance REST API via `crypto/binance_data.py`

**Pipeline (full `run()` method):**

1. **Data Fetch:** `fetch_crypto_prices(tickers, interval="1d", start, end)` from Binance
2. **EDA:** Correlation matrices, price plots, distribution analysis
3. **Statistical Tests:** ADF, Hurst, Variance Ratio, Half-Life, Cointegration (same functions as mean_reversion.py)
4. **2-Asset Portfolio:** OLS hedge ratio → `spread = Y - β × X`
5. **3-Asset Portfolio:** Johansen eigenvector → `portfolio = Σ(prices × coeff)`
6. **Backtesting:** Uses `backtesting.py` library with `Z_Score_Naive` strategy class
7. **Optimization:** Grid search over parameters
8. **Persistence:** Saves to MinIO (charts, HTML) and PostgreSQL (metrics)

**Z_Score_Naive Strategy (for `backtesting.py`):**
```python
class Z_Score_Naive(Strategy):
    lookback = 30
    threshold = 2.0
    stoploss = 0.001
    
    def next(self):
        if z_score < -threshold and no position:
            self.buy()
        elif z_score > threshold and no position:
            self.sell()
        elif z_score > 0 and long:
            self.position.close()
        elif z_score < 0 and short:
            self.position.close()
        # Stop-loss check on open position
```

**Optimization Configurations:**

| Label | Maximize |
|-------|----------|
| Max Equity | `Equity Final [$]` |
| Min Drawdown | `Max. Drawdown [%]` |
| Min Volatility | `neg_Volatility` (negated annual volatility) |
| Max Sharpe | `Sharpe Ratio` |

**Optimization Parameter Grid:**
- `lookback`: `range(20, 40, 5)` → [20, 25, 30, 35]
- `threshold`: `np.arange(2, 5.5, 0.5)` → [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
- `stoploss`: `np.arange(0.001, 0.005, 0.001)` → [0.001, 0.002, 0.003, 0.004]

**Persistence:**
- **MinIO:** Chart images (PNG) + backtesting.py interactive HTML plots
- **PostgreSQL:** Backtest results with strategy_id "crypto_mean_reversion"

### 6.2 Binance Data Fetcher (`crypto/binance_data.py`, 375 lines)

| Function | Details |
|----------|---------|
| `fetch_klines(symbol, interval="1d", start, end)` | Paginated Binance `/api/v3/klines` endpoint. `MAX_LIMIT=1000` per request. Columns: `open_time, open, high, low, close, volume, ...` |
| `fetch_crypto_prices(symbols, interval, start, end, cache=True)` | Returns wide DataFrame with close prices. Appends `USDT` to symbol if needed. |

**CSV Caching:** Per-symbol CSV files in `trading_strategies/crypto/data/`. Incremental updates — only fetches missing date ranges.

---

## 7. FX Intraday

### 7.1 London Breakout Strategy

**File:** `fx_intraday/london_breakout_bktest.py` (standalone, ~200 lines)

**Concept:** Trade the London FX session opening based on Tokyo session range.

**Data:** 1-minute GBPUSD from CSV (histdata.com), Eastern Standard Time.

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `risky_stop` | `0.01` (100 bps) | Risk tolerance for false alarms |
| `open_minutes` | `30` | Signal window after London open |
| Stop Loss | `risky_stop / 2 = 50 bps` | Half of risk interval |
| Target | `risky_stop / 2 = 50 bps` | Half of risk interval |

**Threshold Calculation:**
- Collect all prices from EST 2:00 AM (last hour of Tokyo session)
- `upper = max(tokyo_prices)`, `lower = min(tokyo_prices)`

**Signal Rules (EST 3:00 - 3:30 AM):**

| Condition | Signal |
|-----------|--------|
| `price > upper` AND `price - upper < risky_stop` | Long (+1) |
| `price < lower` AND `lower - price < risky_stop` | Short (-1) |
| `price - upper > risky_stop` | False alarm (too volatile), skip |
| `lower - price > risky_stop` | False alarm, skip |

**Position Management:**
- `cumsum` ensures only **one signal per direction** per day
- Positions cleared when `|price - executed_price| > 50 bps`
- ALL positions cleared at EST 12:00 PM (London close) via `-cumsum`

---

### 7.2 Dual Thrust Strategy

**File:** `fx_intraday/dual_thrust_bktest.py` (standalone, ~190 lines)

**Concept:** Opening range breakout using previous days' high/low/close range.

**Data:** 1-minute GBPUSD from CSV, EST timezone.

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rg` | `5` | Days of lookback for range calculation |
| `param` | `0.5` | Trigger range parameter (0 to 1) |

**Range Calculation (frequency conversion, minute → daily):**

For each of the past `rg` days:
$$\text{range1} = \max(\text{High}, rg) - \min(\text{Close}, rg)$$
$$\text{range2} = \max(\text{Close}, rg) - \min(\text{Low}, rg)$$
$$\text{range} = \max(\text{range1}, \text{range2})$$

**Threshold Formulas (at EST 3:00 AM market open):**
$$\text{upper} = \text{param} \times \text{range} + \text{opening\_price}$$
$$\text{lower} = -(1 - \text{param}) \times \text{range} + \text{opening\_price}$$

With `param=0.5`: symmetric thresholds, 50/50 chance for long/short.

**Signal Generation:**
- `price > upper` → Long (+1)
- `price < lower` → Short (-1)
- Reversal: if position switches from short to long (or vice versa), signal = ±2
- **No stop loss** in this strategy
- All positions cleared at EST 12:00 PM via `-cumsum`

---

## 8. Derivatives / Options

### 8.1 Options Straddle Strategy

**File:** `derivatives/options_straddle_bktest.py` (standalone, ~300 lines)

**Concept:** Long straddle — buy call AND put at same strike price and expiration.

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `contractsize` | `1` | Per-share basis for AAPL |
| `threshold` | `10` | Max call-put price difference ($) for entry |

**Data Source:** yfinance `Ticker.option_chain()` for real-time options + `yf.download()` for spot.

**Strike Price Selection:**
- `find_strike_price(df)`: extracts integer strike prices available in BOTH calls and puts (via regex `\d{4}`)

**Signal Generation:**
$$\text{signal} = \begin{cases} 1 & \text{if } |\text{call\_price} - \text{put\_price}| < \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

**Profit Calculation:**
$$\text{profit} = |\text{spot}_{final} - \text{strike} \times \text{contractsize}| - \text{call\_premium} - \text{put\_premium}$$

**Breakeven Points:**
- Lower: `strike × contractsize - (call + put)`
- Upper: `strike × contractsize + (call + put)`

**Payoff Diagram:** Plots V-shaped long straddle P&L:
- Green (profit) when spot outside breakeven range
- Red (loss) when spot inside breakeven range

**Expiration Selection:** Finds expiry ≥ 30 days from today, picks from available `ticker.options`.

---

### 8.2 VIX Calculator

**File:** `derivatives/vix_calculator.py` (512 lines, standalone)

**Concept:** CBOE VIX methodology applied to any ticker (demo: AAPL).

**Reference:** CBOE White Paper (http://www.cboe.com/micro/vix/vixwhite.pdf)

**VIX Parameters:**

| Parameter | Value |
|-----------|-------|
| `timeframe_front` | 2 (months) |
| `timeframe_rear` | 3 (months) |
| `expiration_hour` | 16 |
| `expiration_day` | 9 (~3rd Friday) |
| `num_of_mins_year` | 365 × 24 × 60 = 525,600 |

**Key Functions:**

**1. Settlement Day Calculation (`get_settlement_day`):**
- Counts backward from month end
- Skips weekends and US federal holidays
- Finds the `expiration_day`-th last business day

**2. Time to Expiration:**
$$T = \frac{(\text{settlement\_day} - \text{current\_day}).\text{total\_seconds}()}{60 \times 525600}$$

**3. Forward Level & Strike (`get_forward_strike`):**
$$\text{forward} = K_{\min\_diff} + e^{rT} \times (C_{K} - P_{K})$$
Where $K_{\min\_diff}$ is the strike with minimum `|call - put|` difference.

Strike $K_0$ = largest strike ≤ forward.

**4. OTM Option Selection:**
- **Calls:** Strike > $K_0$, out-of-the-money
- **Puts:** Strike < $K_0$, out-of-the-money
- Excludes options after two consecutive zero-price strikes

**5. Sigma Calculation (variance swap formula):**
$$\sigma^2 = \frac{2}{T} \sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i) - \frac{1}{T}\left(\frac{F}{K_0} - 1\right)^2$$

Where:
- $\Delta K_i$ = interval between adjacent strikes (midpoint for interior strikes)
- $Q(K_i)$ = option prior settle price
- $F$ = forward level, $K_0$ = ATM strike

**6. VIX Final Calculation (weighted average):**
$$\text{VIX} = 100 \times \sqrt{\frac{N_T}{N_{30}} \left[\frac{T_1 \sigma_1^2 (N_{T_2} - N_{30})}{N_{T_2} - N_{T_1}} + \frac{T_2 \sigma_2^2 (N_{30} - N_{T_1})}{N_{T_2} - N_{T_1}}\right]}$$

Where $N$ = number of minutes for each term, 30 = target timeframe.

**Data Sources:**
- yfinance `Ticker.option_chain()` for options data
- yfinance `^IRX` (13-week T-bill rate) as proxy for risk-free rate
- `USFederalHolidayCalendar` from pandas for holiday skipping

---

## 9. Portfolio Analysis

### 9.1 Asset Allocation

**File:** `portfolio_analysis/asset_allocation.py` (standalone, ~200 lines)

**Tickers:** Quantum computing stocks — RGTI, QBTS, IONQ, NBIS  
**Data:** yfinance 2021-01-01 to 2025-01-01

**Analysis Components:**

**1. Buy & Hold Comparison:**
$$\text{return}_{\text{equal}} = \frac{1}{4}(r_{\text{RGTI}} + r_{\text{QBTS}} + r_{\text{IONQ}} + r_{\text{NBIS}})$$

**2. Rolling Returns:**
- `ROLL_WINDOW = min(21, len(data) - 1)` (adaptive)
- Rolling return: `Close / Close.shift(ROLL_WINDOW) - 1`

**3. Sharpe Ratio (rolling):**
$$\text{Sharpe} = \frac{\bar{r} - \text{MARR}}{\sigma_r}$$
Where MARR (Minimal Acceptable Rate of Return) = 0.

**4. Optimization via `scipy.optimize.minimize` (SLSQP):**

**Objective 1 — Max Sharpe:**
$$\max_{w} \frac{\bar{r}_w}{\sigma_{r_w}} \quad \text{s.t.} \quad \sum w_i \leq 1, \quad 0 \leq w_i \leq 1$$

**Objective 2 — Max Median Yearly Return:**
$$\max_{w} \text{median}(r_w) \quad \text{s.t.} \quad \sum w_i \leq 1, \quad 0 \leq w_i \leq 1$$

Initial guess: `(1, 0, 0, 0)`. Note: SLSQP easily gets stuck in local optima.

**5. Dynamic Asset Allocation Strategy:**
```
Default: 80% RGTI + 20% NBIS daily returns
When RGTI_rolling_return(shift 1) > -0.17: 100% RGTI
When NBIS_rolling_return(shift 1) > 0.29:  100% NBIS
```
Cumulative strategy returns plotted against buy-and-hold and optimized portfolios.

---

## 10. Risk Modelling

### 10.1 Monte Carlo Simulation

**File:** `risk_modelling/monte_carlo_bktest.py` (standalone, ~230 lines)

**Concept:** Geometric Brownian Motion simulation for stock price forecasting.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `testsize` | `0.5` | Train/test split ratio |
| `simulation` | `100` | Number of simulation paths |
| `simu_start` | `100` | Min simulations for accuracy test |
| `simu_end` | `1000` | Max simulations for accuracy test |
| `simu_delta` | `100` | Step size for simulation count test |

**Data:** yfinance GE 2016-01-15 to 2019-01-15

**GBM Formula:**
$$\text{drift} = \bar{r}_{\log} - \frac{\sigma^2_{\log}}{2}$$
$$\text{SDE}_t = \text{drift} + \sigma_{\log} \times \mathcal{N}(0, 1)$$
$$S_{t+1} = S_t \times e^{\text{SDE}_t}$$

Where:
$$r_{\log} = \ln\left(\frac{S_t}{S_{t-1}}\right)$$

**Best Fit Selection:**
$$\text{pick} = \arg\min_k \text{std}\left(d_k[:\text{train}] - S_{\text{actual}}[:\text{train}]\right)$$

The simulation with smallest standard deviation against training data is selected.

**Accuracy Test:**
- Runs simulations from 100 to 1000 (step 100)
- Compares direction prediction: `sign(actual_return) vs sign(predicted_return)`
- Plots horizontal bar chart showing that **accuracy does not improve with more simulations**

**Key Insight (documented in code):**
> "Monte Carlo simulation in trading is house of cards. It is merely illusion that monte carlo simulation can forecast any asset price or direction."

---

## 11. Shared Utilities

### 11.1 `trading_strategies/backtest_utils.py`

| Function | Parameters | Logic |
|----------|-----------|-------|
| `mdd(series)` | price series | Iterates comparing each point to prior peak. Returns max percentage drawdown. |
| `candlestick(df, ax, ...)` | DataFrame with OHLC, matplotlib axis | Custom candlestick chart using green/red `fill_between`. Plots as bars with wicks. |
| `portfolio(data, capital0=10000, positions=100)` | signals DataFrame, initial capital, share count | `holdings = positions * Close * shares`. `cash = capital0 - cumsum(signals * Close * shares)`. `total = holdings + cash`. |
| `profit(port)` | portfolio DataFrame | Plots equity curve with LONG (green ^) / SHORT (red v) markers. |

### 11.2 `statistical_arbitrage/edge_mean_reversion.py`

| Function | Formula/Logic |
|----------|---------------|
| `adf(ts)` | Manual ADF via lagged differences + `mackinnonp`. Returns (statistic, p-value). |
| `hurst(ts)` | $H = \frac{\log(\text{variance of } \tau\text{-lagged diffs})}{\log(2\tau)}$. H<0.5 = mean reverting, H=0.5 = random walk, H>0.5 = trending. |
| `variance_ratio(ts, lag=2)` | Uses `arch.unitroot.VarianceRatio` test. |
| `half_life(ts)` | OLS: `diff(ts) = β × lag(ts) + ε`. Half-life = $-\ln(2) / \beta_0$. |
| `perform_adf_test(ts, verbose=True)` | Wrapper. p ≤ 0.05 → stationary. |
| `perform_hurst_exp_test(ts)` | Wrapper. hurst < 0.5 → mean reverting. |
| `perform_variance_ratio_test(ts, lag=2)` | Wrapper. p < 0.05 → not random walk. |
| `perform_coint_test(x, y)` | `statsmodels.tsa.stattools.coint`. p ≤ 0.05 → cointegrated. |

### 11.3 `statistical_arbitrage/edge_risk_kit.py`

| Function | Formula |
|----------|---------|
| `drawdown(return_series)` | `wealth = 1000 × (1+r).cumprod()`. `previous_peaks = wealth.cummax()`. `drawdown = (wealth - peaks) / peaks` |
| `annualize_rets(r, periods)` | $(1 + \bar{r}_{\text{compound}})^{\text{periods}} - 1$ |
| `annualize_vol(r, periods)` | $\sigma \times \sqrt{\text{periods}}$ |
| `sharpe_ratio(r, rfr, periods)` | $(r_a - r_f) / \sigma_a$ |
| `summary_stats(r, rfr=0.03, periods=12)` | Returns DataFrame with annualized return, vol, Sharpe, skewness, kurtosis, max drawdown |

---

## 12. Strategy Registry & Loader

### 12.1 Trading Strategies Registry (`trading_strategies/__init__.py`, 126 lines)

**Lazy Import Map (`_LAZY_IMPORTS`):**

| Class Name | Module Path |
|------------|-------------|
| `MACDOscillatorStrategy` | `.momentum_trading.macd_oscillator` |
| `AwesomeOscillatorStrategy` | `.momentum_trading.awesome_oscillator` |
| `HeikinAshiStrategy` | `.momentum_trading.heikin_ashi` |
| `ParabolicSARStrategy` | `.momentum_trading.parabolic_sar` |
| `RSIPatternStrategy` | `.pattern_recognition.rsi_pattern` |
| `ShootingStarStrategy` | `.pattern_recognition.shooting_star` |
| `SupportResistanceStrategy` | `.pattern_recognition.support_resistance` |
| `BollingerPatternStrategy` | `.pattern_recognition.bollinger_pattern` |
| `PairsTradingStrategy` | `.statistical_arbitrage.pairs_trading` |
| `MeanReversionStrategy` | `.statistical_arbitrage.mean_reversion` |
| `CryptoMeanReversionStrategy` | `.crypto.mean_reversion_strategy` |

**Strategy Keys (`_STRATEGY_KEYS`):**

| Key | Class |
|-----|-------|
| `macd` | `MACDOscillatorStrategy` |
| `awesome_oscillator` | `AwesomeOscillatorStrategy` |
| `heikin_ashi` | `HeikinAshiStrategy` |
| `parabolic_sar` | `ParabolicSARStrategy` |
| `rsi_pattern` | `RSIPatternStrategy` |
| `shooting_star` | `ShootingStarStrategy` |
| `support_resistance` | `SupportResistanceStrategy` |
| `bollinger_pattern` | `BollingerPatternStrategy` |
| `pairs_trading` | `PairsTradingStrategy` |
| `mean_reversion` | `MeanReversionStrategy` |
| `crypto_mean_reversion` | `CryptoMeanReversionStrategy` |

**Strategy Metadata (`_STRATEGY_META`):** Each entry includes `id`, `name`, `description`, `category`, `requires_sentiment`, `min_tickers`. Available without importing strategy modules.

### 12.2 Framework Loader (`strategies/loader.py`)

**Known Module Map (`STRATEGY_MODULE_MAP`):**

| Key | Module |
|-----|--------|
| `macd_oscillator` | `trading_strategies.momentum_trading.macd_oscillator` |
| `awesome_oscillator` | `trading_strategies.momentum_trading.awesome_oscillator` |
| `heikin_ashi` | `trading_strategies.momentum_trading.heikin_ashi` |
| `parabolic_sar` | `trading_strategies.momentum_trading.parabolic_sar` |
| `rsi_pattern` | `trading_strategies.pattern_recognition.rsi_pattern` |
| `shooting_star` | `trading_strategies.pattern_recognition.shooting_star` |
| `support_resistance` | `trading_strategies.pattern_recognition.support_resistance` |
| `pairs_trading` | `trading_strategies.statistical_arbitrage.pairs_trading` |

---

## Appendix: Complete Parameter Reference

### All BaseStrategy Parameters at a Glance

| Strategy | Parameter | Default | Range |
|----------|-----------|---------|-------|
| **Awesome Oscillator** | `ao_short` | 5 | 2–20 |
| | `ao_long` | 34 | 20–100 |
| **MACD Oscillator** | `ma_short` | 10 | 2–50 |
| | `ma_long` | 21 | 10–200 |
| | `use_ema` | True | — |
| **Heikin-Ashi** | `confirmation_candles` | 1 | 1–5 |
| | `use_ma_filter` | False | — |
| | `ma_period` | 20 | 5–100 |
| **Parabolic SAR** | `af_start` | 0.02 | 0.01–0.1 |
| | `af_increment` | 0.02 | 0.01–0.05 |
| | `af_max` | 0.2 | 0.1–0.5 |
| **Bollinger Pattern** | `bb_period` | 20 | 5–100 |
| | `bb_std` | 2.0 | 0.5–4.0 |
| | `pattern_period` | 75 | 20–200 |
| | `alpha` | 0.01 | — |
| **RSI Pattern** | `rsi_period` | 14 | 5–50 |
| | `oversold_threshold` | 30 | 10–50 |
| | `overbought_threshold` | 70 | 50–90 |
| **Shooting Star** | `lower_bound` | 0.2 | — |
| | `body_size` | 0.5 | — |
| | `stop_threshold` | 0.05 | — |
| | `holding_period` | 7 | — |
| **Support & Resistance** | `n1` | 2 | — |
| | `n2` | 2 | — |
| | `back_candles` | 30 | — |
| | `level_proximity` | 0.02 | — |
| **Pairs Trading** | `bandwidth` | 60 | 30–500 |
| | `z_entry` | 1.0 | 0.5–3.0 |
| | `z_exit` | 0.0 | — |
| **Mean Reversion** | `lookback` | 30 | — |
| | `threshold` | 2.0 | — |
| | `stoploss` | 0.05 | — |
| **Crypto Mean Reversion** | `lookback` | 30 | — |
| | `threshold` | 2.0 | — |
| | `stoploss` | 0.001 | — |
| | `cash` | 10000 | — |
| | `commission` | 0.002 | — |
| | `run_optimisation` | True | — |

### RiskParams Defaults (BaseStrategy)

| Parameter | Default |
|-----------|---------|
| `stop_loss_pct` | 0.05 (5%) |
| `take_profit_pct` | 0.10 (10%) |
| `max_position_size` | 0.25 (25% of capital) |
| `max_drawdown_pct` | 0.20 (20%) |
| `trailing_stop` | False |
| `position_sizing` | "fixed" |

---

*End of Report — 48 files analyzed across 10 directories.*
