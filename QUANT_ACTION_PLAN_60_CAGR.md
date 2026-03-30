# CENTURION CAPITAL — 60% CAGR ACTION PLAN
## Quantitative Trading System Audit & Implementation Roadmap
### March 28, 2026 | Renaissance-Grade Assessment

---

## EXECUTIVE SUMMARY

**Goal:** 60% annual CAGR on equity investments (IND primary, US secondary)

**Current System State:**
- Overall score: **7/10** (post-March 2026 fixes)
- Estimated current CAGR: **12–20%** (Carver framework partially wired, legacy Kelly still dominant for IND)
- Sharpe ratio: **0.25–0.40** (too low for 60% CAGR without leverage)
- 42 strategies built, 20+ backtested, but **ensemble not optimized**
- Carver systematic framework **built but not wired for IND execution**
- Monte Carlo exists but **dismissed as unreliable** (needs proper implementation)
- RL bot exists but **disabled by default** (RL_ENABLED=False)

**What 60% CAGR requires (math):**
- At 20% annual volatility (current target, no leverage): **Sharpe ≥ 3.0** — unrealistic for equities
- At 40% annual volatility (2× leverage): **Sharpe ≥ 1.5** — very hard, hedge-fund grade
- At 60% annual volatility (3× leverage): **Sharpe ≥ 1.0** — achievable with concentrated momentum + options overlay
- **Realistic path:** Concentrated momentum (8-12 stocks) + F&O overlay + sector rotation + aggressive risk budgeting → **35-60% CAGR at Sharpe 0.8-1.2 with 40-60% vol**

**Key Insight from RenTech:**
RenTech's Medallion fund achieves ~66% gross returns through:
1. **Thousands of small, uncorrelated bets** (not 6-10 stocks)
2. **High-frequency statistical edges** (mean-reversion at tick level)
3. **Extreme diversification across instruments** (equities, futures, FX, commodities)
4. **Leverage** (typically 7-12× on low-vol strategies)
5. **Transaction cost minimization** (proprietary execution)

For a retail trader with Kite (no HFT, no futures/options automation yet), the realistic path is:
**Concentrated momentum + systematic swing trading + F&O income overlay → 35-60% CAGR**

---

## PART 1: CRITICAL GAPS & VULNERABILITIES

### 1.1 STRATEGY-LEVEL GAPS

| # | Gap | Severity | Current State | Impact on CAGR |
|---|-----|----------|---------------|----------------|
| S1 | **Carver pipeline not wired for IND execution** | CRITICAL | US=Carver, IND=legacy Kelly | Lose 8-15% annual alpha |
| S2 | **No momentum factor tilting** | CRITICAL | Equal-weight strategies; no momentum factor ranking | Miss top-decile momentum premium (~12% annually on NSE) |
| S3 | **No options income overlay** | HIGH | Option chain fetched but no systematic selling strategy | Miss 15-25% annual premium from covered calls/puts |
| S4 | **Forecast scalars not calibrated** | HIGH | Hardcoded since implementation | Stale scalars → forecast inflation → oversized positions |
| S5 | **No intraday mean-reversion for liquid stocks** | MEDIUM | Only swing/positional timeframe | Miss 5-8% from intraday reversion on NIFTY50 |
| S6 | **RL bot disabled** | MEDIUM | RL_ENABLED=False, 500K steps may be insufficient | Miss 3-5% adaptive alpha |
| S7 | **No earnings drift strategy** | MEDIUM | Earnings blackout only blocks, doesn't exploit | Miss post-earnings drift (PEAD) — 3-5% per event |
| S8 | **Sector rotation uses only 1-month momentum** | LOW | Single lookback period | Sub-optimal timing, whipsaw |

### 1.2 RISK MANAGEMENT GAPS

| # | Gap | Severity | Current State | Impact |
|---|-----|----------|---------------|--------|
| R1 | **No portfolio-level correlation risk** | CRITICAL | Each position sized independently | Correlated drawdowns amplified 2-3× |
| R2 | **Capital rolling not live** | CRITICAL | VolatilityTarget defined but P&L never updates capital | Position sizes don't adapt to wins/losses |
| R3 | **Portfolio drawdown halt at 50%** | HIGH | Way too late — account halved before protection kicks in | Should be 15% halt, 25% max |
| R4 | **No dynamic Kelly scaling** | HIGH | Fixed half-Kelly (0.5×) | Over-bets on low-conviction, under-bets on high-conviction |
| R5 | **Cost speed limit never called** | HIGH | Filter defined but not invoked | Low-edge trades executed, eating returns |
| R6 | **No tail-risk hedging** | HIGH | No portfolio puts, no VIX-based hedging | Vulnerable to 20%+ crashes |
| R7 | **Trailing SL hardcoded 5%/3%** | MEDIUM | Ignores ATR/volatility | Gets stopped out in volatile names, too loose in calm ones |
| R8 | **No realized slippage tracking** | MEDIUM | Expected vs actual fill not compared | Can't measure real execution quality |

### 1.3 EVALUATION & BACKTESTING GAPS

| # | Gap | Severity | Current State | Impact |
|---|-----|----------|---------------|--------|
| E1 | **No Monte Carlo risk estimation** | CRITICAL | Existing MC dismissed as broken (GBM only) | Can't estimate confidence intervals, tail risk, ruin probability |
| E2 | **No regime-conditional performance** | HIGH | Sharpe computed globally, not per regime | Can't identify strategy weakness in bear/choppy markets |
| E3 | **No transaction costs in walk-forward** | HIGH | OOS results overly optimistic | Overfit to gross returns; live returns 3-5% worse |
| E4 | **Missing risk metrics: Sortino, Calmar, Omega, CVaR** | HIGH | Only Sharpe, Drawdown, Win Rate | Incomplete risk picture |
| E5 | **No strategy decay detection** | MEDIUM | Walk-forward runs but degradation not alerted | Dead strategies keep trading |
| E6 | **No parameter stability analysis** | MEDIUM | Best params saved but drift not tracked | Unstable parameters = overfit |
| E7 | **No multi-asset correlation backtest** | MEDIUM | No portfolio-level attribution | Can't isolate which alpha source decayed |

### 1.4 EXECUTION & ARCHITECTURE GAPS

| # | Gap | Severity | Current State | Impact |
|---|-----|----------|---------------|--------|
| X1 | **No order amendment** | HIGH | Can't modify live orders (only cancel + re-place) | Slippage on SL adjustments |
| X2 | **No partial fill handling** | HIGH | Assumes all-or-nothing | Illiquid stocks may partially fill |
| X3 | **No multi-leg orders** | HIGH | Single-leg only | Can't do options spreads |
| X4 | **Pipeline not incremental** | MEDIUM | Full recalculation every cycle | 25-40s latency per cycle |
| X5 | **No live P&L dashboard alerting** | MEDIUM | Logs but doesn't alert | Miss drawdown signals |

---

## PART 2: OPPORTUNITIES FOR 60% CAGR

### 2.1 THE ALPHA STACK (Layered Sources)

To reach 60% CAGR, you need **multiple uncorrelated alpha sources** stacked:

| Alpha Source | Expected Contribution | Sharpe Add | Implementation Status |
|-------------|----------------------|------------|----------------------|
| **Momentum Factor (Top-decile NSE)** | +12-18% | +0.25 | NOT IMPLEMENTED |
| **Systematic Swing (Carver IND)** | +8-15% | +0.20 | 70% built, not wired |
| **Options Premium Selling** | +15-25% | +0.30 | Chain built, strategy missing |
| **Post-Earnings Drift (PEAD)** | +3-5% per event | +0.10 | NOT IMPLEMENTED |
| **Sector Rotation (Multi-TF)** | +5-8% | +0.10 | Partial (1M only) |
| **Mean-Reversion (Intraday NIFTY50)** | +5-8% | +0.15 | NOT IMPLEMENTED |
| **RL Adaptive Layer** | +3-5% | +0.08 | Built but disabled |
| **Delivery Volume Smart Money** | +2-3% | +0.05 | Implemented |
| **TOTAL STACK** | **+53-87%** | **+1.23** | |

**After costs/slippage/taxes (30-40% haircut):** **35-60% net CAGR** — achievable.

### 2.2 BEST QUANTITATIVE STRATEGIES FROM LITERATURE

#### A. Cross-Sectional Momentum (Jegadeesh & Titman, 1993; Asness et al., 2013)
- **What:** Rank all NIFTY 200 stocks by 12-1 month return, buy top decile, short bottom decile
- **Long-only variant:** Buy top-20 momentum stocks, rebalance monthly
- **NSE evidence:** 12-18% annual premium documented in Indian markets (Agarwalla et al., 2013)
- **Key parameters:** Formation period=12M (skip 1M), holding period=1M, rebalance=monthly
- **Risk:** Momentum crashes (2008-09 style) — need crash protection

#### B. Overnight Returns Anomaly (Lou et al., 2019; Berkman et al., 2012)
- **What:** Stocks with high overnight returns tend to underperform intraday
- **Implementation:** Buy at close, sell at open (or vice versa)
- **NSE evidence:** Opening price momentum documented
- **Integration:** Can overlay on existing swing positions

#### C. Short-Term Mean Reversion (Poterba & Summers, 1988; Lo & MacKinlay, 1990)
- **What:** 1-5 day mean reversion in liquid stocks (NIFTY50)
- **Statistics:** Hurst exponent < 0.5, half-life < 5 days
- **Implementation:** Z-score of 5-day returns, buy < -2σ, sell > +2σ
- **Your system:** Mean reversion strategy exists but not connected to live execution

#### D. Covered Call Writing / Cash-Secured Put Selling (CBOE BuyWrite Index)
- **What:** Sell OTM calls on long positions; sell OTM puts on desired entries
- **Evidence:** BXM (S&P BuyWrite) outperforms S&P 500 on risk-adjusted basis by ~2% annually
- **NSE variant:** NIFTY weekly options (every Thursday expiry) — very liquid
- **Expected yield:** 1-3% per month (12-36% annually) from premium alone
- **Integration:** Your option chain code can discover expiries/strikes — need systematic strategy logic

#### E. Factor Momentum (Arnott et al., 2021; Ehsani & Linnainmaa, 2022)
- **What:** Momentum applied to factors themselves — overweight factors that performed well recently
- **Implementation:** Track value/momentum/quality/size factors monthly, tilt towards recent winners
- **Integration:** Adjust forecast combiner weights dynamically

#### F. Regime-Conditional Strategy Switching (Ang & Timmermann, 2012)
- **What:** Different strategies optimal in different regimes
- **Trending Market:** Momentum + EWMAC dominant
- **Range-Bound:** Mean-reversion + options selling dominant
- **Crisis:** Cash + put protection dominant
- **Your system:** Regime detector exists but doesn't switch strategy mix

#### G. Post-Earnings Announcement Drift — PEAD (Bernard & Thomas, 1989)
- **What:** Stocks with positive earnings surprises drift up 2-5% over next 60 days
- **NSE evidence:** Documented in Indian markets (Bharath et al., 2009)
- **Implementation:** Monitor quarterly results, BUY on positive surprise, ride drift
- **Your system:** `earnings_momentum.py` exists but only gives +0.12 boost for 5 days (too weak, too short)

#### H. Volatility Risk Premium Harvesting (Carr & Wu, 2009)
- **What:** Implied volatility consistently exceeds realized volatility
- **Implementation:** Sell options (straddles/strangles), hedge with delta
- **India VIX:** Average premium of 3-5% over realized (collectible systematically)
- **Your system:** VIX computed, option chain available — strategy logic missing

### 2.3 ADVANCED EVALUATION TECHNIQUES

#### A. Monte Carlo Simulation — PROPERLY DONE

Your existing `monte_carlo_bktest.py` uses GBM (Geometric Brownian Motion) which assumes:
- Normal distribution of returns (WRONG — returns have fat tails)
- Constant volatility (WRONG — volatility clusters)
- No serial correlation (WRONG — momentum exists)

**Proper Monte Carlo for trading systems:**

1. **Trade-Level Bootstrap Monte Carlo** (best for your system):
   ```
   Input: Your actual trade returns from backtest
   Method: Randomly reshuffle trade sequence 10,000 times
   Output: Distribution of equity curves, confidence intervals
   Key Metrics:
   - P(ruin) at various drawdown levels
   - 5th/25th/50th/75th/95th percentile CAGR
   - Expected max drawdown at 95% confidence
   - Kelly-optimal position size (from actual trade distribution)
   ```

2. **Block Bootstrap** (preserves serial correlation):
   ```
   Method: Resample blocks of 5-20 consecutive trades (not individual trades)
   Reason: Preserves winning/losing streaks, regime clustering
   Output: More conservative (realistic) confidence intervals
   ```

3. **Regime-Conditional Monte Carlo**:
   ```
   Method: Separate trade pools by regime (bull/bear/range/crisis)
   Simulate: Draw from regime-appropriate pool based on current regime
   Output: Regime-aware risk estimates
   ```

#### B. Conditional Value-at-Risk (CVaR / Expected Shortfall)

- **What:** Average loss in the worst X% of scenarios (typically 5%)
- **Why better than VaR:** Measures how bad the BAD scenarios actually are
- **Formula:** CVaR_α = E[Loss | Loss > VaR_α]
- **Implementation:** From Monte Carlo output, take mean of bottom 5% of simulated returns
- **Action:** Use CVaR to set position limits (don't take trades where CVaR > 3× risk budget)

#### C. Hurst Exponent & Half-Life (Mean Reversion Detection)

- **What:** H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending
- **Half-life:** Expected time for mean-reversion to occur (in bars)
- **Your system:** `edge_mean_reversion.py` computes these but results aren't used in live scoring
- **Action:** For each stock, compute Hurst; route H < 0.45 stocks to mean-reversion strategy, H > 0.55 to momentum

#### D. Combinatorial Symmetric Cross-Validation (CSCV) — White et al., 2000

- **What:** All possible train/test splits (not just sequential)
- **Why:** Detects overfitting that walk-forward misses (walk-forward has look-ahead bias in window selection)
- **Your system:** Referenced in `integrated_scorer.py` but implementation depth unclear
- **Action:** Run CSCV on all 20+ strategies, discard any with PBO (Probability of Backtest Overfitting) > 50%

#### E. Maximum Diversification Ratio (Choueifaty & Coignard, 2008)

- **What:** Portfolio that maximizes: DR = (Σ w_i × σ_i) / σ_portfolio
- **Why:** Diversification ratio > 1 means you're getting "free" risk reduction
- **Your system:** IDM approximates this but doesn't optimize it
- **Action:** Replace handcrafted IDM with MDR optimization

#### F. Strategy Decay Half-Life Monitoring

- **What:** Track Sharpe ratio with exponential weighting; alert when recent Sharpe < 50% of historical
- **Implementation:** Expanding-window Sharpe with 63-day half-life weighting
- **Action:** Auto-reduce allocation to strategies whose recent Sharpe has decayed > 50%

---

## PART 3: IMPLEMENTATION ACTION PLAN

### PHASE 0: FOUNDATION FIXES (Week 1) — Close Critical Gaps

**Priority: Fix what's broken before building new**

#### 0.1 Wire Carver Pipeline to IND Execution

**File:** `kite_connect/trading/auto_executor.py`

**Current:** IND path uses legacy Kelly sizing via RiskManager
**Target:** IND path uses full Carver pipeline (vol-target → forecast → combine → size → risk → execute)

**Changes:**
1. In `auto_executor.py`, replace the legacy screen → risk → execute path with:
   ```python
   # BEFORE (legacy):
   screened = screener.screen(universe)
   plans = risk_mgr.generate_trade_plans(screened)
   
   # AFTER (Carver):
   from services.carver_pipeline import run_carver_pipeline
   result = run_carver_pipeline(
       tickers=universe,
       capital=config.CARVER_INITIAL_CAPITAL,
       mode="ind",
       kite=self._kite
   )
   plans = result.trade_plans  # Already vol-sized, cost-filtered
   ```
2. Ensure `carver_pipeline.py` reads IND-specific config (INR capital, NSE costs)
3. Add fallback: if Carver pipeline fails, fall back to legacy (don't break production)
4. Test with paper trader first (CARVER_ENABLED=True, paper mode)

**Effort:** 1-2 days | **Impact:** +8-15% annual alpha

#### 0.2 Live Capital Rolling

**File:** `services/volatility_target.py`, `kite_connect/trading/auto_executor.py`

**Current:** `VolatilityTarget` exists but `cum_pnl` never updated from real trades
**Target:** Real P&L from Kite portfolio updates capital daily

**Changes:**
1. In `auto_executor.py`, after each execution cycle:
   ```python
   realized_pnl = sum(closed_trade.pnl for closed_trade in trade_monitor.closed_trades())
   unrealized_pnl = sum(pos.unrealized_pnl for pos in kite.positions()["net"])
   vol_target.update_capital(realized_pnl, unrealized_pnl)
   ```
2. Persist capital state to `data/portfolio_state.json` (crash recovery)
3. Add daily capital snapshot to database (TimescaleDB) for analytics

**Effort:** 1 day | **Impact:** Position sizes adapt to actual P&L

#### 0.3 Activate Cost Speed Limit

**File:** `services/carver_pipeline.py`, `services/cost_speed_limit.py`

**Current:** `filter_by_cost()` exists but never invoked in the pipeline
**Target:** Step 5 of Carver pipeline calls cost filter

**Changes:**
1. In `carver_pipeline.py`, after forecast combination and before position sizing:
   ```python
   from services.cost_speed_limit import filter_by_cost
   filtered_forecasts = filter_by_cost(combined_forecasts, cost_model)
   ```
2. Log which stocks were filtered and their SR vs cost ratio

**Effort:** 0.5 days | **Impact:** Removes negative-edge trades

#### 0.4 Portfolio Drawdown Halt → 15%

**File:** `services/volatility_target.py`, `services/portfolio_vol_monitor.py`

**Current:** Halt at 50% drawdown (way too late)
**Target:** WARNING at 10%, CRITICAL at 15%, HALT at 20%

**Changes:**
```python
# volatility_target.py
capital_halt_fraction = 0.20  # Was 0.50
capital_warning_fraction = 0.10
capital_critical_fraction = 0.15

# portfolio_vol_monitor.py — wire into execution gate
RISK_LEVELS = {
    "NORMAL": {"drawdown_pct": 0.0, "scale": 1.0},
    "WARNING": {"drawdown_pct": 0.10, "scale": 0.5},  # Half-size
    "CRITICAL": {"drawdown_pct": 0.15, "scale": 0.25},  # Quarter-size
    "HALTED": {"drawdown_pct": 0.20, "scale": 0.0},  # No new trades
}
```

**Effort:** 0.5 days | **Impact:** Preserve capital during drawdowns

#### 0.5 Calibrate Forecast Scalars from Data

**File:** `services/forecast_scalar.py`, `services/carver_calibration.py`

**Current:** Hardcoded scalars (screener=0.20, DE=20.0, carry=40.0, EWMAC from table)
**Target:** Scalars computed from expanding-window backtest (Carver Chapter 7)

**Changes:**
1. In `carver_calibration.py`, add `calibrate_scalars()`:
   ```python
   def calibrate_scalars(raw_forecasts: pd.Series) -> float:
       """Carver method: scalar = 10 / mean(|raw_forecast|)"""
       abs_mean = raw_forecasts.abs().mean()
       if abs_mean < 1e-6:
           return 1.0
       return min(10.0 / abs_mean, 50.0)  # Cap at 50 to avoid explosion
   ```
2. Run on last 504 days of data (2 years)
3. Schedule weekly recalibration in `scheduler.py`
4. Log scalar drift (alert if >20% change)

**Effort:** 1 day | **Impact:** Correct forecast scaling → correct position sizing

---

### PHASE 1: MOMENTUM & FACTOR ENGINE (Week 2-3) — Primary Alpha Source

#### 1.1 Cross-Sectional Momentum Factor (NEW)

**New File:** `services/momentum_factor.py`

**Purpose:** Rank all NIFTY 200 stocks by 12-1 month return, select top decile

**Implementation:**
```python
class MomentumFactor:
    def __init__(self, formation_period=252, skip_period=21, n_stocks=20):
        self.formation = formation_period  # 12 months
        self.skip = skip_period            # Skip most recent month (reversal)
        self.n_stocks = n_stocks           # Top 20

    def rank(self, universe: List[str]) -> pd.DataFrame:
        """
        1. Fetch 13-month OHLCV for all stocks
        2. Compute 12-1 month return: price[t-21] / price[t-252] - 1
        3. Rank descending
        4. Return top N with momentum score
        """
        returns = {}
        for ticker in universe:
            data = yf.download(ticker, period="13mo")
            if len(data) >= 252:
                ret = data['Close'].iloc[-self.skip] / data['Close'].iloc[-self.formation] - 1
                returns[ticker] = ret

        ranked = pd.Series(returns).sort_values(ascending=False)
        top = ranked.head(self.n_stocks)
        return pd.DataFrame({
            'ticker': top.index,
            'momentum_return': top.values,
            'momentum_rank': range(1, len(top) + 1)
        })
```

**Integration:** Feed into Carver pipeline as additional alpha source (weight: 25%)

**Expected Impact:** +12-18% annual return from momentum premium

#### 1.2 Multi-Timeframe Sector Rotation (ENHANCED)

**File:** `services/sector_rotation.py`

**Current:** Only 1-month momentum
**Target:** Dual-timeframe: 1M + 3M with regime awareness

**Changes:**
```python
class EnhancedSectorRotation:
    def rank_sectors(self):
        # Short-term: 1-month momentum (existing)
        # Medium-term: 3-month momentum (new)
        # Combined: 0.6 × 1M + 0.4 × 3M
        # Regime filter: In bear regime, only DEFENSIVE sectors (FMCG, Pharma, IT)
        # In bull regime, overweight CYCLICAL sectors (Banks, Auto, Metals)
        pass
```

**Effort:** 1 day | **Impact:** +2-3% from better sector timing

#### 1.3 Post-Earnings Announcement Drift — PEAD (NEW)

**New File:** `services/pead_strategy.py`

**Purpose:** Exploit systematic drift after earnings surprises

**Implementation:**
```python
class PEADStrategy:
    """
    Research: Bernard & Thomas 1989, Ball & Brown 1968
    Indian market evidence: Bharath et al. 2009

    Logic:
    1. Monitor quarterly earnings announcements (from Trendlyne/NSE)
    2. Compute Standardized Unexpected Earnings (SUE):
       SUE = (EPS_actual - EPS_expected) / std(EPS_surprise_history)
    3. If SUE > 1.0: BUY signal (positive surprise, stock will drift up)
    4. Hold for 30-60 trading days (drift window)
    5. Exit when SUE signal decays or 60 days elapsed
    """

    def __init__(self):
        self.sue_threshold = 1.0       # Standard deviations
        self.hold_period = 45          # Trading days
        self.decay_factor = 0.95       # Daily signal decay

    def detect_surprise(self, ticker: str) -> Optional[float]:
        """Fetch latest earnings, compare vs consensus"""
        # Use Trendlyne API or NSE results page
        # Compute SUE
        pass

    def generate_signal(self, ticker: str, sue: float) -> float:
        """Convert SUE to Carver-scale forecast (-20 to +20)"""
        # Scalar: map SUE range [-3, +3] to forecast [-20, +20]
        forecast = min(max(sue * 6.67, -20), 20)
        return forecast
```

**Integration:** 
- Add as 5th Carver rule alongside EWMAC, Carry, Screener, DecisionEngine
- Weight: 15% of combined forecast
- Only active during earnings season (4 windows of ~3 weeks each)

**Expected Impact:** +3-5% from PEAD × 4 seasons × 10 relevant events

#### 1.4 Factor Momentum (Dynamic Strategy Weighting)

**New File:** `services/factor_momentum.py`

**Purpose:** Overweight recently-performing factors/strategies

**Implementation:**
```python
class FactorMomentum:
    """
    Research: Arnott et al. 2021, Ehsani & Linnainmaa 2022
    
    Instead of fixed Carver weights (EWMAC=22%, carry=22%, screener=17%, DE=22%),
    dynamically tilt towards strategies that performed well recently.
    """
    
    def compute_strategy_momentum(self, strategy_returns: Dict[str, pd.Series],
                                    lookback: int = 63) -> Dict[str, float]:
        """
        For each strategy, compute 3-month risk-adjusted return.
        Reweight proportional to recent Sharpe.
        """
        weights = {}
        for name, returns in strategy_returns.items():
            recent = returns.tail(lookback)
            sharpe = recent.mean() / (recent.std() + 1e-6) * 16
            weights[name] = max(sharpe, 0.05)  # Floor at 5%
        
        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
```

**Integration:** Replace fixed weights in `forecast_combiner.py` with dynamic weights

**Effort:** 2 days | **Impact:** +2-4% from adapting to what's working

---

### PHASE 2: OPTIONS INCOME OVERLAY (Week 3-4) — Major Alpha Source

#### 2.1 Systematic Covered Call Writing (NEW)

**New File:** `kite_connect/options/covered_call_strategy.py`

**Purpose:** Generate 1-3% monthly income on long equity positions

**Implementation:**
```python
class CoveredCallStrategy:
    """
    Research: CBOE BuyWrite Index (BXM), Feldman & Roy 2005
    
    For each long position in portfolio:
    1. Sell 1 OTM call (delta ~0.25-0.30) expiring weekly
    2. Strike = Current Price × (1 + 2σ_weekly)
    3. Premium target: 0.5-1.5% of position value per week
    4. Roll: If stock > 90% of strike, roll up and out
    5. Assignment risk: Accept assignment (take profit) or roll
    
    NSE specifics:
    - Weekly options available on NIFTY, BANKNIFTY, select stocks
    - Thursday expiry
    - Lot size awareness required
    """
    
    def __init__(self, kite, option_chain_service):
        self.kite = kite
        self.oc = option_chain_service
        self.delta_target = 0.25  # OTM enough to avoid assignment
        self.min_premium_pct = 0.003  # 0.3% minimum premium
        self.max_days_to_expiry = 7  # Weekly options

    def find_optimal_strike(self, ticker: str, position_qty: int):
        """Select strike with best risk/reward"""
        chain = self.oc.fetch_option_chain(self.kite, ticker)
        # Find call with delta closest to target
        # Filter: premium > min_premium_pct × spot
        # Prefer: nearest weekly expiry
        pass

    def manage_position(self, short_call_order_id: str):
        """Roll, close, or let expire"""
        # If stock > 90% of strike: roll up and out
        # If DTE = 0 and stock < strike: let expire worthless (keep premium)
        # If IV spike: consider closing early (lock in profit)
        pass
```

**Expected Impact:** +12-25% annual from premium income alone

#### 2.2 Cash-Secured Put Selling (Systematic Entries)

**New File:** `kite_connect/options/put_selling_strategy.py`

**Purpose:** Get paid to wait for entry prices on desired stocks

**Implementation:**
```python
class PutSellingStrategy:
    """
    Research: Volatility Risk Premium (Carr & Wu, 2009)
    
    For stocks in the BUY queue but not yet at entry price:
    1. Sell 1 OTM put at desired entry strike
    2. Collect premium while waiting
    3. If assigned: you get the stock at your desired price (win-win)
    4. If not assigned: keep premium, sell again next week
    
    Risk management:
    - Only sell on stocks you WANT to own (from Carver BUY signals)
    - Never sell puts without full cash reserve (cash-secured, not naked)
    - Maximum 3 concurrent put positions
    - Stop loss: Close if put doubles in value (100% loss on premium)
    """
    pass
```

**Expected Impact:** +5-10% annual from put premium + better entries

#### 2.3 VIX-Conditional Options Sizing

**File:** `kite_connect/options/` (new)

**Logic:**
- India VIX < 15: Sell more options (low vol = less risk, but less premium)
- India VIX 15-20: Normal sizing
- India VIX 20-25: Reduce selling by 50% (pickup premium increases but so does risk)
- India VIX > 25: STOP selling options, consider buying puts for protection

---

### PHASE 3: ADVANCED RISK FRAMEWORK (Week 4-5)

#### 3.1 Trade-Level Monte Carlo Simulation (NEW)

**New File:** `services/monte_carlo_risk.py`

**Purpose:** Replace broken GBM Monte Carlo with proper trade-level bootstrap

**Implementation:**
```python
class TradeBootstrapMonteCarlo:
    """
    Proper Monte Carlo for trading systems:
    NOT forecasting price direction (your readme is right: GBM is useless for that)
    INSTEAD: Estimating risk of your ACTUAL TRADING SYSTEM from its trade distribution
    
    This answers: "Given my system's actual win rate, avg win, avg loss, 
    what is the probability of a 20% drawdown? 50% drawdown? Ruin?"
    """
    
    def __init__(self, n_simulations: int = 10000, n_trades_per_sim: int = 500):
        self.n_sims = n_simulations
        self.n_trades = n_trades_per_sim

    def simulate(self, trade_returns: List[float]) -> MonteCarloResult:
        """
        Input: List of actual trade returns [+2.1%, -1.3%, +4.5%, -0.8%, ...]
        Method: Random resample with replacement (bootstrap)
        Output: Distribution of outcomes
        """
        equity_curves = []
        for _ in range(self.n_sims):
            # Random permutation of actual trades
            sampled = np.random.choice(trade_returns, size=self.n_trades, replace=True)
            equity = np.cumprod(1 + np.array(sampled))
            equity_curves.append(equity)
        
        equity_matrix = np.array(equity_curves)
        
        return MonteCarloResult(
            median_cagr=np.median(equity_matrix[:, -1]) ** (252/self.n_trades) - 1,
            p5_cagr=np.percentile(equity_matrix[:, -1], 5) ** (252/self.n_trades) - 1,
            p95_cagr=np.percentile(equity_matrix[:, -1], 95) ** (252/self.n_trades) - 1,
            max_drawdown_p95=self._compute_drawdown_percentile(equity_matrix, 95),
            probability_of_ruin=self._compute_ruin_probability(equity_matrix, threshold=0.5),
            optimal_kelly=self._compute_kelly_from_distribution(trade_returns),
            cvar_5pct=self._compute_cvar(trade_returns, alpha=0.05),
        )

    def block_bootstrap(self, trade_returns: List[float], block_size: int = 10):
        """Preserves serial correlation (winning/losing streaks)"""
        # Resample blocks of consecutive trades instead of individual
        pass

    def regime_conditional(self, trade_returns: List[float], 
                           regime_labels: List[str]):
        """Simulate drawing from regime-appropriate trade pool"""
        # Split trades by regime, simulate with regime-transition matrix
        pass
```

**Key Metrics Output:**
- **P(ruin):** Probability of 50% drawdown → guides position sizing
- **CVaR (5%):** Average loss in worst 5% of scenarios → guides stop-loss levels  
- **Optimal Kelly:** Data-driven Kelly fraction (not hardcoded 0.5)
- **Confidence interval on CAGR:** "Your system produces 25-55% CAGR with 90% confidence"

**Effort:** 3-4 days | **Impact:** Correct position sizing, know actual risk

#### 3.2 Portfolio Correlation Risk (NEW)

**New File:** `services/portfolio_correlation.py`

**Purpose:** Size positions considering correlation between holdings

**Implementation:**
```python
class PortfolioCorrelationRisk:
    """
    Current gap: Each position sized independently (as if uncorrelated)
    Reality: 6 banking stocks = 6× one bet, not 6 diversified bets
    
    Solution: Correlation-adjusted position sizing
    """
    
    def compute_portfolio_vol(self, positions: Dict[str, float], 
                                returns: pd.DataFrame) -> float:
        """True portfolio vol = sqrt(w' × Σ × w)"""
        weights = pd.Series(positions)
        cov = returns[list(positions.keys())].cov() * 252  # Annualize
        port_var = weights @ cov @ weights
        return np.sqrt(port_var)

    def max_position_given_portfolio(self, new_stock: str,
                                      current_positions: Dict[str, float],
                                      max_portfolio_vol: float = 0.25) -> float:
        """
        Given current portfolio, what's the max weight for new stock
        that keeps portfolio vol under target?
        """
        # Binary search or analytical solution
        # Returns max allowable weight
        pass

    def diversification_ratio(self, positions, returns):
        """DR = sum(w_i * σ_i) / σ_portfolio"""
        # DR > 1 means diversification is helping
        # DR close to 1 means highly correlated holdings
        pass
```

**Integration:** Wire into `position_sizer.py` as a post-sizing constraint

**Effort:** 2-3 days | **Impact:** Prevent correlated drawdowns

#### 3.3 Dynamic Kelly Scaling (ENHANCED)

**File:** `kite_connect/trading/risk_manager.py`

**Current:** Fixed half-Kelly (0.5×)
**Target:** Scale Kelly fraction by signal confidence

**Changes:**
```python
def dynamic_kelly(self, win_prob: float, avg_win: float, avg_loss: float,
                    confidence: float) -> float:
    """
    Kelly fraction = p - q/R  where R = avg_win/avg_loss, q = 1-p
    Scale by confidence: Kelly_scaled = Kelly × confidence × 0.5 (half-Kelly)
    
    confidence comes from IntegratedScorer (0 to 1)
    """
    R = avg_win / abs(avg_loss) if avg_loss != 0 else 2.0
    q = 1 - win_prob
    kelly = win_prob - q / R
    kelly = max(kelly, 0)  # Floor at 0 (don't short)
    kelly_scaled = kelly * confidence * 0.5  # Half-Kelly, confidence-weighted
    return min(kelly_scaled, 0.05)  # Cap at 5% per trade
```

**Impact:** High-conviction bets get larger, low-conviction get smaller

#### 3.4 Tail-Risk Hedging via Portfolio Puts

**New File:** `services/tail_risk_hedge.py`

**Purpose:** Protect portfolio during crashes

**Implementation:**
```python
class TailRiskHedge:
    """
    Spend 0.5-1.0% of portfolio per month on OTM NIFTY puts
    
    Logic:
    - Buy NIFTY puts at delta -0.10 to -0.15 (deep OTM, ~7-10% below spot)
    - Monthly roll (buy next month's put before current expires)
    - During normal markets: cost drain of ~6-12% annually
    - During crash (>15% drop): puts pay 3-10× (offsetting portfolio losses)
    
    Net effect: Reduces Sharpe slightly but dramatically reduces max drawdown
    Budget: 0.5% of portfolio/month in normal VIX, 0.25% when VIX > 20 (puts expensive)
    """
    pass
```

**Integration:** Scheduler auto-rolls puts monthly; increases put size when portfolio-level HHI is high

**Effort:** 3-4 days | **Impact:** Max drawdown from 30-40% → 15-20%

---

### PHASE 4: EVALUATION UPGRADE (Week 5-6)

#### 4.1 Comprehensive Risk Metrics Dashboard

**New File:** `services/risk_metrics.py`

**Add to paper trader + live dashboard:**

```python
class RiskMetrics:
    """All missing risk-adjusted return metrics"""
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free: float = 0.07/252) -> float:
        """Downside deviation only (penalizes losses, not gains)"""
        excess = returns - risk_free
        downside = excess[excess < 0]
        downside_std = np.sqrt(np.mean(downside**2))
        return excess.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """Annual return / Max drawdown (quality of recovery)"""
        annual_ret = (1 + returns).prod() ** (252/len(returns)) - 1
        max_dd = RiskMetrics.max_drawdown(returns)
        return annual_ret / abs(max_dd) if max_dd != 0 else 0

    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """P(gain > threshold) / P(loss > threshold)"""
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        return gains / losses if losses > 0 else float('inf')

    @staticmethod
    def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
        """Conditional VaR (Expected Shortfall at alpha)"""
        var = returns.quantile(alpha)
        return returns[returns <= var].mean()

    @staticmethod
    def ulcer_index(equity_curve: pd.Series) -> float:
        """Measure of drawdown severity and duration"""
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak * 100
        return np.sqrt(np.mean(dd**2))

    @staticmethod
    def recovery_factor(returns: pd.Series) -> float:
        """Net profit / Max drawdown"""
        net_profit = (1 + returns).prod() - 1
        max_dd = abs(RiskMetrics.max_drawdown(returns))
        return net_profit / max_dd if max_dd > 0 else 0
```

#### 4.2 Regime-Conditional Performance Tracking

**New File:** `services/regime_performance.py`

```python
class RegimePerformance:
    """Track strategy performance separately per regime"""
    
    def stratify_returns(self, daily_returns: pd.Series,
                          regime_labels: pd.Series) -> Dict[str, PerformanceStats]:
        """
        Output:
        - Bull Sharpe: 1.2
        - Bear Sharpe: -0.3  ← Problem! Strategy loses in bear markets
        - Range Sharpe: 0.5
        - Crisis Sharpe: -1.8  ← Needs hedging
        """
        stats = {}
        for regime in regime_labels.unique():
            mask = regime_labels == regime
            regime_returns = daily_returns[mask]
            stats[regime] = {
                'sharpe': self._sharpe(regime_returns),
                'sortino': self._sortino(regime_returns),
                'max_dd': self._max_dd(regime_returns),
                'avg_trade_return': regime_returns.mean(),
                'n_trades': len(regime_returns),
            }
        return stats
```

**Integration:** Add to walk-forward summary and live dashboard

#### 4.3 Transaction Cost Simulation in Walk-Forward

**File:** `services/walk_forward.py`

**Changes:**
```python
# In the backtest loop, after computing raw returns:
def _apply_costs(self, trade_returns: List[float], trade_count: int,
                   avg_position_value: float) -> List[float]:
    """
    IND costs per round-trip:
    - STT: 0.1% (sell side)
    - Transaction charges: 0.00345% (NSE)
    - GST: 18% on brokerage + transaction
    - Stamp duty: 0.015% (buy side)
    - Brokerage: 0 (Zerodha delivery)
    - Spread/slippage: 0.10-0.30% (depends on liquidity)
    Total: ~0.30-0.50% per round-trip
    
    Apply to each trade return:
    net_return = gross_return - round_trip_cost
    """
    ROUND_TRIP_COST = 0.004  # 0.4% (conservative estimate)
    return [r - ROUND_TRIP_COST for r in trade_returns]
```

#### 4.4 Strategy Decay Detection & Auto-Deallocation

**New File:** `services/strategy_decay.py`

```python
class StrategyDecayMonitor:
    """
    Track rolling Sharpe of each strategy.
    Alert when recent performance degrades significantly.
    Auto-reduce allocation.
    """
    
    def check_decay(self, strategy_name: str, 
                     recent_returns: pd.Series,
                     historical_sharpe: float) -> DecayStatus:
        """
        Compute 63-day rolling Sharpe (exponentially weighted)
        Compare to historical Sharpe
        
        Decay levels:
        - HEALTHY: recent_sharpe > 0.5 × historical_sharpe
        - DEGRADED: 0.25 × historical < recent < 0.5 × historical
        - DEAD: recent_sharpe < 0.25 × historical_sharpe
        - INVERTED: recent_sharpe < 0 (strategy is now losing money)
        
        Action:
        - HEALTHY: Full allocation
        - DEGRADED: Halve allocation, alert
        - DEAD: Zero allocation, alert, schedule re-calibration
        - INVERTED: Zero allocation + investigate
        """
        pass
```

---

### PHASE 5: REGIME-ADAPTIVE STRATEGY SWITCHING (Week 6-7)

#### 5.1 Regime-Conditional Strategy Mix

**File:** `services/regime_detector.py`, `services/forecast_combiner.py`

**Current:** Regime detector identifies bull/bear/range/crisis but only DecisionEngine uses it
**Target:** Strategy weights change based on regime

**Implementation:**
```python
REGIME_STRATEGY_WEIGHTS = {
    "trending_bull": {
        "ewmac": 0.30,     # Momentum works great
        "carry": 0.15,
        "screener": 0.15,
        "decision_engine": 0.15,
        "momentum_factor": 0.20,  # NEW: top-decile momentum
        "pead": 0.05,
    },
    "trending_bear": {
        "ewmac": 0.10,     # Momentum gets crushed in reversals
        "carry": 0.10,
        "screener": 0.10,
        "decision_engine": 0.30,  # Fundamentals matter more
        "momentum_factor": 0.05,
        "mean_reversion": 0.20,   # NEW: mean-reversion works in bear rallies
        "pead": 0.15,             # Earnings drift still works
    },
    "range_bound": {
        "ewmac": 0.05,     # Momentum fails in ranges
        "carry": 0.20,     # Carry is steady in ranges
        "screener": 0.15,
        "decision_engine": 0.15,
        "mean_reversion": 0.30,  # Dominant strategy in ranges
        "pead": 0.10,
        "options_selling": 0.05, # High vol = rich premiums
    },
    "crisis": {
        "cash": 0.70,      # Mostly cash
        "tail_hedge": 0.20, # Long puts
        "carry": 0.10,
    },
}
```

**Integration:** `forecast_combiner.py` reads current regime from `regime_detector.py` and applies appropriate weights.

#### 5.2 Mean-Reversion Module Activation

**File:** `trading_strategies/statistical_arbitrage/mean_reversion.py`

**Current:** Mean reversion exists as standalone backtest
**Target:** Integrated as a Carver forecast source

**Changes:**
1. Add `generate_forecast()` method returning Carver-scale forecast (-20 to +20)
2. Filter stocks with Hurst < 0.45 (confirmed mean-reverting)
3. Signal: Z-score of 5-day return < -2σ → forecast = +20 (strong buy bounce)
4. Wire into `forecast_combiner.py` as 6th rule

---

### PHASE 6: US STOCKS INSIGHTS ENGINE (Week 7-8)

Since US stocks are manual execution only, focus on **insights & analysis quality**.

#### 6.1 US Market Alpha Insights Dashboard

**Enhance existing US pipeline to provide:**

1. **Top Momentum Stocks (S&P 500)**
   - 12-1 month return ranking
   - Relative strength vs SPY
   - Sector breakdown

2. **Mean-Reversion Opportunities**
   - Stocks >2σ below 50-day MA in S&P 500
   - Hurst exponent confirmation
   - Expected half-life of reversion

3. **Earnings Drift Opportunities**
   - Recent earnings surprises (SUE > 1.0)
   - Days since announcement (drift window remaining)
   - Expected drift magnitude

4. **Options Premium Opportunities**
   - High IV rank stocks (>70th percentile)
   - IV vs HV spread (implied > realized = premium harvestable)
   - Upcoming catalysts (earnings, FDA, etc.)

5. **Macro Regime Context**
   - US VIX level and regime
   - Fed rate expectations (CME FedWatch)
   - Credit spreads (investment grade vs high yield)
   - Market breadth (advance/decline)

**Implementation:** Enhance `services/us_carver_pipeline.py` to output a `USMarketInsights` dataclass with all the above, displayed on the frontend.

#### 6.2 US Portfolio Optimization (Manual Execution Aid)

```python
class USPortfolioOptimizer:
    """
    Given your current US holdings, suggest:
    1. Rebalancing trades (momentum tilt)
    2. Exit candidates (decayed momentum)
    3. Hedging trades (correlation reduction)
    4. Options overlay (covered calls on large positions)
    """
    pass
```

---

### PHASE 7: RL BOT ENHANCEMENT (Week 8-9)

#### 7.1 Enable RL with Proper Configuration

**File:** `config.py`

```python
RL_ENABLED = True  # Enable
RL_ALGORITHM = "PPO"
RL_TOTAL_TIMESTEPS = 1_000_000  # Double from 500K
RL_TRAIN_DAYS = 756  # 3 years (more data)
RL_WALK_FORWARD_FOLDS = 8  # More folds for robustness
RL_LAYER_WEIGHT = 0.15  # 15% of integrated scorer
```

#### 7.2 Enhanced Reward Function

**File:** `services/rl_bot/reward.py`

Add **risk-adjusted reward** that directly optimizes Sortino:

```python
def sortino_reward(self, portfolio_returns: np.ndarray, lookback: int = 20) -> float:
    """
    Reward = Sortino ratio of recent returns
    This trains the agent to maximize upside and minimize downside
    """
    excess = portfolio_returns[-lookback:] - self.risk_free_daily
    downside = excess[excess < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-6
    sortino = excess.mean() / downside_std
    return sortino * 0.1  # Scale to reasonable reward magnitude
```

#### 7.3 Multi-Agent Ensemble

Instead of one RL agent, train **3 specialized agents:**
1. **Trend Agent:** Trained on trending market data only (Hurst > 0.55)
2. **Reversion Agent:** Trained on mean-reverting data only (Hurst < 0.45)
3. **Ensemble:** Regime detector selects which agent to listen to

---

### PHASE 8: CONTINUOUS IMPROVEMENT INFRASTRUCTURE (Week 9-10)

#### 8.1 Automated Strategy Tournament

**New File:** `services/strategy_tournament.py`

```python
class StrategyTournament:
    """
    Monthly automated competition:
    1. Run all 20+ strategies on last 3 months of data
    2. Rank by OOS Sharpe, Sortino, Max DD, Calmar
    3. Top 5 strategies get allocation in live portfolio
    4. Bottom 5 strategies get zero allocation
    5. Any strategy with negative Sharpe over 3 months → auto-disabled
    """
    pass
```

#### 8.2 Real-Time Execution Quality Monitoring

**New File:** `services/execution_quality.py`

```python
class ExecutionQualityMonitor:
    """
    Track:
    - Implementation shortfall: Expected return − Actual return (per trade)
    - Slippage: Fill price − Signal price
    - Market impact: Price movement caused by our orders
    - Fill rate: % of orders fully filled
    - Latency: Signal generation → Order placed → Order filled
    
    Alert if:
    - Average slippage > 30 bps (currently estimated at 20 bps)
    - Fill rate < 95%
    - Average latency > 2 seconds
    """
    pass
```

#### 8.3 Weekly Automated Report

**Enhancement to `scheduler.py`:**

Every Saturday morning, auto-generate:
1. Portfolio P&L (weekly, MTD, YTD)
2. vs NIFTY benchmark (alpha generated)
3. Sharpe, Sortino, Calmar (rolling 30d, 90d, 252d)
4. Strategy attribution (which strategies contributed most)
5. Regime analysis (what regime are we in, what to expect)
6. Upcoming catalysts (earnings, macro events)
7. Risk metrics (portfolio vol, max correlation pair, CVaR)
8. Monte Carlo update (P(ruin), confidence intervals)

---

## PART 4: PRIORITIZED IMPLEMENTATION ROADMAP

### TIER 1: HIGH-IMPACT, LOW-EFFORT (Do First)

| # | Action | Files | Days | Expected CAGR Impact |
|---|--------|-------|------|---------------------|
| 1 | Wire Carver pipeline to IND | auto_executor.py | 1-2 | +8-15% |
| 2 | Activate cost speed limit | carver_pipeline.py | 0.5 | +2-3% (avoid bad trades) |
| 3 | Live capital rolling | volatility_target.py, auto_executor.py | 1 | Correct sizing |
| 4 | Drawdown halt 50% → 20% | volatility_target.py, portfolio_vol_monitor.py | 0.5 | Capital preservation |
| 5 | Calibrate forecast scalars | carver_calibration.py | 1 | Correct sizing |
| 6 | Add Sortino/Calmar/CVaR metrics | New: risk_metrics.py | 1 | Better evaluation |
| 7 | Transaction costs in walk-forward | walk_forward.py | 0.5 | Realistic backtests |

**Subtotal: ~6 days, +10-18% CAGR improvement**

### TIER 2: HIGH-IMPACT, MEDIUM-EFFORT (Do Second)

| # | Action | Files | Days | Expected CAGR Impact |
|---|--------|-------|------|---------------------|
| 8 | Cross-sectional momentum factor | New: momentum_factor.py | 3 | +12-18% |
| 9 | Monte Carlo risk engine (trade bootstrap) | New: monte_carlo_risk.py | 3-4 | Correct position sizing |
| 10 | Portfolio correlation risk | New: portfolio_correlation.py | 2-3 | Prevent correlated drawdowns |
| 11 | PEAD earnings drift | New: pead_strategy.py | 2-3 | +3-5% |
| 12 | Regime-conditional strategy weights | forecast_combiner.py, regime_detector.py | 2-3 | +5-8% (right strategy at right time) |
| 13 | Dynamic Kelly scaling | risk_manager.py | 1-2 | Better risk-adjusted sizing |
| 14 | Strategy decay detection | New: strategy_decay.py | 2 | Prevent dead strategies from losing money |

**Subtotal: ~18 days, +20-30% CAGR improvement**

### TIER 3: MAJOR ALPHA SOURCE (Do Third)

| # | Action | Files | Days | Expected CAGR Impact |
|---|--------|-------|------|---------------------|
| 15 | Covered call strategy | New: covered_call_strategy.py | 4-5 | +12-25% |
| 16 | Cash-secured put selling | New: put_selling_strategy.py | 3-4 | +5-10% |
| 17 | Tail-risk hedging (portfolio puts) | New: tail_risk_hedge.py | 3-4 | Max DD 30% → 15% |
| 18 | RL bot enable + enhance | config.py, reward.py | 3-4 | +3-5% |
| 19 | Factor momentum (dynamic weights) | New: factor_momentum.py | 2 | +2-4% |

**Subtotal: ~18 days, +22-44% CAGR improvement**

### TIER 4: REFINEMENT & INFRASTRUCTURE (Do Last)

| # | Action | Files | Days | Expected CAGR Impact |
|---|--------|-------|------|---------------------|
| 20 | Multi-TF sector rotation | sector_rotation.py | 1 | +2-3% |
| 21 | Mean-reversion as Carver rule | mean_reversion.py | 2 | +3-5% in range markets |
| 22 | US insights dashboard | us_carver_pipeline.py | 3-4 | Better manual US trades |
| 23 | Strategy tournament | New: strategy_tournament.py | 3 | Ongoing alpha maintenance |
| 24 | Execution quality monitor | New: execution_quality.py | 2 | Slippage reduction |
| 25 | Regime-conditional performance | New: regime_performance.py | 2 | Better strategy oversight |
| 26 | Weekly automated report | scheduler.py | 2-3 | Decision support |
| 27 | Multi-agent RL ensemble | rl_bot/*.py | 5 | +1-3% |

**Subtotal: ~23 days**

---

## PART 5: REALISTIC CAGR PROJECTION

### Scenario Analysis (Post All Implementations)

| Scenario | Market Regime | Strategy Mix | Estimated CAGR | Sharpe | Max Drawdown |
|----------|--------------|-------------|----------------|--------|-------------|
| **Bull** | Trending up, VIX < 18 | Momentum + EWMAC + Options | 60-80% | 1.5 | 12-18% |
| **Normal** | Mixed, VIX 15-22 | Balanced all strategies | 35-55% | 0.9 | 18-25% |
| **Bear** | Trending down, VIX > 22 | Mean-reversion + Puts + Cash | 5-15% | 0.3 | 20-30% |
| **Crisis** | Crash, VIX > 30 | Cash + Hedges | -5 to +5% | -0.2 | 15-20% (hedged) |
| **Blended (typical year)** | 60% normal, 25% bull, 10% bear, 5% crisis | Dynamic | **38-58%** | **0.8-1.1** | **18-25%** |

### Key Assumptions
1. Capital: ₹5-10L for IND, $10K for US
2. No leverage beyond 1× for equity CNC (options provide synthetic leverage)
3. Options overlay provides 12-25% additional return
4. Momentum premium captures 50-70% of theoretical due to costs/slippage
5. Strategy ensemble reduces individual strategy failure impact
6. Monte Carlo P(ruin) < 5% at all times

### Risk Warnings
- **60% CAGR is aggressive** — requires concentrated positions, options selling, and favorable markets
- **Drawdowns of 20-30% are expected** even with risk management
- **Options selling has tail risk** (unlimited loss on naked positions — always hedge)
- **Momentum crashes** can give back 6-12 months of gains in 2-4 weeks
- **Tax drag:** 15% STCG (holding < 1 year) reduces effective returns by 5-10%

---

## PART 6: QUICK REFERENCE — KEY PAPERS & RESOURCES

### Must-Read Quantitative Papers
1. **Jegadeesh & Titman (1993)** — "Returns to Buying Winners and Selling Losers" (momentum)
2. **Carver, R. (2015-2022)** — "Systematic Trading", "Leveraged Trading", "Advanced Futures Trading"
3. **Asness, Moskowitz & Pedersen (2013)** — "Value and Momentum Everywhere"
4. **Lo & MacKinlay (1990)** — "When Are Contrarian Profits Due to Stock Market Overreaction?"
5. **Bernard & Thomas (1989)** — "Post-Earnings-Announcement Drift" (PEAD)
6. **Carr & Wu (2009)** — "Variance Risk Premiums" (options selling edge)
7. **De Prado (2018)** — "Advances in Financial Machine Learning" (your AFML features)
8. **Choueifaty & Coignard (2008)** — "Toward Maximum Diversification"
9. **Ang & Timmermann (2012)** — "Regime Changes and Financial Markets"
10. **Arnott et al. (2021)** — "Factor Momentum" (dynamic strategy weighting)

### Indian Market Specific
11. **Agarwalla, Jacob & Varma (2013)** — "Momentum Effect in Indian Stock Market" (BSE/NSE)
12. **Sehgal & Jain (2011)** — "Short-term Momentum Patterns in Stock and Sectoral Returns"
13. **NSE Research Papers** — nse-india.com/research (sector rotation, institutional flows)

### Implementation References
14. **QuantConnect Lean** — open-source algorithmic trading engine (C#, architecture reference)
15. **Zipline/Backtrader** — Python backtesting frameworks (compare evaluation approaches)
16. **Ernest Chan** — "Quantitative Trading", "Algorithmic Trading" (practical implementation)
17. **Risk.net** — CVaR, Expected Shortfall implementation guides

---

## APPENDIX A: CONFIGURATION CHANGES SUMMARY

```python
# config.py — Recommended changes for 60% CAGR target

# === CARVER FRAMEWORK (activate for IND) ===
CARVER_ENABLED = True                    # Was: True but not wired for IND
CARVER_ANNUAL_VOL_TARGET = 0.30          # Was: 0.20 (increase for higher returns)
CARVER_INITIAL_CAPITAL = 500_000         # ₹5L
CARVER_DEFAULT_IDM = 1.8                 # Was: 1.6 (slightly more concentrated)
CARVER_INERTIA_THRESHOLD = 0.08          # Was: 0.10 (more responsive rebalancing)

# === RISK MANAGEMENT (tighten) ===
PORTFOLIO_DRAWDOWN_HALT = 0.20           # Was: 0.50 (critical fix)
PORTFOLIO_DRAWDOWN_WARNING = 0.10        # NEW
PORTFOLIO_DRAWDOWN_CRITICAL = 0.15       # NEW
MAX_OPEN_TRADES = 8                      # Was: 6 (slightly more diversified)
MIN_RR_RATIO = 2.0                       # Was: 2.5 (slightly less restrictive to increase trade count)
VIX_CAUTION_THRESHOLD = 18.0             # Was: 20.0 (earlier caution)

# === EVALUATION (add missing metrics) ===
COMPUTE_SORTINO = True                   # NEW
COMPUTE_CALMAR = True                    # NEW
COMPUTE_CVAR = True                      # NEW
MONTE_CARLO_SIMULATIONS = 10_000         # NEW
MONTE_CARLO_BLOCK_SIZE = 10              # NEW (block bootstrap)

# === MOMENTUM FACTOR (new) ===
MOMENTUM_FORMATION_PERIOD = 252          # 12 months
MOMENTUM_SKIP_PERIOD = 21               # Skip most recent month
MOMENTUM_TOP_N = 20                      # Top 20 stocks
MOMENTUM_REBALANCE_FREQUENCY = "monthly" # Monthly rebalance

# === OPTIONS OVERLAY (new) ===
OPTIONS_OVERLAY_ENABLED = True           # NEW
COVERED_CALL_DELTA_TARGET = 0.25         # OTM
COVERED_CALL_MIN_PREMIUM_PCT = 0.003     # 0.3% minimum
PUT_SELLING_MAX_CONCURRENT = 3           # Max concurrent put positions
TAIL_HEDGE_BUDGET_PCT = 0.005            # 0.5% of portfolio per month

# === RL BOT (enable) ===
RL_ENABLED = True                        # Was: False
RL_TOTAL_TIMESTEPS = 1_000_000           # Was: 500_000
RL_TRAIN_DAYS = 756                      # Was: 504 (3 years)
RL_WALK_FORWARD_FOLDS = 8               # Was: 6

# === PEAD (new) ===
PEAD_ENABLED = True                      # NEW
PEAD_SUE_THRESHOLD = 1.0                 # Minimum surprise for signal
PEAD_HOLD_PERIOD = 45                    # Trading days
PEAD_FORECAST_WEIGHT = 0.15              # Weight in combined forecast
```

## APPENDIX B: NEW FILES TO CREATE

```
services/
├── momentum_factor.py          # Phase 1.1 — Cross-sectional momentum
├── pead_strategy.py            # Phase 1.3 — Post-earnings drift
├── factor_momentum.py          # Phase 1.4 — Dynamic strategy weighting
├── monte_carlo_risk.py         # Phase 3.1 — Trade bootstrap Monte Carlo
├── portfolio_correlation.py    # Phase 3.2 — Correlation-aware sizing
├── tail_risk_hedge.py          # Phase 3.4 — Portfolio put protection
├── risk_metrics.py             # Phase 4.1 — Sortino/Calmar/CVaR/Omega
├── regime_performance.py       # Phase 4.2 — Regime-conditional tracking
├── strategy_decay.py           # Phase 4.4 — Auto-deallocation
├── strategy_tournament.py      # Phase 8.1 — Monthly strategy competition
├── execution_quality.py        # Phase 8.2 — Slippage/fill tracking

kite_connect/options/
├── covered_call_strategy.py    # Phase 2.1 — Covered call writing
├── put_selling_strategy.py     # Phase 2.2 — Cash-secured puts
```

## APPENDIX C: FILES TO MODIFY

```
kite_connect/trading/auto_executor.py   # Wire Carver for IND
services/carver_pipeline.py             # Add cost filter call
services/volatility_target.py           # Live capital rolling, drawdown thresholds
services/portfolio_vol_monitor.py       # Wire into execution gate
services/forecast_scalar.py             # Auto-calibration
services/forecast_combiner.py           # Regime-conditional weights, factor momentum
services/walk_forward.py                # Transaction cost simulation
services/regime_detector.py             # Strategy switching output
services/sector_rotation.py             # Multi-timeframe
kite_connect/trading/risk_manager.py    # Dynamic Kelly, correlation constraint
config.py                               # All new config parameters
scheduler.py                            # Weekly report, forecast recalibration
```

---

**Total Implementation Effort: ~65 working days (8-10 weeks)**
**Expected Outcome: 35-60% CAGR (regime-dependent)**
**Key Risk: Concentrated portfolio + options selling in a crash scenario**
**Mitigation: Tail-risk hedging + 20% drawdown halt + regime-adaptive sizing**
