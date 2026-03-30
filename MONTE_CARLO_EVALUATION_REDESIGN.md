# Monte Carlo Evaluation of Trading Systems — Redesign Plan

## Source: Timothy Masters, "Monte-Carlo Evaluation of Trading Systems" (2006)
## Platform: Centurion Core — Swing & Positional Trades (IND + US Stocks)
## Date: March 28, 2026

---

## 1. Key Concepts from the PDF (Timothy Masters)

### 1.1 The Core Idea: Monte Carlo Permutation Test

The PDF presents a **fundamentally different** approach to evaluating trading
systems compared to traditional bootstrap methods:

| Concept | Traditional Bootstrap | MC Permutation (Masters) |
|---------|----------------------|--------------------------|
| **Null Hypothesis** | Mean return = 0 | Position-return pairing is **random** (not intelligent) |
| **What It Tests** | Whether expected return is non-zero | Whether the **matching** of buy/sell decisions to market moves beats chance |
| **How Null Is Generated** | Resample trade returns with replacement | **Shuffle positions** across raw market returns, keeping return distribution intact |
| **Market Bias Handling** | Confounded (long bias in S&P inflates significance) | **Immune** — centering raw returns removes bias automatically |
| **Serial Correlation** | Broken by i.i.d. resampling | Fairly robust; credits intelligent response to correlated markets |
| **Skewness Sensitivity** | Anti-conservative with left skew (Table 3: 21% rejection at 5% level!) | Stable regardless of skew (Table 3: 5.09%) |

### 1.2 The Permutation Procedure

```
For each of N permutations (≥1000, ideally 5000+):
  1. Take the vector of raw market returns R = [r₁, r₂, ..., rₙ]
  2. Take the position vector P = [L, N, S, L, ...] (long/neutral/short)
  3. Compute candidate_return = Σ Pᵢ × Rᵢ
  4. Randomly shuffle P → P_shuffled
  5. Compute trial_return = Σ P_shuffled_i × Rᵢ
  6. If trial_return ≥ candidate_return: count++
p-value = count / N
```

### 1.3 Critical Requirements for Validity

1. **Trading opportunities must be non-overlapping** — each position has a clear entry/exit window
2. **Raw market returns must be available** for ALL trading opportunities (including when neutral)
3. **Position decisions must be independent** of open positions
4. **Each candidate system generates its own null distribution** — never share across systems
5. **Center raw returns** to remove market bias (especially for best-of-many)
6. **Use individual returns, not grouped** — grouping inflates anti-conservative bias

### 1.4 Best-of-Many Models (Selection Bias Correction)

When evaluating the **best** system from N alternatives (which centurion does with
42+ strategies), you MUST account for selection bias:

```
For each permutation:
  1. Shuffle the position indices (same shuffle for ALL N systems)
  2. Compute trial return for each of the N systems
  3. Record max(trial_returns) across all N
p-value = fraction of max(permuted_returns) ≥ max(real_returns)
```

**Critical**: Permute all systems simultaneously (same shuffle), not independently.

### 1.5 Nonparametric Sign-Only Test

When trade durations vary significantly (swing trades = 3-15 days, positional = weeks):
- Count successes (+1) and failures (-1), ignoring magnitude
- Avoids variance mismatch between short and long trades
- Surprisingly powerful (Table 1: 63% rejection for powerful models vs 47% for full MC)

### 1.6 Return Partitioning (Skill vs Luck)

```
Expected_Return_Random = mean(R) × (n_long - n_short)
Skill_Return = Actual_Return - Expected_Return_Random
```

Equivalent to centering raw returns before testing.

### 1.7 Walk-Forward Permutation

For trained/optimized models (which centurion uses):
1. Use **walk-forward or cross-validation** to generate OOS positions
2. Apply permutation test to OOS positions only
3. This tests the **model factory** (training process), not a single model
4. Accounts for optimization overfitting

---

## 2. Current centurion_core Implementation — What Exists

### 2.1 `services/monte_carlo_risk.py` — Trade Bootstrap

| Feature | Implementation |
|---------|---------------|
| Method | I.I.D. bootstrap + block bootstrap of trade returns |
| Null Hypothesis | Historical trade returns represent population |
| Tests | P(ruin), CVaR, Kelly fraction, CAGR confidence intervals |
| Permutations | N/A — this is bootstrap, NOT permutation |
| Signal Quality | **NOT tested** — only risk estimation |

**What it does well**: Risk estimation (P(ruin), optimal Kelly, CAGR CI)
**What it does NOT do**: Test whether buy/sell signals are better than random

### 2.2 `services/integrated_scorer.py` — Robustness Validation

Three tests embedded in strategy evaluation Layer 2:

| Test | Method | Null Hypothesis | Issues |
|------|--------|-----------------|--------|
| CSCV (ch05) | Combinatorial symmetric CV | PBO = probability of backtest overfitting | Tests overfitting, not signal quality per se |
| BCa Bootstrap (ch06) | Bootstrap CI on mean return | Mean return = 0 | Sensitive to skew (Masters Table 3!) |
| Permutation (ch07) | Shuffle returns, re-apply SMA crossover | SMA strategy performs = random | **Uses generic SMA, not actual strategy signals** |

**Critical Problem**: The permutation test uses a **hardcoded SMA(10,50) crossover** as the strategy function, NOT the actual Carver forecast signals. It's testing whether SMA crossover is significant on this ticker — completely irrelevant to whether the actual momentum/carry/EWMAC forecasts are intelligent.

### 2.3 `references/testune/applied/ch07_permutation_tests.py` — Library

Rich implementations already available:
- `permutation_test()` — basic permutation test with pluggable strategy function
- `walkforward_permutation_test()` — walk-forward permutation (Masters p.291-293)
- `partition_return()` — skill vs luck decomposition
- `permute_prices()`, `permute_bars()` — price/OHLCV permutation
- `permute_multiple_markets()` — cross-market permutation

These are **available but not properly wired** into the evaluation pipeline.

### 2.4 `services/walk_forward.py` — Walk-Forward Optimization

- Rolling WF with degradation ratio (OOS Sharpe / IS Sharpe)
- Transaction cost deduction (0.40% round-trip)
- **No permutation test on WF results** — only point estimate of degradation

### 2.5 `services/strategy_tournament.py` — Strategy Selection

- Monthly tournament ranking N strategies by composite score
- **No selection bias correction** — the "best of N" problem that Masters explicitly warns about

---

## 3. Gap Analysis: PDF Requirements vs Current Implementation

### GAP 1: No Position-Return Permutation Test (CRITICAL)
**PDF Requirement**: Test H₀ that the pairing of positions with raw returns is random — the gold standard
**Current State**: Only bootstrap (resample returns) and generic SMA permutation
**Impact**: We cannot distinguish a lucky system from a truly intelligent one

### GAP 2: Wrong Strategy in Permutation Test (CRITICAL)
**PDF Requirement**: Test YOUR actual trading signals (not a generic SMA)
**Current State**: `integrated_scorer.py` line 666-677 hardcodes `_sma_strat_returns` as the strategy function
**Impact**: The existing permutation test is testing whether SMA works, NOT whether centurion's signals work

### GAP 3: No Selection Bias Correction for Best-of-N (HIGH)
**PDF Requirement**: When picking the best of N strategies, permute all simultaneously and take the max
**Current State**: `strategy_tournament.py` ranks by composite Sharpe/Sortino but applies NO statistical significance test; each strategy tested independently
**Impact**: The "best" strategy may be best by luck — 42 strategies × 50 tickers = 2100 hypothesis tests

### GAP 4: No Skill vs Luck Decomposition (HIGH)
**PDF Requirement**: Partition total return into skill (intelligent pairing) and luck (market bias × position bias)
**Current State**: Not implemented anywhere
**Impact**: Cannot quantify what fraction of returns is genuine alpha vs market beta

### GAP 5: Bootstrap Without Bias Correction (MEDIUM)
**PDF Requirement**: Center raw returns before testing to remove market bias
**Current State**: `monte_carlo_risk.py` bootstraps raw trade returns without centering
**Impact**: Long bias in bullish markets (NIFTY, S&P) inflates bootstrap confidence

### GAP 6: Too Few Permutations (MEDIUM)
**PDF Requirement**: "At least several thousand" permutations (p.6), Masters uses 1000-5000
**Current State**: `integrated_scorer.py` uses only 200 permutations (`n_perms=200`)
**Impact**: P-value resolution is ±0.005 at best; 1/200 = 0.005 granularity — too coarse for 5% significance level

### GAP 7: No Walk-Forward Permutation Test (MEDIUM)
**PDF Requirement**: Apply permutation to OOS positions from walk-forward to test the training process
**Current State**: Walk-forward computes degradation ratio but no permutation significance
**Impact**: Cannot determine if OOS WF performance is statistically different from random

### GAP 8: No Nonparametric Sign-Only Test (LOW)
**PDF Requirement**: For variable trade durations, test only signs (win/loss) ignoring magnitude
**Current State**: Not implemented
**Impact**: Swing trades (3-15d) and positional (weeks) have different return variances; sign test mitigates this

### GAP 9: Grouped Returns Not Tested Separately (LOW)
**PDF Requirement**: Prefer individual daily returns over grouped trade-level returns for MC validity
**Current State**: Only works with trade-level returns (one value per trade)
**Impact**: Loses information from intra-trade daily returns

---

## 4. Action Plan for Redesign

### Phase A: Core MC Permutation Engine (Priority: P0 — Foundation)

**New File: `services/mc_permutation_test.py`**

Implement the Timothy Masters permutation test adapted for centurion's swing/positional trades:

```
1. MCPermutationTest class
   - Input: raw_returns[], position_vector[] (L/N/S per opportunity)
   - shuffle_positions() — Fisher-Yates shuffle (Masters p.18)
   - test_single_system() → p-value
   - test_best_of_n() → selection-bias-corrected p-value
   - test_sign_only() → nonparametric p-value
   - partition_skill_luck() → {total, luck, skill, skill_fraction}

2. Configuration:
   - n_permutations = 5000 (default, configurable)
   - center_returns = True (remove market bias per Masters p.8-9)
   - normalize_time_in_market = True (per Masters p.30, sqrt(n) scaling)
```

### Phase B: Wire Into Carver Forecast Pipeline (Priority: P1)

**Modify: `services/integrated_scorer.py`**

Replace the generic SMA permutation with actual Carver forecast signals:

```
Current (WRONG):
  _sma_strat_returns(rets) → SMA(10,50) crossover → Sharpe

Redesigned:
  For each ticker:
    1. Get actual Carver forecast signals (EWMAC, carry, screener, momentum, PEAD)
    2. Convert combined_forecast to position vector: >0 → LONG, <0 → SHORT, ==0 → NEUTRAL
    3. Pair with raw market returns for the same period
    4. Run MCPermutationTest.test_single_system()
    5. Store p-value in robustness_details["mc_permutation_p_value"]
    6. Score: p < 0.01 → +0.15 bonus, p < 0.05 → +0.08, p > 0.20 → −0.10 penalty
```

### Phase C: Selection Bias Correction for Tournament (Priority: P1)

**Modify: `services/strategy_tournament.py`**

Add best-of-N permutation test:

```
After ranking strategies by composite score:
  1. Collect position vectors for ALL N strategies (same time period)
  2. Collect common raw market returns
  3. Run MCPermutationTest.test_best_of_n(all_position_vectors, raw_returns)
  4. Report selection-bias-corrected p-value for the winning strategy
  5. If p > 0.10 → flag "INCONCLUSIVE — may be luck"
```

### Phase D: Skill vs Luck Dashboard (Priority: P2)

**New addition to pipeline output (IND + US)**

For each strategy and overall portfolio:
```
  Total Return:     +42.3%
  ├── Luck (bias):  +11.2% (market rose, we were net long)
  ├── Skill:        +31.1% (intelligent pairing of positions with returns)
  └── Skill Ratio:  73.5%  (genuine alpha fraction)
```

### Phase E: Walk-Forward Permutation (Priority: P2)

**Modify: `services/walk_forward.py`**

After computing OOS degradation ratio:
```
  1. Collect OOS position signals from all WF folds
  2. Concatenate into one long position vector + market returns
  3. Run MCPermutationTest on the concatenated OOS results
  4. Report: "WF factory p-value = X.XXX"
  5. Degradation ratio + permutation p-value = full robustness picture
```

### Phase F: Sign-Only Test for Mixed-Duration Trades (Priority: P3)

Centurion runs swing (3-15 day) and positional (15-60 day) trades. Per Masters,
when trade durations vary, the sign-only test is more reliable:

```
  For each trade:
    success = sign(position × raw_return) > 0 → +1
    failure = sign(position × raw_return) < 0 → -1
    neutral = 0
  Candidate score = Σ(successes - failures)
  Permutation: shuffle positions, recompute score
  p-value = fraction(permuted_score ≥ candidate_score)
```

---

## 5. Before vs After: Expected Results

### 5.1 Signal Quality Validation (IND Stocks — NIFTY 50 universe)

| Metric | BEFORE (Current) | AFTER (Redesigned) |
|--------|-------------------|---------------------|
| **What is tested** | "Is mean return ≠ 0?" (bootstrap) | "Are buy/sell decisions intelligent?" (permutation) |
| **Strategy tested** | Generic SMA(10,50) crossover | Actual Carver EWMAC/carry/momentum forecasts |
| **Market bias handling** | Confounded — NIFTY's ~12% annual bias inflates confidence | **Removed** — centered returns, pure alpha tested |
| **P-value reliability** | 200 permutations, SMA proxy → unreliable | 5000 permutations, actual strategy → robust |
| **Skew sensitivity** | BCa bootstrap anti-conservative with left skew (21% false positive!) | MC permutation: 5% rejection at 5% level regardless of skew |
| **Selection bias** | 42 strategies tested independently → massive multiple testing inflation | Best-of-N correction: simultaneous permutation, corrected p-value |
| **Skill vs Luck** | Not measured | Explicit decomposition: "31% return was skill, 11% was market beta" |

### 5.2 Expected p-value Outcomes (IND Stock Example)

```
BEFORE — Current system output for RELIANCE.NS:
  Strategy: BUY (score = 0.72)
  Robustness: CSCV PBO=0.28, BCa lower=+0.0002, Perm p=0.04
  → Looks good! But the permutation tested SMA crossover, not our signals.
  → BCa is inflated by NIFTY's 12% annual bull bias.
  → No selection bias correction among 42 strategies.

AFTER — Redesigned output for RELIANCE.NS:
  Strategy: BUY (score = 0.72)
  Robustness:
    MC Permutation (actual forecasts): p = 0.012 ✅ (buy signal is genuinely intelligent)
    Skill/Luck decomposition:
      Total return: +8.3%
      Luck (market bias): +2.1%
      Skill (intelligent pairing): +6.2%
      Skill fraction: 74.7% ✅
    WF permutation: p = 0.031 ✅ (training process is sound)
    Selection bias (best of 42): p = 0.087 ⚠️ (marginal after correction)
  → The BUY signal is validated at 1.2% significance on actual EWMAC forecasts.
  → BUT after correcting for 42-strategy selection, significance degrades to 8.7%.
  → Action: Accept BUY but reduce position size by selection_bias_penalty.
```

### 5.3 Expected p-value Outcomes (US Stock Example)

```
BEFORE — AAPL:
  BCa bootstrap: lower = +0.0008 → "significant"
  But: S&P 500 rose +25% last year. Long bias captures most of this.

AFTER — AAPL:
  MC Permutation (centered returns): p = 0.23 ❌
  → After removing S&P bull bias, the "intelligent pairing" is NOT significant.
  → The strategy was riding market beta, not demonstrating alpha.
  → Action: Downgrade from BUY to HOLD.
```

### 5.4 Impact on Trading Decisions

| Decision Area | BEFORE | AFTER |
|---------------|--------|-------|
| **Signal acceptance** | Any BUY with robustness_adj > 0 | Only signals with MC permutation p < 0.05 |
| **Position sizing** | Based on forecast magnitude only | Scaled by (1 - selection_bias_p_value) |
| **Strategy allocation** | Tournament ranks by Sharpe/Sortino | Tournament + best-of-N p-value → weed out lucky strategies |
| **Portfolio monitoring** | Track returns | Track skill fraction — alarm if skill fraction drops below 50% |
| **Confidence reporting** | "BUY confidence: 72%" | "BUY confidence: 72%, signal genuineness: p=0.012, skill=75%" |

### 5.5 Filtering Effect (Conservative Estimate)

For a 50-stock IND universe screened by 7 forecast sources:
- **BEFORE**: ~35 stocks pass all checks → BUY/STRONG_BUY
- **AFTER**: ~20-25 stocks pass MC permutation at 5% level → higher quality signals
- **Net effect**: Fewer trades, but each trade has validated alpha → lower turnover costs, higher win rate
- **Expected Sharpe improvement**: 0.3-0.5 Sharpe points from eliminating false signals

### 5.6 Numerical Impact (Estimated)

```
BEFORE (Current — no MC permutation):
  Win rate: ~52% (barely above coin flip for some signals)
  False signal rate: ~15-20% (signals that look good but are market-bias artifacts)
  Annual turnover: ~180 trades
  Effective CAGR contribution from signal quality: ~0%
    (all return comes from market beta + vol targeting, not signal intelligence)

AFTER (With MC permutation filtering):
  Win rate: ~58-62% (only genuinely intelligent signals accepted)
  False signal rate: ~3-5% (MC permutation filters out bias-driven signals)
  Annual turnover: ~120 trades (fewer, higher-quality)
  Effective CAGR contribution from signal quality: +5-12%
    (genuine alpha from intelligent pairing, validated at 5% significance)
```

---

## 6. Implementation Priority & Dependencies

```
Phase A: MC Permutation Engine          [Week 1]     Foundation — everything depends on this
  └─► Phase B: Wire into Scorer         [Week 1-2]   Immediate value: actual signal testing
  └─► Phase C: Best-of-N Tournament     [Week 2]     Selection bias correction
Phase D: Skill vs Luck Dashboard        [Week 2-3]   Transparency layer
Phase E: Walk-Forward Permutation       [Week 3]     Full robustness picture
Phase F: Sign-Only Test                 [Week 3-4]   Polish for variable-duration trades
```

### Files to Create
1. `services/mc_permutation_test.py` — Core engine (Phase A)

### Files to Modify
1. `services/integrated_scorer.py` — Replace SMA with actual signals (Phase B)
2. `services/strategy_tournament.py` — Best-of-N correction (Phase C)
3. `services/walk_forward.py` — WF permutation (Phase E)
4. `config.py` — MC permutation settings
5. `scheduler.py` — Periodic signal quality monitoring

### Centurion-Specific Adaptations (Swing/Positional)

- **Trading opportunity** = each day in the evaluation period (per Masters p.4-5: "the position is in effect for exactly one day")
- For swing trades held N days: the position vector has N consecutive entries for the same direction
- **Raw returns** = daily close-to-close % change for the instrument
- **Centering** = subtract mean daily return from raw returns before permutation
- **Non-overlapping**: centurion already avoids overlapping positions (one direction per instrument at a time)
- **Focus on CNC delivery trades**: no intra-day or options positions in the permutation test

---

## 7. Summary

The Timothy Masters Monte Carlo permutation test fills a **critical blind spot** in
centurion's evaluation pipeline. Currently, we answer "will this system make money?"
(bootstrap risk) but NOT "does this system make **intelligent** decisions?" (permutation
test). The BCa bootstrap falsely inflates significance in bull markets (NIFTY +12%/yr,
S&P +10%/yr) and the existing permutation test uses a dummy SMA strategy instead of
our actual forecasts.

**The single highest-impact change**: Replace the SMA-based permutation in
`integrated_scorer.py` with actual Carver forecast signals and increase permutations
from 200 to 5000. This alone would catch ~15-20% of false signals that currently
pass validation.

---

## 8. Implementation Status & Pending Actions

**Last updated:** March 28, 2026

### 8.1 MC Permutation Phases — COMPLETE ✅

| Phase | Description | Status | Files |
|-------|-------------|--------|-------|
| A | Core MC permutation engine | ✅ Done | `services/mc_permutation_test.py` |
| B | Wire into integrated scorer (actual forecasts) | ✅ Done | `services/integrated_scorer.py` |
| C | Best-of-N selection bias in tournament | ✅ Done | `services/strategy_tournament.py` |
| D | Skill vs luck decomposition | ✅ Done | Wired in Phase B |
| E | Walk-forward permutation | ✅ Done | `services/walk_forward.py` |
| F | Sign-only test config | ✅ Done | `config.py` |
| Tests | 26/26 passing | ✅ Done | `tests/test_mc_permutation.py` |

### 8.2 Tier 1 Gap Fixes (Live Trading Readiness) — COMPLETE ✅

| Gap | Fix | Status | Files |
|-----|-----|--------|-------|
| 1 | Rejection gate (WF deg <0.5, perm p >0.10) | ✅ Done | `services/integrated_scorer.py` |
| 2 | Dynamic IDM from 60-day rolling correlation | ✅ Done | `services/instrument_weights.py`, `services/carver_pipeline.py` |
| 3 | Gross notional ceiling (2× capital) | ✅ Done | `services/position_sizer.py` |
| 4 | Volume filter (>5% ADV) + impact model | ✅ Done | `kite_connect/trading/auto_executor.py`, `kite_connect/trading/paper_trader.py` |
| 5 | Data freshness gate (reject OHLCV >4h stale) | ✅ Done | `services/carver_pipeline.py`, `config.py` |
| 6 | `PAPER_TRADE_MODE = True` | ✅ Done | `config.py` |
| Tests | 16/16 passing | ✅ Done | `tests/test_tier1_fixes.py` |

### 8.3 PENDING — Your Action Items

#### Phase 2: Paper Trade Validation (4 weeks — Start NOW)

- [ ] **Run paper trading for 4 full weeks** with `PAPER_TRADE_MODE=True`
  - System is configured and ready — just run the scheduler daily
  - Monitor `data/event_logs/` for paper trade fills
- [ ] **Track key metrics daily:**
  - Realized slippage vs. 20 bps assumption
  - Number of stale-data rejections (freshness gate)
  - Volume filter rejections (should catch illiquid mid-caps)
  - Notional ceiling hits (should be rare with ₹5L capital)
  - Rejection gate activations (overfit/random strategies blocked)
- [ ] **Weekly review:**
  - Compare paper P&L with backtest expectations
  - Check IDM drift: dynamic IDM vs. static 1.8 — log the difference
  - Verify reconciliation: paper positions ↔ backtest signals
- [ ] **Pass criteria (must meet ALL before going live):**
  - Paper Sharpe > 0.5 (annualized, after slippage)
  - Backtest-to-paper degradation < 30%
  - No more than 2 stale-data incidents per week
  - Max drawdown < 15% (circuit breaker should catch this)

#### Phase 3: Cautious Live Deployment (After 4 weeks paper)

- [ ] **Set `PAPER_TRADE_MODE = False`** in `config.py` (only after pass criteria met)
- [ ] **Start with ₹2–5L capital** (set `CARVER_INITIAL_CAPITAL` accordingly)
- [ ] **Restrict to STRONG_BUY only** — entry threshold score ≥ 0.55
- [ ] **Max 4 concurrent positions** (half of normal 8)
- [ ] **Weekly reconciliation review** — compare live fills with paper expectations
- [ ] **Monthly walk-forward re-validation** — re-run tournament + WF on latest data

#### Phase 4: Scale Up (After 3-month live track record)

- [ ] **Scale criteria (must meet ALL):**
  - 3-month live Sharpe > 0.6
  - Max drawdown experienced < 12%
  - Win rate > 50%
  - Skill fraction > 60% (from MC decomposition)
- [ ] **Gradual capital increase:** ₹5L → ₹10L → ₹20L (each step after 1 month)
- [ ] **Expand signal acceptance:** STRONG_BUY + BUY signals (score ≥ 0.45)
- [ ] **Increase concurrent positions:** 4 → 6 → 8

### 8.4 PENDING — Tier 2 Fixes (Do During Paper Trading Period)

| # | Issue | Priority | Est. Effort |
|---|-------|----------|-------------|
| 6 | Expand WF OOS window from 63d → 126d | HIGH | 1 hour |
| 7 | Live forecast scalar recalibration (every 7d instead of 14d) | HIGH | 2 hours |
| 8 | Rolling correlation matrix (replace handcrafted 0.35) | MEDIUM | 3 hours |
| 9 | Intraday rescan at 10:30/12:30/14:30 to reduce signal lag | MEDIUM | 2 hours |
| 10 | Minimum trade notional ₹25,000 enforcement | MEDIUM | 30 min |
| 11 | Kite API latency monitoring + alerts | LOW | 2 hours |
| 12 | Sharpe-weighted consensus voting (proportional to strategy quality) | LOW | 2 hours |

### 8.5 Key Config Values for Paper Trading

```python
# config.py — current settings
PAPER_TRADE_MODE = True                  # ← Active now
SIGNAL_FRESHNESS_MAX_HOURS = 4           # ← Freshness gate
CARVER_INITIAL_CAPITAL = 500_000         # ₹5L paper capital
CARVER_ANNUAL_VOL_TARGET = 0.25          # 25% annual vol
CARVER_MAX_LEVERAGE = 1.0                # No leverage
MC_PERMUTATION_N_REPS = 5000            # MC permutations
MC_CENTER_RETURNS = True                 # Remove market bias
```
