# CENTURION R24 GODMODE ACTION PLAN

**Goal**: Sharpe ≥ 1.80, CAGR ≥ 55%, MaxDD ≤ 35%, PBO ≤ 35%
**Created**: 2026-04-14
**Base**: R21A Godmode-v2 (Sharpe 1.571, CAGR 61.0%, MaxDD 31.6%)
**Scope**: Indian equities, swing/positional, long-only, Kite Connect

---

## 1. CURRENT STATE (VERIFIED)

### R21A Godmode-v2 Results

| Metric | Current | Target | Gap | Status |
|--------|---------|--------|-----|--------|
| **Sharpe** | 1.571 | 1.80 | +0.229 | **GAP** |
| **CAGR** | 61.0% | 55% | −6.0pp | **MET** |
| **MaxDD** | 31.6% | 35% | −3.4pp | **MET** |
| **PBO** | 0.0% (flawed) | ≤35% | unknown | **UNVALIDATED** |
| Calmar | 1.93 | — | — | Solid |
| Sortino | 1.74 | — | — | Strong |
| Win Rate | 38.5% | — | — | Acceptable |
| Profit Factor | 1.36 | — | — | Marginal |
| Detrended Sharpe | −0.026 | >0 | **RED FLAG** | Mostly beta |
| Sharpe 90% CI | [1.101, 2.068] | — | — | Lower bound >1 |

**Key insight**: CAGR and MaxDD already exceed targets. The ONLY gap is
**Sharpe +0.229** (14.6% improvement needed). Detrended Sharpe near zero means
most returns come from market drift (long-only in 2012–2025 bull).

### Features ALREADY Active in Godmode-v2

These were verified in `_kaggle_staging/centurion_core/` source code:

| Feature | Status | Evidence |
|---------|--------|----------|
| P4 Regime-Sharpe weights | ✅ ACTIVE | `regime=_eq_regime` at `full_pipeline_backtest.py` L1537 |
| Meta-labeling (20-feat RF) | ✅ ACTIVE | Called at L1543-1556, `META_LABELING_ENABLED=True` |
| Empirical FDM (rolling corr) | ✅ ACTIVE | Wired at L1525-1532, `EMPIRICAL_FDM_ENABLED=True` |
| Multi-timeframe blend | ✅ ACTIVE | Weekly 25% + Monthly 10% at L1258-1305, enabled |
| Cost-aware inertia | ✅ ACTIVE | `COST_AWARE_INERTIA=True`, alpha/cost ratio=2.0 |
| Regime-adaptive vol | ✅ ACTIVE | R21A_REGIME_VOL=True, boost 1.25/defend 0.55 |
| Strategy decay filter | ✅ WIRED | Called in combine_forecasts, but `{}` data → no-op |

### Features VERIFIED NOT Active

| Feature | Status | Evidence |
|---------|--------|----------|
| S13 Vol-regime multiplier | ❌ NOT in backtest | `vol_regime_multiplier` never passed; only in carver_pipeline (live) |
| Harvest DIP_BUYER | ❌ OFF | `_HARVEST_DIP_BUYER = False` at L77; `_set_all_modes_off` doesn't touch it |
| Harvest PROFIT_TAKER | ❌ OFF | `_HARVEST_PROFIT_TAKER = False` at L81 |
| Dynamic Bull Leverage | ⚠ EXISTS but 2.0=2.0 | `LEVERAGE_BULL_CONFIRMED=2.0` (dialed back from 2.5) |
| Inertia | 25% (not 20%) | `CARVER_INERTIA_THRESHOLD=0.25` (raised from 0.20, H5) |

---

## 2. THE SHARPE PROBLEM — WHY +0.23 IS HARD

### Structural Ceiling for Long-Only Indian Equities

The theoretical Sharpe ceiling for a long-only equity system is bounded by:

$$SR_{system} \leq SR_{market} + SR_{alpha}$$

- Indian market Sharpe (NIFTY50, 2012-2025): ~0.70-0.80
- Best systematic equity long-only alpha: +0.50-0.80
- **Theoretical ceiling**: 1.2-1.6 Sharpe

R21A at 1.571 is already near the ceiling for this architecture. The
Detrended Sharpe of −0.026 confirms: nearly ALL of the 1.571 is explained by
being long equities in a bull market, not from timing alpha.

### What World-Class Systems Do Differently

| Fund/System | Sharpe | Key Edge | Architecture |
|-------------|--------|----------|--------------|
| Renaissance Medallion | ~6.0 | Stat-arb, HFT, multi-asset | L/S, 100+ instruments, 10K+ signals |
| AQR Alternatives | 1.5-2.0 | Multi-factor, multi-asset | L/S equities + macro + commodities |
| Carver pysystemtrade | 1.0-1.5 | 200+ futures, diversification | Multi-instrument, trend, carry |
| Two Sigma | 2.0-3.0 | ML + alternative data | L/S, multi-asset, petabytes of data |

Common themes achieving Sharpe >1.8:
1. **Long/Short** capability (profit in both directions)
2. **Multi-asset diversification** (equities + futures + FX + commodities)
3. **Hundreds of instruments** (diversification = free Sharpe)
4. **Ultra-low correlation between signals** (FDM maximization)
5. **Execution optimization** (reducing implementation shortfall)

### What We CAN Do (Without L/S or Multi-Asset)

Given constraints (long-only, Indian equities, swing/positional, Kite Connect):
1. **Expand universe** to NIFTY500 (already in progress)
2. **Wire remaining disabled features** (S13, harvest)
3. **Add genuinely uncorrelated signal styles**
4. **Tighten left tail** (smarter drawdown management)
5. **Fix PBO methodology** (validate robustness)

---

## 3. GAP-BY-GAP ANALYSIS

### Gap A: S13 Vol-Regime Multiplier (Sharpe +0.03-0.05)

**Current**: `combine_forecasts()` accepts `vol_regime_multiplier` parameter
and applies it (caps [0.5, 1.5]). Live pipeline (`carver_pipeline.py` L531-544)
computes and passes it. Backtest does NOT pass it.

**Fix**: In `full_pipeline_backtest.py`, compute per-symbol vol-regime multiplier:
```python
# After computing ohlcv_slice for each symbol:
if len(ohlcv_slice) >= 252:
    rets = ohlcv_slice['Close'].pct_change().dropna()
    rolling_vol = rets.rolling(20).std()
    median_vol = rolling_vol.median()
    current_vol = rolling_vol.iloc[-1]
    _vrm = float(median_vol / current_vol) if current_vol > 0 else 1.0
else:
    _vrm = 1.0
# Pass to combine_forecasts:
combined = combine_forecasts(sym, fc_dict, active_weights,
    forecast_history=_fh_for_sym, regime=_eq_regime,
    vol_regime_multiplier=_vrm)
```

**Impact**: Amplifies forecasts in calm-vol stocks (clearer signals), dampens
in noisy-vol stocks. Reduces false signals in volatile periods.
**Effort**: LOW (10 lines of code)
**Risk**: LOW

### Gap B: Harvest DIP_BUYER (Sharpe +0.03-0.04, CAGR +1-2pp)

**Current**: `_HARVEST_DIP_BUYER = False` (line 77). Logic exists at L1807:
in downtrend (equity < SMA200×0.98), if MR signal is strong, boost vol by 3.33×.

**Fix**: Either change default to `True`, or (better) set it in `run_kaggle.py`
task function:
```python
bt_mod._HARVEST_DIP_BUYER = True
```

**Impact**: Accelerates recovery from bear regimes by leaning into mean-reversion.
**Effort**: TRIVIAL (1 line)
**Risk**: MEDIUM — increases exposure in bear markets, could worsen MaxDD by 1-2pp.
Monitor MaxDD closely.

### Gap C: Harvest PROFIT_TAKER (Sharpe +0.01-0.02, CAGR +1pp)

**Current**: `_HARVEST_PROFIT_TAKER = False` (line 81). Logic at L1925:
in uptrend, tightens stops from 10σ to 6σ for earlier profit booking.

**Fix**: `bt_mod._HARVEST_PROFIT_TAKER = True`

**Impact**: Takes profits earlier in bull regimes. Reduces drawdown from peak
but may also cap upside.
**Effort**: TRIVIAL (1 line)
**Risk**: LOW — tighter stops in bull are conservative.

### Gap D: Inertia 25% → 20% (Sharpe +0.01-0.02)

**Current**: `CARVER_INERTIA_THRESHOLD = 0.25` (raised from 0.20 at H5).
Also `COST_AWARE_INERTIA = True` with `INERTIA_ALPHA_COST_RATIO = 2.0`.

**Fix**: Reduce to 0.20 (was the original value). The cost-aware gate already
handles the cost vs alpha tradeoff, so the 25% fixed threshold is redundant
conservatism.

**Impact**: More responsive to signal changes. With cost-aware inertia active,
net effect is small but positive.
**Effort**: TRIVIAL
**Risk**: LOW (cost-aware gate prevents excessive churn)

### Gap E: University Expansion to NIFTY500 (Sharpe +0.10-0.20)

**THIS IS THE SINGLE BIGGEST LEVER.**

**Current**: NIFTY50 + NEXT50 = ~100 large-cap stocks. Alpha decays fastest
in large-cap where institutional coverage is most intense.

**Why NIFTY500 helps**:
- Mid/small-cap stocks are less efficiently priced → higher per-signal alpha
- More instruments → diversification benefit (even with ρ=0.35 among Indian
  stocks, going from 100→500 gives ~15-25% better risk-adjusted returns)
- Unique sector exposure (chemicals, textiles, etc. not in NIFTY100)
- Lower correlation with NIFTY50-dominated market factor

**Realistic Sharpe impact from universe expansion**:
Using Carver's diversification formula:
$$SR_{portfolio} \approx SR_{avg} \times \frac{\sqrt{N_{eff}}}{\sqrt{N_{eff} \cdot \rho + (1-\rho)}}$$

With N_eff=100→300 (not all 500 will pass liquidity filters), ρ=0.30,
and mid-cap SR premium +30%:
- SR improvement: ~1.15-1.25× multiplier
- 1.57 × 1.15 = 1.81 (meets target)

**Status**: NIFTY500 extraction v9 running on Kaggle (from Day 1126/3861).
Once complete, can build NIFTY500 backtest.

**Effort**: HIGH (extraction in progress, then new backtest run)
**Risk**: MEDIUM (mid-cap liquidity, higher slippage 50-70 bps, data quality)

### Gap F: Add Cross-Sectional Momentum Signal (Sharpe +0.05-0.08)

**Current**: All signals are TIME-SERIES (per-stock absolute signals).
No CROSS-SECTIONAL signal (relative ranking across stocks).

**Why this matters**: Cross-sectional momentum (rank stocks by 12-1 month
return, buy top quartile) is one of the most robust factors in finance
(Jegadeesh & Titman 1993). Crucially, it has LOW correlation with
time-series momentum (ρ ≈ 0.15-0.25), making it an excellent FDM diversifier.

**Implementation**:
```python
# In a new signal source (strategies/cross_sectional_momentum.py):
# For each rebalance day:
#   1. Compute 12-1 month return for all eligible stocks
#   2. Rank stocks cross-sectionally
#   3. Convert rank to forecast: top 20% → +10, bottom 20% → -10 (or 0 for long-only)
#   4. Weight: ~10-15% in signal portfolio
```

**Impact**: Adds genuinely decorrelated alpha. When combined via FDM, boosts
portfolio Sharpe by ~0.05-0.08.
**Effort**: MEDIUM (new signal source, ~100 lines)
**Risk**: LOW (one of most documented anomalies)

### Gap G: Fix PBO Methodology (Validates robustness, no direct Sharpe impact)

**Current**: PBO uses signal data availability (% of days producing forecast)
as proxy for accuracy. Result: PBO=0% (artificially low, unreliable).

**Fix**: Implement per-signal daily P&L attribution:
```
For each day d, for each signal s:
  signal_pnl[s][d] = position_from_signal_s × daily_return
```
Then feed into CSCV algorithm:
- N=10 partitions, combinatorial cross-validation
- Compute PBO as fraction of synthetic permutations that outperform

**Impact**: Validates ALL other improvements. Without valid PBO, any Sharpe
improvement could be overfitting.
**Effort**: HIGH (significant changes to backtest engine)
**Risk**: LOW (doesn't change trading, only validates)

---

## 4. PHASED IMPLEMENTATION PLAN

### Phase R24A: Quick Wins (Effort: 2-3 days, Expected: Sharpe +0.08-0.12)

Enable verified disabled features in backtest. No architectural changes.

| Step | Change | File | Expected |
|------|--------|------|----------|
| 1 | Wire S13 `vol_regime_multiplier` in backtest loop | `full_pipeline_backtest.py` | Sharpe +0.04 |
| 2 | Enable `_HARVEST_DIP_BUYER = True` | `run_kaggle.py` task or `full_pipeline_backtest.py` | Sharpe +0.03 |
| 3 | Enable `_HARVEST_PROFIT_TAKER = True` | Same | Sharpe +0.02 |
| 4 | Reduce `CARVER_INERTIA_THRESHOLD` 0.25 → 0.20 | `config.py` | Sharpe +0.01 |

**Run as R24A on Kaggle. Validation gates**:
- Sharpe ≥ 1.65 (improvement from 1.57)
- CAGR ≥ 55%
- MaxDD ≤ 34%
- If gates fail: revert individual features one at a time to isolate

**Projected after R24A**: Sharpe ~1.65-1.69

### Phase R24B: Cross-Sectional Momentum (Effort: 1 week, Expected: Sharpe +0.05-0.08)

Add genuinely new decorrelated signal style.

| Step | Change | Expected |
|------|--------|----------|
| 1 | Create `strategies/cross_sectional_momentum.py` | New signal source |
| 2 | Register in `forecast_combiner.py` DEFAULT_FORECAST_WEIGHTS | Weight: 10% |
| 3 | Reduce EWMAC family total weight from ~32% to ~25% (free up capacity for XS-MOM) | Rebalance |
| 4 | Run R24B backtest | Validate decorrelation |

**Validation gates**:
- Sharpe ≥ 1.72
- Correlation(XS-MOM, EWMAC) < 0.30
- PBO (even with flawed method) < 35%

**Projected after R24B**: Sharpe ~1.72-1.77

### Phase R24C: Universe Expansion to NIFTY500 (Effort: 2-3 weeks, Expected: Sharpe +0.10-0.20)

This is the structural change that crosses the 1.80 threshold.

| Step | Change | Expected |
|------|--------|----------|
| 1 | Complete NIFTY500 extraction (v9 in progress) | Prerequisite |
| 2 | Implement liquidity filter (min ₹1cr daily turnover) | Data quality |
| 3 | Apply tiered cost model (NIFTY50: 27bps, NEXT50: 42bps, Mid200: 55bps, Small: 72bps) | Realistic costs |
| 4 | Run R24C backtest with expanded universe | Full validation |

**Validation gates**:
- Sharpe ≥ 1.80
- CAGR ≥ 55%
- MaxDD ≤ 35%
- Turnover-adjusted Sharpe ≥ 1.50

**Projected after R24C**: Sharpe ~1.80-2.00

### Phase R24D: PBO Fix (Effort: 1 week, Validates robustness)

| Step | Change |
|------|--------|
| 1 | Add per-signal daily P&L tracking to backtest engine |
| 2 | Implement CSCV algorithm with N=10 partitions |
| 3 | Compute valid PBO for R24C configuration |
| 4 | If PBO > 35%: identify and reduce overfit signals |

**Gate**: PBO ≤ 35% using valid methodology

---

## 5. CUMULATIVE PROJECTIONS

| Phase | Sharpe | CAGR | MaxDD | Confidence |
|-------|--------|------|-------|------------|
| **R21A (current)** | 1.571 | 61.0% | 31.6% | ✅ Measured |
| **R24A (quick wins)** | ~1.65-1.69 | ~62-64% | ~32-33% | 80% confidence |
| **R24B (+XS-MOM)** | ~1.72-1.77 | ~63-66% | ~32-33% | 70% confidence |
| **R24C (+NIFTY500)** | ~1.80-2.00 | ~65-80% | ~30-35% | 60% confidence |

**Realistic scenario (60% realization rate)**:
- After all phases: Sharpe ~1.75-1.85, CAGR ~62-72%, MaxDD ~31-34%

---

## 6. RISKS AND MITIGATIONS

### Risk 1: Detrended Sharpe is −0.026 (Mostly Market Beta)

**What it means**: Strip out market drift and there's essentially zero alpha.
A bear market or mean-reverting period would expose this.

**Mitigation**:
- Cross-sectional momentum is MARKET-NEUTRAL by construction (long top,
  avoid bottom) — adds genuine alpha beyond beta
- Universe expansion captures more idiosyncratic alpha (less correlated with
  NF50 beta)
- Monitor Detrended Sharpe after each phase — must improve toward >0

### Risk 2: Overfitting (PBO Unknown)

**What it means**: Without valid PBO, every improvement might be noise.

**Mitigation**:
- Phase R24D fixes PBO methodology
- Apply "Deflated Sharpe Ratio" (Bailey & López de Prado 2014) to adjust
  for multiple testing
- Walk-forward: require train/test SR gap < 0.35

### Risk 3: Mid-Cap Liquidity (NIFTY500)

**What it means**: Small-cap stocks may have 50-70 bps round-trip costs,
wide spreads, and execution gaps.

**Mitigation**:
- Apply strict liquidity filter (min ₹1cr daily turnover)
- Tiered slippage model already exists in backtest
- Position sizing via Carver formula naturally reduces small-cap exposure
  (higher vol → smaller positions)

### Risk 4: CAGR Regression

**What it means**: Currently CAGR 61% exceeds target by 6pp, but tighter
risk management (stop tightening, inertia changes) could reduce it.

**Mitigation**:
- CAGR headroom is buffer (target 55%, current 61%)
- Harvest PROFIT_TAKER intentionally trades CAGR for Sharpe (takes profit
  earlier → smoother returns)
- Monitor CAGR ≥ 55% gate at every phase

---

## 7. NEW ALPHA SOURCES — CANDIDATES FOR R24B+

Priority ranked by (decorrelation × evidence × implementability):

| Rank | Signal | Correlation with Existing | Evidence | Data Source | Effort |
|------|--------|--------------------------|----------|-------------|--------|
| 1 | **Cross-sectional momentum** (12-1mo relative strength) | ρ=0.15-0.25 with TS-MOM | Jegadeesh & Titman 1993, Asness 2013 | yfinance | MEDIUM |
| 2 | **Quality factor** (ROE + low leverage + earnings stability) | ρ=0.05-0.15 | Novy-Marx 2013, AQR | BSE quarterly filings | HIGH |
| 3 | **Short-term reversal** (5-day reversion) | ρ=−0.20 with trend | Lo & MacKinlay 1990 | yfinance | MEDIUM |
| 4 | **Options skew signal** (VIX + P/C ratio) | ρ=0.10 | Xing et al. 2010 | NSE Options data | HIGH |
| 5 | **Institutional flow** (FII+DII net buying) | ρ=0.15-0.20 | India-specific, SEBI data | nse-india.com | MEDIUM |

**Recommendation**: Implement #1 (cross-sectional momentum) in R24B.
Consider #2 (quality) and #5 (institutional flow) for future phases.

---

## 8. HONEST ASSESSMENT

**Can we reliably achieve Sharpe ≥ 1.80?**

- **Best case (all phases succeed)**: YES — Sharpe 1.80-2.00. Universe
  expansion is the decisive factor. If NIFTY500 works as expected with
  proper liquidity filters, the diversification benefit alone could push
  past 1.80.

- **Most likely case (partial wins)**: Sharpe 1.70-1.85. Quick wins give
  +0.08-0.12, cross-sectional adds +0.05, universe expansion in mid-cap
  stocks adds +0.05-0.15 depending on liquidity and data quality.

- **Worst case (mid-cap disappoints)**: Sharpe 1.65-1.70. If mid-cap alpha
  is eaten by higher costs, we plateau. System still beats R23 targets
  (Sharpe ≥ 1.5) but misses Godmode.

**The uncomfortable truth**: A long-only Indian equity swing system with
Sharpe ≥ 1.80 sustained over 13 years is in the top 0.1% of documented
systematic strategies. The systems that consistently exceed this (Medallion,
Two Sigma) use L/S, multi-asset, HFT, and alternative data at massive scale.
1.80 is achievable but requires near-perfect execution of all phases.

**The optimistic counter**: Indian mid-cap is genuinely less efficient than
developed-market large-cap. The alpha opportunity in NIFTY500 is real and
under-exploited by institutional quant strategies. If the extraction (v9)
produces clean data and the signals generalize to mid-cap, Sharpe 1.80+ is
plausible.

---

## 9. IMMEDIATE NEXT STEPS

1. **Wait for NIFTY500 extraction v9** to complete (currently Day 1126/3861)
2. **Implement R24A quick wins** (S13 wiring + harvest flags + inertia)
3. **Push R24A to Kaggle** as V58 backtest
4. **Evaluate results** against Phase R24A gates
5. **If gates pass**: proceed to R24B (cross-sectional momentum)
6. **If gates fail**: diagnose which feature hurt, revert selectively
