# Signal Insights & Recommendations

*Generated: 2026-04-01 16:08*

## 1. Regime-Specific Signal Failures

### BULL: Performing Well (Hit Rate = 51.1%, Sharpe = 0.26)

### BEAR: Underperforming (Hit Rate = 46.3%)
- Avg return: -0.43%
- False signal rate: 40.4%
- Sharpe: -0.26
- BUY hit rate: 53.0% vs SELL: 37.5%
  → **SELL signals are weaker** in BEAR

### SIDEWAYS: Performing Well (Hit Rate = 54.1%, Sharpe = 0.24)

## 2. Overfitting Indicators

- Overfitting haircut applied: 80.7%
- Conservative CAGR after haircut: +8.9%

## 3. Weak Pipeline Components

- **BEAR**: Hit rate = 46.3%, PF = 0.83, False% = 40.4%
- Low confidence signals (<0.3): Hit rate = 45.6%, Sharpe = -0.33
  → **Filter low-confidence signals** — they add noise, not alpha

## 4. Stress Test Results

| Scenario | N | Hit Rate | Avg Ret | Sharpe | PF | Max DD |
|----------|---|----------|---------|--------|----|--------|
| High Volatility (vol_z>1.5)              |   874 |  63.6% |  +2.25% |  1.11 |  2.27 |  84.0% |
| Extreme Bear (trend<-5%, vol_z>1)        |   102 |  38.2% |  -1.93% | -1.48 |  0.29 |  86.6% |
| Low Confidence (<0.3)                    |  1956 |  45.6% |  -0.57% | -0.33 |  0.79 | 100.0% |
| First Year (early signals)               |  3303 |  46.7% |  -0.27% | -0.15 |  0.90 | 100.0% |
| Last Year (OOS proxy)                    |  3238 |  48.0% |  -0.35% | -0.20 |  0.86 | 100.0% |

## 5. Recommendations

1. **Regime-Adaptive Position Sizing**: Reduce position sizes by 60-70% in BEAR regime. Current BEAR Sharpe (-0.26) suggests the system's trend-following signals are partially offset by whipsaw losses.

2. **Confidence Threshold Filter**: Raise minimum forecast threshold from 2.0 to 5.0 to eliminate weak signals. Low-confidence signals show Sharpe = -0.33.

3. **Optimal Holding Period**: 5D shows the best Sharpe (0.26). Consider calibrating position holding to this horizon for maximum risk-adjusted returns.

4. **Reduce Strategy Variants**: The overfitting haircut (81%) is high. Consider reducing the number of forecast sources from 22 to ~12 (drop lowest Sharpe contributors) to reduce data-mining bias.

5. **Drawdown Circuit Breaker**: Max DD = 45.7%. Implement equity curve filter — halt new trades when equity drops below 63-day SMA to limit tail risk.

## 6. Final Assessment

| Metric | Value |
|--------|-------|
| Ideal CAGR | +45.9% |
| Realistic CAGR | +45.6% |
| Conservative CAGR | +8.9% |
| CAGR 90% CI | [+4.8%, +122.0%] |
| Overfitting Risk | HIGH (81%) |

**Bottom line**: The defensible CAGR range for centurion_core is **+8.9% to +45.6%** with 90% confidence bounds of [+4.8%, +122.0%].