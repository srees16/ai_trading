# CAGR Estimation — Centurion Core

*Generated: 2026-04-01 16:08*
*Market: IND | Backtest period: 3.9 years*

## CAGR Summary

| Metric | Ideal | Realistic | Conservative |
|--------|-------|-----------|--------------|
| **CAGR** | +45.9% | +45.6% | +8.9% |
| **Sharpe** | 1.06 | 1.05 | 0.04 |
| **Max DD** | 45.7% | 45.8% | 59.5% |

## Definitions

- **Ideal**: Raw backtest returns (includes base transaction costs + slippage)
- **Realistic**: Ideal + additional execution friction (market impact, partial fills, timing delays)
- **Conservative**: Realistic × walk-forward degradation (0.65) − data-mining bias haircut

## Statistical Confidence

- 90% Bootstrap CI for CAGR: [+4.8%, +122.0%]
- Overfitting haircut: 80.7%
- Block bootstrap: 2000 simulations, block size = 21 days

## Portfolio Backtest Details

- Starting capital: 500,000
- Final equity: 2,153,347
- Total trades: 2,185
- Avg positions: 8.2
- Transaction costs: 0
- Max drawdown duration: 374 days

## Regime-Conditioned Performance

| Regime | Ann. Return | Sharpe | Max DD |
|--------|-------------|--------|--------|
| BULL       | +61.0% | 1.31 | 33.4% |
| BEAR       | +22.2% | 0.65 | 32.5% |
| SIDEWAYS   | +44.0% | 1.05 | 29.7% |

## Methodology

1. **No look-ahead bias**: Expanding window, signals generated using only past data
2. **Walk-forward validation**: 252-day train / 63-day test rolling windows
3. **Position sizing**: Volatility-targeted (Carver AFTS), regime-adaptive leverage
4. **Costs**: Commission + slippage modeled per-trade
5. **Conservative haircut**: Accounts for data-mining bias (22 strategy variants tested),
   walk-forward degradation ratio, and expected max Sharpe under null hypothesis
6. **Bootstrap CI**: Block bootstrap (21-day blocks) preserves autocorrelation structure