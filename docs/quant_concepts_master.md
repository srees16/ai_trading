# Quantitative Concepts Master Reference
## Consolidated from 16 PDFs in `rag_pipeline/ingest_docs/`
> Generated: 2026-04-01 | Centurion Core Audit

---

## 1 — Signal Generation & Indicators

| ID | Concept | Source PDFs | Description |
|----|---------|-------------|-------------|
| SG-01 | EWMAC (Exponentially Weighted Moving Average Crossover) | Carver AFTS, Carver ST | Multiple speed variations (2/8 through 64/256). S12 adjusted trend, S17 normalised trend. Forecast scaled to avg\|f\|≈10, capped ±20. |
| SG-02 | Carry (Roll Yield) | Carver AFTS, Carver ST | Expected return from holding — carry = (div_yield − funding_cost) for equities, roll yield for futures. Vol-adjusted, scaled. |
| SG-03 | Value (Slow Mean Reversion) | Carver AFTS | 5-year deviation from mean → negated → forecast. Captures multi-year reversion. |
| SG-04 | Acceleration | Carver AFTS (S23) | Rate of change of EWMAC signal. Catches trend reversals 3-5 days early. |
| SG-05 | Skew Signal | Carver AFTS (S24) | Realized skew risk premium — sell negative-skew assets, buy positive-skew. |
| SG-06 | Breakout / Channel | Carver AFTS, Penfold, Kaufman, Murphy | Turtle-style channel breakout (N-day). Donchian 5/20, Four-Week Rule. |
| SG-07 | Cross-Sectional Momentum | Jegadeesh-Titman (1993), Kaufman | 12-month formation, 1-month skip. Rank and overweight winners. |
| SG-08 | Meta-Labeling | de Prado AFML Ch.3 | Binary ML layer: primary model predicts side, meta-model predicts size (probability of win). Triple-barrier labels. |
| SG-09 | Hilbert Transform / MAMA-FAMA | Ehlers Cybernetic Analysis, Rocket Science | InPhase/Quadrature → dominant cycle period → adaptive MA (MAMA). Anticipates turns. |
| SG-10 | Fisher Transform | Ehlers Cybernetic Analysis | Converts any indicator to near-Gaussian for sharp turning-point signals. |
| SG-11 | Super Smoother Filter | Ehlers Rocket Science | 2/3-pole Butterworth design. Nearly flat response to cutoff then sharp attenuation. |
| SG-12 | Cyber Cycle / Sinewave / Adaptive RSI | Ehlers Cybernetic Analysis | Cycle-mode extraction from price, phase-based entries, dominant-cycle-adaptive RSI. |
| SG-13 | Intermarket Analysis | Ruggiero Cybernetic Trading | Cross-market correlation signals (VIX ↔ equities, bonds ↔ rates, gold ↔ USD). Divergence engine. |
| SG-14 | Seasonal Patterns | Ruggiero, Kaufman | Day-of-week, month, Ruggiero/Barna Seasonal Index. Calendar-based bias. |
| SG-15 | PEAD (Post-Earnings Announcement Drift) | Academic | Earnings surprise → drift over 45-60 days. SUE threshold. |
| SG-16 | Sentiment (NLP) | FinBERT | News sentiment via transformer model → z-score → Carver-scaled forecast. |
| SG-17 | Pairs Trading / Cointegration | Aronson, Kaufman | Engle-Granger cointegration → z-score spread entry/exit. |
| SG-18 | Open Interest / Delivery Volume | NSE-specific | F&O OI analysis (long/short buildup), delivery % for institutional conviction. |
| SG-19 | Neural Networks / Genetic Algorithms | Ruggiero | NN for parameter optimization, GA for rule discovery. |
| SG-20 | Fractional Differentiation | de Prado AFML Ch.5 | Preserves memory while achieving stationarity; d-value tuned via ADF test. |
| SG-21 | Pattern Recognition | Murphy, Kaufman | Chart patterns (H&S, double top/bottom, triangles), candlestick patterns. |
| SG-22 | Equity Curve Filtering | Penfold, Ruggiero | Trade only when strategy equity is above its MA. Self-referential filter. |

---

## 2 — Statistical Validation & Overfitting Prevention

| ID | Concept | Source PDFs | Description |
|----|---------|-------------|-------------|
| SV-01 | White's Reality Check (WRC) | Aronson EBTA Ch.6 | Bootstrap null distribution for best-of-N signal selection. Corrects data-snooping bias. |
| SV-02 | Benjamini-Hochberg FDR | Aronson EBTA Ch.6 | Controls expected proportion of false discoveries among rejected hypotheses at level q. |
| SV-03 | Monte Carlo Permutation Test | Masters MC Eval, Masters Testing & Tuning | Shuffle position-return pairings → null distribution. Selection bias correction for best-of-many. |
| SV-04 | Deflated Sharpe Ratio (DSR) | de Prado AFML Ch.14 | Adjusts Sharpe for multiple testing: PSR = Z[(SR − SR*)√(n−1) / √(1 − γ₃SR + (γ₄−1)/4·SR²)]. |
| SV-05 | Combinatorial Purged Cross-Validation (CPCV) | de Prado AFML Ch.12 | C(N,k) combinations of purged CV folds → multiple backtest paths. |
| SV-06 | Purged K-Fold CV + Embargo | de Prado AFML Ch.7 | Remove training observations overlapping test labels. Embargo after train window. |
| SV-07 | Walk-Forward Optimization | Carver ST, Masters, Ruggiero | Rolling reoptimization: train on IS, test on OOS. Degradation ratio = OOS/IS Sharpe. |
| SV-08 | Bootstrap Confidence Intervals | Masters Testing & Tuning, Aronson | Percentile/pivot bootstrap for bounding future performance. Block bootstrap for dependent data. |
| SV-09 | Data-Mining Bias Estimation | Aronson EBTA Ch.6 | Bias ≈ σ√(2·ln(N)) for best-of-N selection. Markowitz/Xu correction factor. |
| SV-10 | t-Statistic Gate | Aronson EBTA Ch.5 | Signal must have t ≥ 2.0 (p < 0.05) to be considered. Two-sided test on excess returns. |
| SV-11 | Minimum Backtest Length | de Prado AFML Ch.14 | Required number of observations to trust a given Sharpe ratio estimate. |
| SV-12 | Signal Fire Count | Aronson EBTA Ch.5 | Min ~30 independent signal transitions for reliable statistics. Sample ramp function. |
| SV-13 | Detrended Returns | Aronson EBTA Ch.1 | Zero-centre returns by subtracting rolling mean → isolate timing skill from market drift. |
| SV-14 | Trimmed/Winsorized Metrics | Aronson EBTA Ch.5 | Remove top/bottom 5% of returns for robust Sharpe estimation. |
| SV-15 | ROC / Confusion Matrix | Masters Assessing | Precision, recall, F1, AUC-ROC for classifier evaluation. |
| SV-16 | Chi-Square Goodness of Fit | Vince Math MM, Ruggiero | Test whether return distribution matches assumed parametric form. |
| SV-17 | K-S Test | Vince Math MM | Kolmogorov-Smirnov test for distribution fitting validation. |
| SV-18 | Serial Correlation Tests | Vince Math MM, Masters MC Eval | Runs test for trade dependency detection. Impacts null distribution in permutation tests. |
| SV-19 | Nested Cross-Validation | Masters Assessing | Proper hyper-parameter tuning without leakage — inner CV for tuning, outer for evaluation. |
| SV-20 | Optimization Bias Decomposition | Masters Testing & Tuning | Pre-optimization + optimization + post-optimization bias estimation. Training vs selection bias. |

---

## 3 — Strategy Robustness & Filtering

| ID | Concept | Source PDFs | Description |
|----|---------|-------------|-------------|
| SR-01 | Forecast Diversification Multiplier (FDM) | Carver AFTS/ST | FDM = 1/√(w'Σw). Compensates for correlation between forecast sources. |
| SR-02 | Instrument Diversification Multiplier (IDM) | Carver AFTS/ST | Scales up exposure for diversified portfolio. IDM ≈ 1.0 (1 instrument) to 2.5 (10+). |
| SR-03 | Cost Speed Limit | Carver AFTS Ch.12 | Don't trade if expected Sharpe < speed_limit × cost. Prevents over-trading. |
| SR-04 | Position Inertia / Buffering | Carver ST | Only change position if new target differs by > 10% from current. Reduces turnover. |
| SR-05 | Forecast Capping | Carver ST | Cap all forecasts at ±20 (2× average). Prevents extreme positions. |
| SR-06 | Handcrafted Weights | Carver ST | Prefer hand-set weights over optimized to avoid in-sample overfitting. |
| SR-07 | Parsimony / Occam's Razor | Aronson, Masters, Penfold | Fewer parameters = more robust. Ideal: 2-4 indicator variables. |
| SR-08 | Parameter Sensitivity Analysis | Masters Testing & Tuning, Kaufman | Visualize performance across parameter sweeps. Flat plateau = robust. |
| SR-09 | Equity Curve Stability | Penfold | Universe of alternative equity curves with upper/lower bands. R² of equity. |
| SR-10 | Survivorship Bias Prevention | Carver ST, de Prado, Kaufman | Track and correct for delisted/removed tickers. |
| SR-11 | Strategy Decay Detection | Kaufman, Penfold | Rolling Sharpe degradation → DEGRADED/DEAD classification → auto-reallocation. |
| SR-12 | Strategy Tournament | Carver ST | Monthly rank strategies by composite score, auto-disable losers. |

---

## 4 — Risk Management & Position Sizing

| ID | Concept | Source PDFs | Description |
|----|---------|-------------|-------------|
| RM-01 | Volatility Targeting | Carver AFTS/ST | Set annual vol target (e.g., 20%), scale positions to achieve it. |
| RM-02 | Half-Kelly Criterion | Carver ST, Vince | Optimal bet size = Kelly/2 for safety margin. τ = SR/2. |
| RM-03 | Optimal f / Leverage Space Model | Vince (both books) | Fraction maximizing geometric growth (TWR). Safe f = DD-constrained version. |
| RM-04 | Risk of Ruin (ROR) | Vince, Penfold | P(equity falls below threshold). Target: ROR = 0%. |
| RM-05 | Hierarchical Risk Parity (HRP) | de Prado AFML Ch.16 | Tree-based allocation: hierarchical clustering → quasi-diagonalization → recursive bisection. |
| RM-06 | Maximum Drawdown Management | All | DD thresholds: warning → critical → halt. Circuit breaker patterns. |
| RM-07 | Trailing Stops (Volatility-Based) | Carver, Ehlers, Kaufman | Swing = 2.5σ, positional = 3.5σ. Profit-lock at 4σ. |
| RM-08 | Sector/Concentration Limits | Carver ST, Penfold | Max per-sector allocation (e.g., 30%). HHI concentration score. |
| RM-09 | Regime-Adaptive Leverage | Carver AFTS, Ruggiero | Scale leverage by regime (bull → higher, crisis → lower). |
| RM-10 | Margin Management | Vince Leverage Space | Alert at 80% utilisation, halt at 90%. Pre-order margin checks. |
| RM-11 | Correlation Risk | Carver ST, de Prado | Monitor pair correlations, diversification ratio. Reduce when corrs spike. |
| RM-12 | Tail Risk Hedging | Natenberg, Carver | Portfolio puts at 5% OTM. Trigger on VIX spike or DD threshold. |
| RM-13 | Options Overlay (Income) | Natenberg | Covered calls (30-delta) + cash-secured puts (25-delta) for income. |
| RM-14 | Greeks-Based Risk | Natenberg | Delta, gamma, theta, vega exposure monitoring and hedging. |

---

## 5 — Performance Evaluation

| ID | Concept | Source PDFs | Description |
|----|---------|-------------|-------------|
| PE-01 | Sharpe Ratio | All | Primary metric. Annualized: SR × √252. |
| PE-02 | Sortino Ratio | Kaufman, Penfold | Downside-only volatility in denominator. |
| PE-03 | Calmar Ratio | Kaufman, Penfold | CAGR / max drawdown. |
| PE-04 | Omega Ratio | Academic | Probability-weighted gains vs losses above threshold. |
| PE-05 | Information Ratio | Kaufman | Alpha / tracking error vs benchmark. |
| PE-06 | Ulcer Index | Penfold | Duration-weighted drawdown severity. |
| PE-07 | Profit Factor | Ehlers, Penfold, Kaufman | Gross profits / gross losses. |
| PE-08 | Expectancy | Penfold | Win_rate × avg_win − loss_rate × avg_loss. Per-trade expected value. |
| PE-09 | CVaR (Conditional VaR) | de Prado, Masters | Expected loss beyond VaR threshold. Tail risk measure. |
| PE-10 | Geometric Mean (TWR) | Vince (both) | Product of (1 + HPR). The TRUE compounded growth metric. |
| PE-11 | Jensen's Alpha | Penfold, Murphy | Excess return beyond CAPM prediction. |
| PE-12 | Recovery Factor | Kaufman | Net profit / max drawdown. Higher = faster recovery. |

---

## 6 — Backtesting Methodology

| ID | Concept | Source PDFs | Description |
|----|---------|-------------|-------------|
| BT-01 | Expanding/Rolling Window | Carver ST, Masters | Train on data[0:t], test at t+1. Expanding (growing IS) or rolling (fixed IS). |
| BT-02 | Transaction Cost Modeling | Carver ST, Kaufman | Include commissions + spread + slippage + market impact. |
| BT-03 | Backtest is NOT Research | de Prado AFML Ch.11 | Backtest validates; research discovers. Never mine data with backtests. |
| BT-04 | Look-Ahead Bias Prevention | de Prado, Aronson | No future information in decisions. Strict temporal ordering. |
| BT-05 | Paper-Live Reconciliation | Carver ST | Compare paper trades vs live fills. Track slippage, divergence. |

---

## Concept Count by Source

| PDF | Concepts Contributed | Primary Domain |
|-----|---------------------|----------------|
| Carver AFTS (2024) | SG-01..05, SR-01..06, RM-01..02, BT-01..02 | Signal + Framework |
| Carver ST (2015) | SG-01..02, SR-04..06, RM-01..02, SV-07 | Framework |
| de Prado AFML (2018) | SG-08, SG-20, SV-04..06, RM-05 | ML + Validation |
| Aronson EBTA (2006) | SV-01..02, SV-09..14 | Statistical Testing |
| Masters MC Eval | SV-03, SV-18 | Permutation Tests |
| Masters Testing & Tuning | SV-03, SV-08, SV-19..20, SR-08 | Testing |
| Masters Assessing | SV-15, SV-19 | Classification |
| Ehlers Cybernetic Analysis | SG-09..12, SG-11 | DSP Indicators |
| Ehlers Rocket Science | SG-09, SG-11 | DSP Theory |
| Ruggiero Cybernetic | SG-13..14, SG-19 | AI/Intermarket |
| Vince Leverage Space | RM-03..04, RM-10 | Position Sizing |
| Vince Math of MM | RM-03, SV-16..18, PE-10 | Money Management |
| Penfold | SG-06, SG-22, SR-09, SR-11, PE-06..08 | Trend Tactics |
| Kaufman T.S.&M. | SG-06..07, SG-14, SG-21, SR-08, PE-01..05 | Encyclopedia |
| Murphy Tech Analysis | SG-06, SG-21 | Classic TA |
| Natenberg Options | RM-12..14 | Options/Volatility |

**Total unique concepts: 70** (22 signal, 20 validation, 12 robustness, 14 risk, 12 performance, 5 backtesting — some overlap across categories).
