# Chapter 5: Estimating Future Performance — Unbiased Methods

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

Standard back-test metrics are **biased high** because the
optimiser chooses parameters that happen to fit in-sample noise.
This chapter covers four unbiased estimation methods.

### Walk-forward Analysis (WFA)

Split the history into successive **train / test** segments:

1. Optimise on window *[0 … T]*
2. Paper-trade on *[T+1 … T+k]*
3. Slide the window forward; repeat

The concatenated out-of-sample returns give a realistic performance
estimate free of look-ahead bias.

### Trading-System Cross-Validation

Adapt k-fold cross-validation for temporal data:
- Use a blocked design so folds respect time order
- Purge rows that leak future information between train / test
- Average fold metrics — variance indicates robustness

### CSCV (Combinatorially Symmetric Cross-Validation)

From López de Prado & Bailey:
- Split data into *S* equal blocks
- Enumerate all *S/2*-combinations for the in-sample set;
  the remaining blocks form the out-of-sample set
- Count in how many splits the in-sample winner also
  beats the median out-of-sample → **Probability of
  Backtest Overfitting (PBO)**
- PBO > 0.5 → the system is likely overfit

### Nested Walk-forward

A two-level walk-forward design:
- **Outer loop** selects macro-parameters (e.g., which features to use)
- **Inner loop** optimises micro-parameters (e.g., look-back lengths)
- Only the inner OOS return is collected

This prevents meta-overfitting of the outer loop.

### Key Takeaways
1. Never trust a single back-test number
2. Walk-forward is the industry-standard baseline
3. CSCV gives a probability of overfitting, not just a point estimate
4. Nesting prevents the outer search from leaking information
