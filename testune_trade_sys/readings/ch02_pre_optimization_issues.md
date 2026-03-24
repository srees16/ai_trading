# Chapter 2: Pre-optimization Issues

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

Before optimising a trading system, we must assess the quality of our data
and indicators.  This chapter covers stationarity and entropy — two
properties that profoundly affect whether a trained system will generalise.

### Assessing and Improving Stationarity

**Why it matters:** A trading system trained under one market regime (e.g.,
low volatility uptrend) may fail in another (high volatility downtrend).
If the indicator values wander slowly between regimes, training data will
be dominated by whichever regime happened to prevail, leading to fragile
models.

**The STATN Program:**
- Computes trend (least-squares slope) and volatility (ATR)
- Classifies each bar as above or below the median
- Tracks "gap sizes" — how many consecutive bars remain in the same state
- Ideal: many short gaps (frequent regime switches)
- Dangerous: few very long gaps (extended single-regime periods)

**Improving Stationarity:**
- **Oscillating (Version 1):** Subtract the lagged indicator value →
  converts trending indicator into a mean-reverting oscillator
- **Extreme induction (Version >1):** Subtract a long-window average
  to force faster mean reversion

### Measuring Indicator Information with Entropy

**Relative entropy** (Eq 2-1) measures how uniformly an indicator's
values are distributed across its range:

$$H = -\frac{1}{\log(B)} \sum_{i=1}^{B} p_i \log(p_i)$$

where $B$ = number of bins and $p_i$ = proportion of data in bin $i$.

- High entropy (→1): roughly uniform distribution → maximum information
- Low entropy (→0): concentrated in few bins → low information, often
  caused by outliers

**Improving Entropy:**
- Remove or compress outliers in the tails
- **Monotonic tail-only cleaning**: replace extreme values with linearly
  spaced values that preserve ordering but reduce spread

### Key Takeaways
1. Always check stationarity before training a model
2. Prefer oscillating (differenced) indicators when trends are slow
3. High-entropy indicators train better predictive models
4. Outlier cleaning improves entropy and model quality
