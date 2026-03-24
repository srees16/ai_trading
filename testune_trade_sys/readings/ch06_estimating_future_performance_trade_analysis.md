# Chapter 6: Estimating Future Performance — Trade-Based Analysis

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

Rather than measuring system-level returns, analyse the **distribution of
individual trades** to build confidence intervals around future
performance.

### Profit Analysis

Compute per-trade P&L and summarise:
- Mean, median, standard deviation
- Win rate, average win / average loss ratio
- Profit factor = gross wins / gross losses

### Parametric Confidence Intervals

If trade returns are approximately Normal:
- Use Student-*t* intervals for the mean trade return
- Wider intervals ⇒ less certainty about the true expectation

### BCA Bootstrap Confidence Intervals

The bias-corrected and accelerated (BCa) bootstrap relaxes the
Normality assumption:

1. Draw *B* bootstrap samples (with replacement) of trade returns
2. Compute the statistic (e.g., mean) for each sample
3. Adjust the percentile interval for **bias** (median shift) and
   **acceleration** (skewness)
4. Result: a more accurate confidence interval than the simple percentile
   method

C++ source: `BOOT_CONF.CPP` (pp. 232-236).

### Lower-Bound Mean Return

Estimate the smallest **plausible** mean trade return at a given
confidence level using the bootstrap lower bound.

### Drawdown Analysis

- Compute peak-to-trough drawdowns from the equity curve
- Use bootstrap resampling of bar returns to estimate the **maximum
  drawdown** at a given confidence level
- C++ source: `DRAWDOWN` program (pp. 272-282)

### Key Takeaways
1. Trade-level analysis gives a finer-grained view than total return alone
2. BCa bootstrap handles skewness and bias automatically
3. Drawdown bounds tell you the *worst realistic loss* to expect
4. Always pair point estimates with confidence intervals
