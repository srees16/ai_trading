# Chapter 4: Post-optimization Issues

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

After optimising a trading system, we need to estimate how much of the
apparent performance is due to genuine predictive power vs overfitting.

### StocBias — Cheap Bias Estimates

The **StocBias** procedure estimates the optimisation bias without
requiring a full walkforward or cross-validation:

1. Optimise the system on the real data → observed metric
2. Permute (shuffle) the returns to destroy temporal structure
3. Re-optimise on each permuted sample → collect metrics
4. Bias estimate = mean of permuted metrics
5. Debiased metric = observed - bias estimate

This works because on random data the expected metric of an optimised
system is the bias.

### Cheap Parameter Relationships

Evaluate the strategy metric across a 2-D grid of parameter values to:
- Identify **ridges** (ranges of good parameters) vs **peaks** (fragile
  single-point optima)
- Ridge-like surfaces indicate robust strategies
- Sharp peaks suggest overfitting to specific parameter values

### Parameter Sensitivity Curves

Vary **one** parameter at a time while holding others fixed:
- Ideal: metric changes smoothly and gradually
- Dangerous: metric changes abruptly at a specific value
- Smooth sensitivity curves suggest the strategy is robust to parameter
  perturbation

### Key Takeaways
1. Always debiase your observed performance
2. Prefer parameter regions on wide ridges rather than narrow peaks
3. Test sensitivity to each parameter individually
4. If debiased performance is negative, the system likely has no edge
