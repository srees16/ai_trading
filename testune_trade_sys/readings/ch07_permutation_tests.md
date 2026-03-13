# Chapter 7: Permutation Tests

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

Permutation tests answer the question: *"Could the system's performance
have arisen by chance?"* by comparing the observed metric against a
distribution built from randomly rearranged data.

### Permuting Returns

The simplest approach — shuffle daily (or bar) returns:
- Destroys any serial dependence the system might exploit
- Preserves the marginal distribution of returns exactly

### Permuting Prices (Inclusive of Trend)

Reconstruct prices from shuffled returns, preserving the overall drift:
- More conservative than plain return shuffling
- Retains the first and last price (hence the net trend)
- C++ source: pp. 303-305

### Permuting Multiple Markets

When a system trades several instruments simultaneously:
- Shuffle **day labels** (not individual returns) so that the
  cross-sectional structure on any single day is preserved
- Only temporal dependencies are destroyed
- C++ source: pp. 305-309

### Permuting Bars (OHLCV)

For bar-level strategies that use open, high, low, close, volume:
- Shuffle the entire bar rather than individual fields
- Preserves the within-bar micro-structure
- C++ source: pp. 309-314

### Permutation Test Framework

The overall procedure:

1. Compute the observed metric on real data
2. Repeat *N* times:
   a. Permute the data
   b. Re-optimise the system on the permuted data
   c. Record the metric
3. p-value = fraction of permuted metrics ≥ observed metric
4. If p < 0.05, reject the null that performance is due to chance

### Walk-forward Permutation Test

Combine walk-forward analysis with permutation testing:
- Each permutation undergoes the same WFA procedure as the real data
- Provides the strongest evidence because it controls for both
  look-ahead bias and data-snooping

### Partition Return Attribution

Split the equity curve into components attributable to different
factors (e.g., trend vs mean-reversion) using block partitioning.
C++ source: pp. 298-302.

### Key Takeaways
1. Permutation tests are model-free — they make no distributional
   assumptions
2. Always re-optimise on each permuted sample, not just re-evaluate
3. p-values below 0.05 provide strong evidence of a genuine edge
4. Walk-forward permutation is the gold standard for statistical
   validation
