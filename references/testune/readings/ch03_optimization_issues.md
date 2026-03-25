# Chapter 3: Optimization Issues

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

This chapter covers regularised linear models (elastic-net via coordinate
descent), nonlinear extensions, and differential evolution for global
optimisation.

### Regularizing a Linear Model

**Elastic-net** combines L1 (Lasso) and L2 (Ridge) penalties:

$$\min_{\beta} \frac{1}{2N} \|y - X\beta\|^2 + \lambda \left[\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2\right]$$

**Coordinate descent** optimises one coefficient at a time while holding
the others fixed.  The soft-thresholding operator is:

$$S(z, \gamma) = \text{sign}(z) \max(|z| - \gamma, 0)$$

**Lambda path:** Start from $\lambda_{max}$ (all coefficients zero) and
decrease.  Warm-starting from the previous solution makes each step cheap.

**Cross-validation** selects the optimal $\lambda$ by testing each
candidate on held-out folds.

### Making a Linear Model Nonlinear

- **Polynomial basis expansion:** map $X$ to higher-degree terms
  (including interactions) to capture nonlinear relationships
- The regularised model on expanded features effectively learns a
  nonlinear decision boundary while still using efficient linear algebra

### Differential Evolution

A population-based global optimiser that works for any black-box objective:

1. **Initialise** random population within bounds
2. **Mutate:** donor = $x_a + F(x_b - x_c)$
3. **Crossover:** mix donor with current individual
4. **Select:** keep the better of trial vs current
5. Repeat for $G$ generations

Strengths: no gradient needed, handles multimodal landscapes, few
hyperparameters ($F$, $CR$, population size).
