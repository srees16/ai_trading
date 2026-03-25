# Chapter 1: Introduction

**Testing and Tuning Market Trading Systems** — Timothy Masters (2018)

## Overview

This chapter introduces the foundational concepts behind testing and tuning
market trading systems.  Key topics:

### Market Prices and Returns
- Log returns vs simple returns — properties and when to use each
- The importance of understanding return distributions for system evaluation

### Two Types of Automated Trading Systems
1. **Indicator-based systems** — rules derived from technical indicators
   (moving averages, RSI, etc.)
2. **Machine-learning-based systems** — models trained on historical data
   to predict future price movements

### The Agony of Believing the Computer
- Overfitting danger: a system that looks brilliant on historical data
  may fail catastrophically in live trading
- The critical distinction between *in-sample* and *out-of-sample* performance

### Future Leak Is More Dangerous Than You May Think
- **Future leak** occurs when information from the future inadvertently
  enters the training process
- Even subtle forms (e.g., using today's close to make decisions for today)
  can drastically inflate backtest performance
- Autocorrelation analysis can help detect suspicious patterns

### The Percent Wins Fallacy
- A high win rate does **not** guarantee profitability
- A system with 80% wins can lose money if the average loss >> average win
- Conversely, trend-following systems often have 30-40% win rates but are
  profitable due to large winners
- **Expectancy** and **profit factor** are better metrics than win rate alone
