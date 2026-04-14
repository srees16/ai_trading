# Kaggle Kernels Reference — April 2026

Two kernels currently running on Kaggle. This doc covers objectives,
expected outputs, pass/fail criteria, and next steps for each.

---

## 1. centurion-backtest-phase-b

**URL:** https://www.kaggle.com/code/srees16/centurion-backtest-phase-b
**Dataset version:** v8 (PIT universe + mean_reversion killed)

### Objective

Validate the Phase B changes in a full 13-year backtest:
- **PIT_UNIVERSE_ENABLED = True** — uses 797 historical NIFTY500 constituents
  with semi-annual rebalancing (no survivorship bias)
- **mean_reversion signal killed** — zeroed weight (was 2.7%, Sharpe=-0.039)
- 11 active signals remain, sector_rotation at 3%

### Config

| Parameter        | Value                          |
|------------------|--------------------------------|
| Capital          | Rs 5,00,000                    |
| Period           | 2012-01-01 to 2025-12-31       |
| Vol target       | 50% annual (adaptive)          |
| Max leverage     | 2.0x                           |
| Transaction cost | 0.22% (NSE delivery + fees)    |
| Universe         | PIT NIFTY500 union (~797 tickers) |
| PIT filtering    | Semi-annual, force-sell delisted |

### Baseline (pre-Phase-B, survivorship-biased)

| Metric   | Value  |
|----------|--------|
| Sharpe   | 1.127  |
| CAGR     | 30.4%  |
| MaxDD    | 29.6%  |
| Calmar   | 1.026  |
| Trades   | 4,313  |

### What to expect

- **CAGR will likely drop** — removing survivorship bias means the universe
  includes stocks that later got delisted/merged (DHFL, CORPBANK, ALBK, etc.)
- **Sharpe may drop slightly** — mean_reversion removal + broader universe
- **More realistic estimate** of what the strategy would have actually done
- Many delisted tickers will fail to download (expected, handled gracefully)

### Output metrics

The notebook will print at the end:
```
annual_return_pct, sharpe, sortino, max_drawdown_pct, calmar, n_trades, total_return_pct
```

### Pass criteria

| Metric | Minimum acceptable | Notes                              |
|--------|--------------------|------------------------------------|
| Sharpe | >= 0.80            | Below 0.80 = strategy too weak     |
| CAGR   | >= 20%             | PIT universe will reduce this      |
| MaxDD  | <= 40%             | Beyond 40% = unacceptable drawdown |

### Next steps — IF PASS

1. Update `my_todos.txt` with new Phase B baseline metrics
2. Compare PIT vs non-PIT: quantify survivorship bias magnitude
3. Proceed to 30-day paper trading:
   - Paper Sharpe >= 0.80
   - No single-day loss > 3%
   - Slippage within 2x backtest assumption (44 bps)
4. If paper passes → staged live deployment (Rs 1L → 2L → 3L → 4L → 5L)

### Next steps — IF FAIL

1. **Sharpe < 0.80:** Check if PIT universe filtering is too aggressive.
   Consider relaxing to NIFTY200 or NIFTY100 PIT subset.
2. **CAGR < 15%:** Survivorship bias was masking a weak strategy.
   Re-examine signal weights. May need optimizer re-run on PIT data.
3. **MaxDD > 40%:** Bear floor (10%) may need tightening.
   Review SEVERE_BEAR_EXPOSURE_FLOOR in config.
4. **Crash/timeout:** Check Kaggle logs. May need to reduce universe
   size or add checkpointing to the backtest (currently no checkpointing).

---

## 2. nifty500-extract-all-signals

**URL:** https://www.kaggle.com/code/srees16/nifty500-extract-all-signals
**Status:** Was at Day ~300/3861 as of last check

### Objective

Extract **per-source daily forecasts** for the entire NIFTY500 universe
across 3,190 trading days (2012-2025). This produces a massive dataset
of individual signal forecasts for every stock on every day.

### Why this matters

The current signal weights (v27) were optimized on NIFTY50+NEXT50 (95 stocks).
This extraction gives us forecasts on 500 stocks, enabling:
- Broader signal validation (do signals work across mid/small caps?)
- Weight re-optimization on a larger universe
- Identification of dormant signals that shine on broader universe
- Data for Phase A (dormant signal activation)

### Signals extracted (20+)

| Category       | Signals                                                                 |
|----------------|-------------------------------------------------------------------------|
| Active (11)    | ewmac_8_32, ewmac_16_64, ewmac_64_256, momentum, ehlers_dsp,          |
|                | acceleration, penfold_trend, carver_value, screener, breakout,          |
|                | sector_rotation                                                         |
| Killed (1)     | mean_reversion (still extracted for analysis, just zero-weighted)        |
| New alpha (6)  | calendar, fundamental_momentum, insider, dispersion,                    |
|                | gold_equity_rotation, crypto_correlation                                |
| Extra          | carry, skew_signal, ewmac_32_128, cross_momentum, intermarket, etc.     |

### Checkpoint strategy

- Checkpoints every 50 days + at 11-hour mark (Kaggle 12h limit)
- Multi-session: each run resumes from last checkpoint
- Upload checkpoint + ticker list back to dataset between sessions

### Output

**File:** `extracted_forecasts_nifty500.pkl` (~500+ MB)

Contains per-day snapshots:
```
(day_idx, date, {ticker: {signal: forecast_value, ...}}, {ticker: next_day_return}, equity)
```

### What to expect

- At ~300/3861 days, the extraction is ~8% complete
- Will need **3-4 more Kaggle sessions** (each covers ~1000 days in 12h)
- Between sessions: download checkpoint, upload to dataset, re-run notebook

### Pass criteria

| Check                      | Expectation                         |
|----------------------------|-------------------------------------|
| Extraction completes       | All 3,190+ days extracted           |
| Signal coverage            | 20+ signals per ticker per day      |
| No data corruption         | Pickle loads cleanly, shapes match  |
| Sufficient ticker coverage | >= 400 tickers with valid forecasts |

### Next steps — WHEN COMPLETE

1. **Download** `extracted_forecasts_nifty500.pkl` from Kaggle output
2. **Run signal validation** on broad universe:
   `optimizer/validate_all_signals_broad.py`
3. **Identify new viable signals** — any of the 6 new alpha signals
   that show Sharpe > 0.05 standalone become candidates for activation
4. **Re-optimize weights** on NIFTY500:
   `optimizer/optimize_signal_weights.py`
5. **Phase A activation** — add newly validated signals, re-run backtest

### Next steps — IF FAILS / STALLS

1. **Kaggle timeout (no checkpoint):** Check if checkpoint code ran.
   Look for `backtest_checkpoint_nifty500_extract.pkl` in output.
2. **Checkpoint exists but extraction incomplete:** Download checkpoint,
   upload to dataset, re-push notebook. It resumes automatically.
3. **Memory error:** NIFTY500 x 20 signals x 3190 days is large.
   May need to batch by ticker groups or reduce signal count.
4. **API rate limits (yfinance):** Reduce batch size or add delays.
   Check if `enable_internet: true` is set in kernel metadata.

---

## Priority order

1. **Phase B backtest** — higher priority. If it passes, we can start
   paper trading immediately while extraction continues in background.
2. **NIFTY500 extraction** — long-running background task. Results feed
   into future optimization cycles (Phase A), not blocking paper trading.

## Key risk flags to keep in mind

| Risk                        | Mitigation                                |
|-----------------------------|-------------------------------------------|
| Detrended Sharpe ≈ 0        | Returns driven by market beta, not alpha  |
| DM bias 247%                | 24-signal search space, overfitting risk  |
| Win rate 38.5%              | Relies on few large winners               |
| WF1 (2018-19) Sharpe=-0.22 | Fails in range-bound flat markets         |
| Hockey-stick equity curve   | Compounding artifact, not realistic       |
