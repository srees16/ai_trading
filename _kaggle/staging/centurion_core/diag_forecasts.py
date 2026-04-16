"""Quick diagnostic: compute Day-0 forecasts and count positive/negative."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
from services.forecast_scalar import ewmac_to_forecast, cap_forecast
from services.forecast_combiner import combine_forecasts, DEFAULT_FORECAST_WEIGHTS
from strategies.ewmac import DEFAULT_VARIATIONS

def daily_price_volatility(close, span=32):
    rets = close.pct_change().dropna()
    if len(rets) < 10:
        return 0.02
    return float(rets.ewm(span=span, adjust=False).std().iloc[-1])

# Load ALL DEFAULT tier stocks
tickers = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS",
           "BAJFINANCE.NS", "SBIN.NS", "ASIANPAINT.NS", "TITAN.NS", "ITC.NS",
           "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "HINDALCO.NS",
           "SUNPHARMA.NS", "TATASTEEL.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]

print(f"Downloading {len(tickers)} stocks...")
data = yf.download(tickers, start="2012-01-01", end="2013-06-01", progress=False)

WARMUP = 262
positive_count = 0
negative_count = 0
results = []

for sym in tickers:
    try:
        close = data["Close"][sym].dropna()
    except:
        continue
    if len(close) < WARMUP:
        continue
    
    close_slice = close.iloc[:WARMUP]
    price = float(close_slice.iloc[-1])
    dpv = daily_price_volatility(close_slice)
    
    # Compute ONLY EWMAC forecasts (the main signal source)
    fc_dict = {}
    for fast, slow in DEFAULT_VARIATIONS:
        if len(close_slice) < slow + 10:
            continue
        fast_ewma = close_slice.ewm(span=fast, adjust=False).mean()
        slow_ewma = close_slice.ewm(span=slow, adjust=False).mean()
        raw = float(fast_ewma.iloc[-1] - slow_ewma.iloc[-1])
        fc = ewmac_to_forecast(raw, dpv, fast, slow)
        key = f"ewmac_{fast}_{slow}"
        fc_dict[key] = fc

    if not fc_dict:
        continue
    
    # Combine forecasts with neutral regime (Day 0)
    combined = combine_forecasts(sym, fc_dict, regime="neutral")
    
    ERP_BIAS = 2.0
    final = combined.combined_forecast + ERP_BIAS
    sign = "POS" if final > 0 else "NEG"
    if final > 0:
        positive_count += 1
    else:
        negative_count += 1
    results.append((sym, combined.combined_forecast, final, sign, fc_dict))

# Sort by combined forecast
results.sort(key=lambda x: x[2], reverse=True)
print(f"\n=== Day 0 Forecast Summary ({len(results)} stocks) ===")
print(f"Positive after ERP: {positive_count}, Negative: {negative_count}")
print()
for sym, raw, final, sign, fcs in results:
    fc_str = ", ".join(f"{k}={v:+.1f}" for k, v in fcs.items())
    print(f"  {sign} {sym:18s}  raw={raw:+6.2f}  +ERP={final:+6.2f}  [{fc_str}]")
