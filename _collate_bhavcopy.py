"""Phase 2+3: Collate per-ticker adjusted OHLCV from cached bhavcopy files.

Optimized version: reads all cached CSVs via vectorized pandas concat
instead of row-by-row iterrows(). ~50x faster.
"""
import time
import os
import glob
import pandas as pd
from datetime import date
from pathlib import Path

START = date(2012, 1, 1)
END = date.today()
CACHE_DIR = Path("data/bhavcopy_cache")
OUT_DIR = Path("data/bhavcopy_collated")

# ── Phase 2a: Load PIT universe ──

print("=" * 70)
print("  Phase 2: Collate per-ticker adjusted OHLCV (vectorized)")
print("=" * 70)

from kite_connect.nse.nse_universe import get_nse_universe_pit_union
pit_symbols = get_nse_universe_pit_union()
raw_set = set(s.upper() for s in pit_symbols)
print(f"NIFTY500 PIT union: {len(raw_set)} symbols", flush=True)

# ── Phase 2b: Read ALL cached CSVs at once, filter vectorized ──

csv_files = sorted(glob.glob(str(CACHE_DIR / "**" / "*.csv"), recursive=True))
print(f"Found {len(csv_files)} cached bhavcopy files", flush=True)

t0 = time.time()
chunks = []
files_read = 0
rows_matched = 0

for i, fpath in enumerate(csv_files):
    try:
        df = pd.read_csv(fpath)
        # Filter for our symbols + EQ/BE series in one vectorized op
        mask = df["SYMBOL"].isin(raw_set) & df["SERIES"].isin(["EQ", "BE"])
        matched = df.loc[mask]
        if not matched.empty:
            # Extract date from filename: YYYYMMDD.csv
            fname = os.path.basename(fpath)
            dt_str = fname.replace(".csv", "")
            matched = matched.copy()
            matched["Date"] = pd.Timestamp(dt_str)
            chunks.append(matched[["Date", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "TOTTRDQTY"]])
            rows_matched += len(matched)
        files_read += 1
    except Exception as e:
        files_read += 1

    if (i + 1) % 500 == 0 or (i + 1) == len(csv_files):
        elapsed = time.time() - t0
        rate = files_read / elapsed if elapsed > 0 else 0
        print(f"  [{i+1}/{len(csv_files)}] rows_matched={rows_matched} rate={rate:.0f} files/s", flush=True)

elapsed = time.time() - t0
print(f"CSV scan complete: {files_read} files, {rows_matched} rows in {elapsed:.1f}s", flush=True)

# Concat all at once
print("Concatenating...", flush=True)
raw_all = pd.concat(chunks, ignore_index=True)
raw_all.rename(columns={
    "SYMBOL": "Ticker",
    "OPEN": "Open",
    "HIGH": "High",
    "LOW": "Low",
    "CLOSE": "Close",
    "TOTTRDQTY": "Volume",
}, inplace=True)

# Cast types
for col in ["Open", "High", "Low", "Close"]:
    raw_all[col] = pd.to_numeric(raw_all[col], errors="coerce")
raw_all["Volume"] = pd.to_numeric(raw_all["Volume"], errors="coerce").fillna(0).astype(int)

n_tickers_raw = raw_all["Ticker"].nunique()
print(f"Raw data: {raw_all.shape[0]} rows, {n_tickers_raw} tickers", flush=True)

# ── Phase 2c: Corporate action adjustments ──

print("\nApplying corporate action adjustments...", flush=True)
from services.bhavcopy_fetcher import fetch_corporate_actions, _parse_split_ratio

t1 = time.time()
ca_count = 0
ca_errors = 0
tickers_with_ca = []

unique_tickers = sorted(raw_all["Ticker"].unique())
for i, ticker in enumerate(unique_tickers):
    try:
        ca = fetch_corporate_actions(ticker, start=START, end=END)
        if ca is not None and not ca.empty:
            ca_count += 1
            tickers_with_ca.append(ticker)
            # Apply backward adjustment: for each corporate action,
            # multiply prices before ex_date by 1/factor, volume by factor
            for _, row in ca.iterrows():
                ex_date = pd.Timestamp(row["ex_date"])
                factor = float(row["factor"])
                mask = (raw_all["Ticker"] == ticker) & (raw_all["Date"] < ex_date)
                for col in ["Open", "High", "Low", "Close"]:
                    raw_all.loc[mask, col] = raw_all.loc[mask, col] / factor
                raw_all.loc[mask, "Volume"] = (raw_all.loc[mask, "Volume"] * factor).astype(int)
    except Exception as e:
        ca_errors += 1

    if (i + 1) % 100 == 0 or (i + 1) == len(unique_tickers):
        elapsed2 = time.time() - t1
        print(f"  [{i+1}/{len(unique_tickers)}] ca_found={ca_count} errors={ca_errors} ({elapsed2:.0f}s)", flush=True)

elapsed2 = time.time() - t1
print(f"Corporate actions: {ca_count} tickers had splits/bonuses, {ca_errors} errors ({elapsed2/60:.1f} min)", flush=True)
if tickers_with_ca:
    print(f"  Adjusted tickers: {', '.join(tickers_with_ca[:20])}{'...' if len(tickers_with_ca) > 20 else ''}")

# ── Phase 3: Save ──

print()
print("=" * 70)
print("  Phase 3: Save collated data")
print("=" * 70)

raw_all.sort_values(["Ticker", "Date"], inplace=True)
raw_all.reset_index(drop=True, inplace=True)

n_tickers = raw_all["Ticker"].nunique()
print(f"Combined shape: {raw_all.shape}")
print(f"Tickers: {n_tickers}")
date_min = raw_all["Date"].min()
date_max = raw_all["Date"].max()
print(f"Date range: {date_min} to {date_max}")
mem_mb = raw_all.memory_usage(deep=True).sum() / 1e6
print(f"Memory: {mem_mb:.1f} MB")

# Per-ticker stats
days_per_ticker = raw_all.groupby("Ticker").size()
print(f"Days/ticker: min={days_per_ticker.min()}, median={int(days_per_ticker.median())}, max={days_per_ticker.max()}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

out_path = OUT_DIR / "nifty500_ohlcv.parquet"
raw_all.to_parquet(out_path, index=False, compression="snappy")
size_mb = os.path.getsize(out_path) / 1e6
print(f"Saved: {out_path} ({size_mb:.1f} MB)")

csv_path = OUT_DIR / "nifty500_ohlcv.csv.gz"
raw_all.to_csv(csv_path, index=False, compression="gzip")
csv_size = os.path.getsize(csv_path) / 1e6
print(f"Saved: {csv_path} ({csv_size:.1f} MB)")

print()
print("=" * 70)
print("  Summary")
print("=" * 70)
counts = raw_all.groupby("Ticker").size()
print(f"  Tickers:         {n_tickers}")
print(f"  Total rows:      {len(raw_all):,}")
print(f"  Date range:      {date_min.date()} to {date_max.date()}")
print(f"  Parquet size:    {size_mb:.1f} MB")
print(f"  CSV.gz size:     {csv_size:.1f} MB")
print(f"  Min days/ticker: {counts.min()}")
print(f"  Max days/ticker: {counts.max()}")
print(f"  Median days:     {counts.median():.0f}")
print(f"  Tickers < 250d:  {(counts < 250).sum()}")
