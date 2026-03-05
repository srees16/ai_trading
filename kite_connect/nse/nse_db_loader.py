"""
Loads the downloaded NSE CSV into the livestocks_ind PostgreSQL database.

Mapping:
  CSV column  ->  DB column
  SYMBOL      ->  name
  HIGH        ->  high
  LOW         ->  low
  VOLUME      ->  volume
  LTP         ->  ltp
  %CHNG       ->  change
"""

import glob
import os
import sys

import pandas as pd

# Append kite_connect to path (not insert) to avoid shadowing top-level packages
_kite_root = os.path.dirname(os.path.dirname(__file__))
if _kite_root not in sys.path:
    sys.path.append(_kite_root)

from core.config import DOWNLOAD_DIR, TABLE_NAME, DB_NAME
from core.db_service import get_connection


def find_latest_csv():
    """Find the most recent MW-*.csv file in Downloads."""
    files = glob.glob(os.path.join(DOWNLOAD_DIR, "MW-*.csv"))
    if not files:
        raise FileNotFoundError("No MW-*.csv file found in Downloads folder")
    return max(files, key=os.path.getmtime)


def clean_numeric(value):
    """Remove commas and convert to float. Returns None for non-numeric."""
    if pd.isna(value) or str(value).strip() in ("", "-"):
        return None
    return float(str(value).replace(",", ""))


def clean_volume(value):
    """Remove commas and convert to int. Returns None for non-numeric."""
    if pd.isna(value) or str(value).strip() in ("", "-"):
        return None
    return int(str(value).replace(",", ""))


def load_csv_to_db():
    csv_path = find_latest_csv()
    print(f"  Loading: {os.path.basename(csv_path)}")

    # Read CSV - columns have trailing whitespace/newlines
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Normalize column names: strip whitespace and newlines
    df.columns = [col.strip().replace("\n", " ").strip() for col in df.columns]

    # Select and rename columns
    col_map = {
        "SYMBOL": "name",
        "HIGH": "high",
        "LOW": "low",
        "VOLUME": "volume",
        "LTP": "ltp",
        "%CHNG": "change",
    }

    # Find actual column names (fuzzy match - check if target key appears in column name)
    actual_map = {}
    for target_key, db_col in col_map.items():
        for col in df.columns:
            col_upper = col.upper().strip()
            # Exact start match or containment
            if col_upper == target_key or col_upper.startswith(target_key + " ") or col_upper.startswith(target_key + "("):
                actual_map[col] = db_col
                break

    print(f"  Column mapping: {actual_map}")

    df_mapped = df.rename(columns=actual_map)[list(actual_map.values())]

    # Clean numeric values (remove commas)
    df_mapped["high"] = df_mapped["high"].apply(clean_numeric)
    df_mapped["low"] = df_mapped["low"].apply(clean_numeric)
    df_mapped["ltp"] = df_mapped["ltp"].apply(clean_numeric)
    df_mapped["change"] = df_mapped["change"].apply(clean_numeric)
    df_mapped["volume"] = df_mapped["volume"].apply(clean_volume)

    # Strip stock name
    df_mapped["name"] = df_mapped["name"].str.strip()

    # Drop rows where name is empty
    df_mapped = df_mapped.dropna(subset=["name"])

    print(f"  Rows to insert: {len(df_mapped)}")

    # Upsert into PostgreSQL (INSERT or UPDATE existing rows)
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    upsert_sql = f"""
        INSERT INTO {TABLE_NAME} (name, high, low, volume, ltp, change, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (name) DO UPDATE
        SET high = EXCLUDED.high,
            low  = EXCLUDED.low,
            volume = EXCLUDED.volume,
            ltp  = EXCLUDED.ltp,
            change = EXCLUDED.change,
            updated_at = NOW();
    """

    rows_upserted = 0
    for _, row in df_mapped.iterrows():
        cur.execute(upsert_sql, (
            row["name"],
            row["high"],
            row["low"],
            row["volume"],
            row["ltp"],
            row["change"],
        ))
        rows_upserted += 1

    cur.close()
    conn.close()

    print(f"\n  [OK] {rows_upserted} rows upserted into {DB_NAME}.{TABLE_NAME}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Load NSE CSV -> livestocks_ind DB")
    print("=" * 60)
    load_csv_to_db()
    print("\n  Done!")
