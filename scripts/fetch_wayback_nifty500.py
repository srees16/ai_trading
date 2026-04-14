"""Fetch NIFTY500 constituent lists from Wayback Machine and build PIT JSON."""
import requests, csv, io, json, os, time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

snapshots = [
    ('2018-10', 'https://web.archive.org/web/20181004/http://www.niftyindices.com:80/IndexConstituent/ind_nifty500list.csv'),
    ('2019-02', 'https://web.archive.org/web/20190201/http://niftyindices.com/IndexConstituent/ind_nifty500list.csv'),
    ('2020-07', 'https://web.archive.org/web/20200725/https://nseindia.com/content/indices/ind_nifty500list.csv'),
    ('2022-05', 'https://web.archive.org/web/20220504/https://archives.nseindia.com/content/indices/ind_nifty500list.csv'),
    ('2022-10', 'https://web.archive.org/web/20221009/https://archives.nseindia.com/content/indices/ind_nifty500list.csv'),
    ('2023-04', 'https://web.archive.org/web/20230404/https://archives.nseindia.com/content/indices/ind_nifty500list.csv'),
    ('2024-02', 'https://web.archive.org/web/20240207/https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'),
    ('2024-02b', 'https://web.archive.org/web/20240226/https://archives.nseindia.com/content/indices/ind_nifty500list.csv'),
    ('2025-06', 'https://web.archive.org/web/20250616/https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'),
    ('2025-08', 'https://web.archive.org/web/20250821/https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'),
]

results = {}
for label, url in snapshots:
    try:
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            text = r.text
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            header = rows[0] if rows else []
            print(f"{label}: status={r.status_code}, rows={len(rows)}, header={header[:5]}")
            sym_col = None
            for i, h in enumerate(header):
                h_clean = h.strip().lower().replace('\n', '')
                if 'symbol' in h_clean:
                    sym_col = i
                    break
            if sym_col is None:
                sample = rows[1][:5] if len(rows) > 1 else "empty"
                print(f"  No symbol col found. Sample: {sample}")
            else:
                syms = []
                for row in rows[1:]:
                    if len(row) > sym_col:
                        s = row[sym_col].strip()
                        if s and s != 'Symbol':
                            syms.append(s)
                print(f"  Symbols: {len(syms)} (first 5: {syms[:5]})")
                results[label] = syms
        else:
            print(f"{label}: status={r.status_code}")
    except Exception as e:
        print(f"{label}: error - {e}")
    time.sleep(1.5)

print(f"\nTotal snapshots with data: {len(results)}")
for k, v in results.items():
    print(f"  {k}: {len(v)} symbols")

# Save raw results
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty500_wayback_raw.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw data to {out_path}")
