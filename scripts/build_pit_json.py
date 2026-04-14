"""Build nifty500_historical_constituents.json from wayback data."""
import json

with open('centurion_core/data/nifty500_wayback_raw.json') as f:
    raw = json.load(f)

# Map wayback labels to semi-annual periods
mapping = {
    '2018-10': '2018-09',
    '2019-02': '2019-03',
    '2020-07': '2020-09',
    '2022-05': '2022-03',
    '2022-10': '2022-09',
    '2023-04': '2023-03',
    '2024-02': '2024-03',
    '2024-02b': '2024-03',  # duplicate
    '2025-06': '2025-03',
    '2025-08': '2025-09',
}

pit = {}
for label, period in mapping.items():
    if label in raw:
        syms = raw[label]
        if period not in pit or len(syms) > len(pit[period]):
            pit[period] = sorted(set(syms))

# For missing periods pre-2018, use earliest available (2018-09)
earliest = pit.get('2018-09', [])
for y in range(2012, 2019):
    for m in ['03', '09']:
        key = f'{y}-{m}'
        if key not in pit:
            pit[key] = earliest

# Fill missing intermediate periods from nearest known
known_sorted = sorted(pit.keys())
all_needed = []
for y in range(2012, 2026):
    for m in ['03', '09']:
        all_needed.append(f'{y}-{m}')

for period in all_needed:
    if period not in pit:
        # Find nearest known period
        nearest = min(known_sorted, key=lambda k: abs(
            int(k[:4])*12 + int(k[5:]) - int(period[:4])*12 - int(period[5:])
        ))
        pit[period] = pit[nearest]

# Sort
pit = dict(sorted(pit.items()))

print(f"Periods with data: {len(pit)}")
for p in sorted(pit.keys()):
    print(f"  {p}: {len(pit[p])} symbols")

# Check key stocks
print()
key_stocks = {
    'DHFL': 'Should be in 2018-09, 2019-03; gone by 2020',
    'YESBANK': 'Should be in most periods',
    'JETAIRWAYS': 'Should be in 2018-09',
    'ZOMATO': 'Should NOT be before 2021-09',
    'PAYTM': 'Should NOT be before 2021-09',
    'LIC': 'Should NOT be before 2022-03',
    'RCOM': 'Should be in 2018, gone later',
}
for sym, note in key_stocks.items():
    found_in = [p for p in sorted(pit.keys()) if sym in pit[p]]
    not_found = "NONE" if not found_in else ""
    if found_in:
        print(f"  {sym}: found in {found_in[0]}..{found_in[-1]} ({len(found_in)} periods) -- {note}")
    else:
        print(f"  {sym}: NOT FOUND in any snapshot -- {note}")

# Save
out = 'centurion_core/data/nifty500_historical_constituents.json'
with open(out, 'w') as f:
    json.dump(pit, f, indent=2, sort_keys=True)
print(f"\nSaved {len(pit)} periods to {out}")
