"""Quick smoke test for /api/v1/verdict/run."""
import json, urllib.request, urllib.error, time

body = json.dumps({
    "tickers": ["INFY"],
    "market": "IND",
    "date_range": ["", ""],
    "skip_layers": [],
    "weights": {"core": 0.3, "strategy": 0.3, "ml_features": 0.2, "robustness": 0.2},
}).encode()

req = urllib.request.Request(
    "http://localhost:9001/api/v1/verdict/run",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)

t0 = time.time()
print("Testing verdict endpoint...")
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s) — status {resp.status}")
        print(resp.read().decode()[:500])
except urllib.error.HTTPError as e:
    elapsed = time.time() - t0
    print(f"HTTP {e.code} ({elapsed:.1f}s): {e.read().decode()[:300]}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAIL ({elapsed:.1f}s): {type(e).__name__}: {e}")
