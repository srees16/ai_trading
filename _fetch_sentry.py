"""Temp script to fetch Sentry shared issues."""
import requests
import json
import re

ISSUES = [
    "2107db4a013040e2a74586b3205e7c96",
    "4f6bcff254d149579723d86e6ae35100",
    "048dd6e6bcb649b18eaa0d5590a87cd3",
    "d29f0752feee427ab6007cf1819b7e47",
]

for sid in ISSUES:
    url = f"https://research-ew.sentry.io/share/issue/{sid}/"
    r = requests.get(url, timeout=15)
    print(f"=== {sid[:12]} (HTTP {r.status_code}) ===")
    
    # Try to find embedded JSON data in the page
    # Sentry embeds initial data as window.__initialData or similar
    patterns = [
        r'window\.__initialData\s*=\s*(\{.*?\});',
        r'window\.__SENTRY__DATA\s*=\s*(\{.*?\});',
        r'"title":\s*"([^"]+)"',
        r'"culprit":\s*"([^"]+)"',
        r'"type":\s*"([^"]+)".*?"value":\s*"([^"]+)"',
    ]
    
    text = r.text
    
    # Extract title
    titles = re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    culprits = re.findall(r'"culprit"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
    
    # Look for exception type and value in metadata
    meta_match = re.search(r'"metadata"\s*:\s*\{[^}]*"type"\s*:\s*"((?:[^"\\]|\\.)+)"[^}]*"value"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    
    if titles:
        print(f"  Title: {titles[0]}")
    if culprits:
        print(f"  Culprit: {culprits[0]}")
    if meta_match:
        print(f"  Exception: {meta_match.group(1)}: {meta_match.group(2)[:300]}")
    
    # Try to extract from script tag with JSON
    script_match = re.search(r'<script[^>]*>\s*window\.__(?:initialData|SENTRY__DATA)\s*=\s*(.*?);\s*</script>', text, re.DOTALL)
    if script_match:
        try:
            data = json.loads(script_match.group(1))
            print(f"  Found embedded data")
            print(json.dumps(data, indent=2)[:500])
        except:
            pass
    
    # Also look for the API URL pattern that loads event data
    api_urls = re.findall(r'"/api/0/[^"]*"', text)
    if api_urls:
        print(f"  API URLs found: {api_urls[:3]}")
    
    if not titles and not meta_match:
        # Check if page has any meaningful content
        content_len = len(text)
        print(f"  Page size: {content_len} bytes")
        # Look for any JSON-like structures
        json_blocks = re.findall(r'\{[^{]*"(?:title|type|value|exception)"[^}]*\}', text[:5000])
        if json_blocks:
            for j in json_blocks[:3]:
                print(f"  JSON block: {j[:200]}")
    
    print()
