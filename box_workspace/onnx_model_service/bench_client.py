"""Load test: send N windows of random features, report throughput and per-worker split.

usage: python bench_client.py [url] [windows] [windows_per_request] [concurrent_requests]
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
total = int(sys.argv[2]) if len(sys.argv) > 2 else 64
per_req = int(sys.argv[3]) if len(sys.argv) > 3 else 8
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
W = 512
x = np.random.default_rng(0).standard_normal((per_req * W, 458)).astype("<f4").tobytes()

def call(_):
    req = urllib.request.Request(url + "/predict", data=x, headers={"X-Window-Size": str(W)})
    with urllib.request.urlopen(req, timeout=600) as r:
        return len(r.read())

before = json.load(urllib.request.urlopen(url + "/health"))
t = time.time()
with ThreadPoolExecutor(conc) as ex:
    list(ex.map(call, range(total // per_req)))
dt = time.time() - t
after = json.load(urllib.request.urlopen(url + "/health"))
print(f"{total} windows in {dt:.2f}s -> {dt / total * 1000:.1f} ms/window, {total * W / dt:,.0f} pulses/s")
for a, b in zip(after["workers"], before["workers"]):
    print(f"  {a['name']:5s} {a['provider']:28s} windows={a['windows'] - b['windows']}")
