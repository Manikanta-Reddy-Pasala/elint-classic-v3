"""Build a fresh test set the classic rules were never tuned on.

Each new file = two existing test files merged on the same time axis
(radars from both are on air at once), so every file has a radar mix that
is not in the 608-file corpus. Rules:
  * the two source files share no radar type (no MF03 + MF03), because
    both engines report one emitter per library id by design;
  * emitter_id of the second file is offset so every radar keeps its own id;
  * truth = number of distinct emitter_id values + known library ids.
Output: <out>/<a>__<b>.csv and <out>/truth.json
"""
import csv
import json
import os
import random
import re
import sys

DATA = os.path.expanduser("~/coderepo/ai_elint/data/pdw_generator")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/coderepo/elint_improvements/holdout")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LIB = {"psr1": "1104", "psr2": "1105", "psr3": "1106", "psr4": "1107", "psr5m1": "1108",
       "mf04": "1109", "mf03": "1113", "psr5m2": "1116", "mf06": "1117"}

rng = random.Random(1117)
files = []
for kind in ("known_emitters", "unknown_emitters"):
    for f in sorted(os.listdir(os.path.join(DATA, kind))):
        if f.endswith(".csv"):
            m = re.match(r"(.+?)_best-(\d+)_plus(\d+)\.csv$", f)
            if m:
                files.append((kind, f, set(m.group(1).split("_"))))


def size_mb(kind, f):
    return os.path.getsize(os.path.join(DATA, kind, f)) / 1e6


pool = [x for x in files if size_mb(x[0], x[1]) < 25]  # keep each mix under ~50 MB
os.makedirs(OUT, exist_ok=True)
truth, seen = {}, set()
tries = 0
while len(truth) < N and tries < 10000:
    tries += 1
    a, b = rng.sample(pool, 2)
    if a[2] & b[2] or (a[1], b[1]) in seen:
        continue
    seen.add((a[1], b[1]))
    name = a[1][:-4] + "__" + b[1][:-4] + ".csv"
    rows, ids = [], set()
    for k, (kind, f, _) in enumerate((a, b)):
        with open(os.path.join(DATA, kind, f), newline="") as fh:
            r = csv.DictReader(fh)
            fields = r.fieldnames
            for row in r:
                eid = int(float(row["emitter_id"])) + (1000 if k else 0)
                row["emitter_id"] = str(eid)
                ids.add(eid)
                rows.append(row)
    rows.sort(key=lambda r: float(r["time_of_arrival_ns"]))
    for i, row in enumerate(rows, start=1):
        row["id"] = str(i)
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    known = sorted(LIB[t] for t in (a[2] | b[2]) if t in LIB)
    truth[name] = {"emitters": len(ids), "known_ids": known, "sources": [a[1], b[1]], "pulses": len(rows)}
    print(f"{len(truth):2d} {name}: {len(ids)} radars, known {known}, {len(rows)} pulses", flush=True)
json.dump(truth, open(os.path.join(OUT, "truth.json"), "w"), indent=1)
print("wrote", len(truth), "files to", OUT)
