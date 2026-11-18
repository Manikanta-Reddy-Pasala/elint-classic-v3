"""PRI types found per library emitter (sanity check on unlabelled sets).
usage: pri_summary.py results.jsonl"""
import collections, json, sys

by = collections.defaultdict(collections.Counter)
pri = collections.defaultdict(list)
for l in open(sys.argv[1]):
    if not l.startswith("{"):
        continue
    for e in json.loads(l)["emitters"]:
        key = str(e["lib"]) if e["lib"] is not None else "unknown_ai"
        p = e["pri_pattern"]
        by[key][p["kind"]] += 1
        if p["pri_us"] is not None:
            pri[key].append(p["pri_us"])
tot = collections.Counter()
print(f"{'emitter':12s} {'groups':>6s}  {'median PRI us':>13s}  types")
for k in sorted(by, key=lambda x: (x == "unknown_ai", x)):
    v = sorted(pri[k]); tot.update(by[k])
    med = v[len(v) // 2] if v else float("nan")
    print(f"{k:12s} {sum(by[k].values()):6d}  {med:13.1f}  " + ", ".join(f"{a} {b}" for a, b in by[k].most_common()))
print("ALL", ", ".join(f"{a} {b}" for a, b in tot.most_common()))
