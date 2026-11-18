"""Grouping-identical check: two classic_eval JSONL files, ignoring timing
(`ms`), the report-only `pri_pattern` key and group order. usage: same_groups.py a.jsonl b.jsonl"""
import json, sys


def load(p):
    out = {}
    for l in open(p):
        if l.startswith("{"):
            r = json.loads(l); r.pop("ms", None)
            for e in r["emitters"]:
                e.pop("pri_pattern", None)
            # unknown groups come out in run-dependent order (also before this
            # change): compare the groups as a sorted list
            r["emitters"] = sorted(json.dumps(e, sort_keys=True) for e in r["emitters"])
            out[r["file"]] = r
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
diff = [f for f in a if a[f] != b.get(f)]
print(f"files {len(a)} vs {len(b)}; groups identical: {len(a) - len(diff)}/{len(a)}; groups {sum(len(r['emitters']) for r in a.values())}")
for f in diff[:5]:
    print("DIFF", f)
sys.exit(1 if diff or len(a) != len(b) else 0)
