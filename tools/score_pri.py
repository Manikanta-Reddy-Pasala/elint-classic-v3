"""Score pri_pattern from classic_eval JSON lines against gen_primix manifest.
usage: score_pri.py results.jsonl manifest.json [--detail]
Target = emitter_id 1 (labelled type); others = gen_freqmodes radars
(constant / stagger / gaussian 5 % jitter; type only).
A true emitter counts only when classic found it as one clean group
(>= 80 % of its pulses, group >= 80 % pure); else 'group_fail'.
"""
import collections, json, os, sys

res = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
man = json.load(open(sys.argv[2]))
detail = "--detail" in sys.argv
TYPES = ["constant", "jitter", "stagger", "dwell_switch", "sliding"]


def find(r, eid):
    n = r["truth_counts"].get(str(eid), 0)
    best = max(r["emitters"], key=lambda g: g["truth"].get(str(eid), 0), default=None)
    if not best:
        return None
    have = best["truth"].get(str(eid), 0)
    return best if have >= 0.8 * n and have >= 0.8 * best["n"] else None


def rel(a, b):
    return abs(a - b) / b


conf = collections.defaultdict(collections.Counter)
conf_o = collections.defaultdict(collections.Counter)
val = collections.defaultdict(list)
checks = collections.defaultdict(collections.Counter)
for r in res:
    stem = os.path.basename(r["file"])[:-4]
    m = man[stem]; lab = m["target"]; t = lab["type"]
    for o in m["others"]:
        go = find(r, o["id"])
        conf_o[o["type"]][go["pri_pattern"]["kind"] if go else "group_fail"] += 1
    g = find(r, 1)
    if g is None:
        conf[t]["group_fail"] += 1
        continue
    p = g["pri_pattern"]; k = p["kind"]
    conf[t][k] += 1
    if detail and k != t:
        print(f"MISS {stem}: true {t} got {k} lab={ {a: b for a, b in lab.items() if a != 'elements_us'} } pri={p['pri_us']} lv={[round(x) for x in p['levels_us']]} h={[round(x) for x in p['harmonics_us']]} jit={p['jitter_pct']}")
    if k != t:
        continue
    val[t].append(rel(p["pri_us"], lab["pri_us"]))
    c = checks[t]
    c["n"] += 1
    c["missing_err<=5pt"] += abs(p["missing_pct"] - lab["missing_pct"]) <= 5
    if t == "jitter":
        c["jitter_pct_err<=1.5pt"] += abs(p["jitter_pct"] - lab["jitter_pct"]) <= 1.5
    if t == "stagger":
        s = p["stagger"]
        c["period<=1%"] += rel(s["period_us"], lab["period_us"]) <= 0.01
        el, want = s["elements_us"], lab["elements_us"]
        # same cycle: any rotation of the true sequence (ties in the start)
        ok = len(el) == len(want) and any(all(rel(a, b) <= 0.01 for a, b in zip(el, want[i:] + want[:i])) for i in range(len(want)))
        c["elements_in_cycle_order"] += ok
        if detail and not ok:
            print(f"STAG {stem}: want {[round(x) for x in want]} got {[round(x) for x in el]}")
    if t == "dwell_switch":
        d = p["dwell"]
        c["pris_all<=1%"] += len(d["pris_us"]) == len(lab["pris_us"]) and all(rel(a, b) <= 0.01 for a, b in zip(d["pris_us"], lab["pris_us"]))
        c["dwell_pulses<=10%"] += rel(d["dwell_pulses"], lab["dwell_pulses"]) <= 0.10
    if t == "sliding":
        s = p["sliding"]
        c["min_max<=3%"] += rel(s["min_us"], lab["min_us"]) <= 0.03 and rel(s["max_us"], lab["max_us"]) <= 0.03

cols = TYPES + ["complex", "too_few", "group_fail"]
print("target radar: true type (rows) vs found (cols)")
print(f"{'true':14s}" + "".join(f"{c[:9]:>10s}" for c in cols) + "   acc")
tot_ok = tot = 0
for t in TYPES:
    row = conf[t]; n = sum(row.values()); ok = row[t]; tot += n; tot_ok += ok
    print(f"{t:14s}" + "".join(f"{row[c]:>10d}" for c in cols) + f"   {ok}/{n}")
print(f"TYPE ACCURACY {tot_ok}/{tot}")
print("\nvalue checks (only where type was right)")
for t in TYPES:
    v = sorted(val[t])
    med = v[len(v) // 2] * 100 if v else float("nan")
    c = checks[t]
    extra = ", ".join(f"{k} {c[k]}/{c['n']}" for k in c if k != "n")
    print(f"  {t:13s} pri_us median err {med:.2f}% ; {extra}")
print("\nother radars (type only): true vs found")
for t, row in conf_o.items():
    print(f"  {t:10s} " + ", ".join(f"{k} {v}" for k, v in row.most_common()))
