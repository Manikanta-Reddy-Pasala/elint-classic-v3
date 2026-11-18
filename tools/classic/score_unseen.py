"""Score classic_eval output on gen_unseen.py sets.
usage: score_unseen.py results.jsonl manifest.json ref.jsonl [--detail]
ref.jsonl = classic_eval on the BASE corpus files (gives each known radar's
expected library name; the corpus is 608/608 so these are the right names).

Per new (unseen) radar:  rec   = one group holds >= 80 % of its pulses and
                                 that group is >= 80 % this radar
                         named = a group that is mostly this radar got a
                                 library name (false name)
                         split = more than one group is mostly this radar
Per known radar:         ok    = still recovered with the same name as in
                                 the base file
File ok = every new radar rec, not named, not split + every known radar ok.
"""
import json, os, sys, collections

res = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
man = json.load(open(sys.argv[2]))
ref = {os.path.basename(r["file"]): r for r in (json.loads(l) for l in open(sys.argv[3]) if l.startswith("{"))}
detail = "--detail" in sys.argv


def best(ems, e):
    g = max(ems, key=lambda g: g["truth"].get(str(e), 0), default=None)
    return g, (g["truth"].get(str(e), 0) if g else 0)


def lib_of(r, e, n):
    g, have = best(r["emitters"], e)
    return g["lib"] if g and have >= 0.8 * n else "lost"


tot = collections.Counter(); bycls = collections.defaultdict(collections.Counter); bad = []
for r in res:
    stem = os.path.basename(r["file"])[:-4]
    if stem not in man:
        continue
    m = man[stem]; cls = m["cls"]; ems = r["emitters"]; tc = {int(k): v for k, v in r["truth_counts"].items()}
    fok = True; why = []
    for nw in m["new"]:
        e = nw["id"]; n = tc.get(e, 0)
        g, have = best(ems, e)
        rec = g is not None and have >= 0.8 * n and have >= 0.8 * g["n"]
        mine = [x for x in ems if x["truth"].get(str(e), 0) >= 0.5 * x["n"] and x["n"] >= 50]
        named = any(x["lib"] is not None for x in mine)
        split = len(mine) > 1
        for k, v in (("new", 1), ("rec", rec), ("named", named), ("split", split)):
            tot[k] += v; bycls[cls][k] += v
        if not rec or named or split:
            fok = False
            why.append(f"new{e}({nw['fmode']},{nw['f']},pw{nw['pw']},{nw['mod']},{nw['pri'][0]}{[round(x) for x in nw['pri'][1]]}): "
                       f"rec={rec} named={[x['lib'] for x in mine]} groups={len(mine)} best={g['lib'] if g else None}:{have}/{n} of {g['n'] if g else 0}")
    base = ref.get(m["base"])
    for e, n in tc.items():
        if e <= 0 or e >= 101 or n < 50:
            continue
        want = lib_of(base, e, n_base) if base and (n_base := {int(k): v for k, v in base["truth_counts"].items()}.get(e, 0)) else None
        got = lib_of(r, e, n)
        ok = got == want
        tot["known"] += 1; tot["known_ok"] += ok; bycls[cls]["known"] += 1; bycls[cls]["known_ok"] += ok
        if not ok:
            fok = False; why.append(f"known{e}: want {want} got {got}")
    tot["files"] += 1; tot["ok"] += fok; bycls[cls]["files"] += 1; bycls[cls]["ok"] += fok
    if not fok:
        bad.append((stem, why))

print(f"{'class':10s} files  ok  new  rec  named  split  known_ok")
for c in sorted(bycls):
    b = bycls[c]
    print(f"{c:10s} {b['files']:5d} {b['ok']:3d} {b['new']:4d} {b['rec']:4d} {b['named']:6d} {b['split']:6d}  {b['known_ok']}/{b['known']}")
print(f"{'TOTAL':10s} {tot['files']:5d} {tot['ok']:3d} {tot['new']:4d} {tot['rec']:4d} {tot['named']:6d} {tot['split']:6d}  {tot['known_ok']}/{tot['known']}")
if detail:
    for s, w in bad:
        print(s, " | ".join(w))
