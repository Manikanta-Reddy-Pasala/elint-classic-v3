"""Per-file DOA spread of found groups (unlabeled real captures).
usage: doa_report.py results.jsonl [--groups]
doa_arc80 = smallest circular DOA arc holding 80 % of a group's pulses (deg).
A direction finder sees one radar inside a few degrees; a wide arc = pulses
from different directions in one group (or an omni receiver)."""
import json, sys, os
for l in open(sys.argv[1]):
    if not l.startswith("{"):
        continue
    r = json.loads(l); g = sorted(r["emitters"], key=lambda e: -e["n"])
    arcs = [(e["n"], e.get("doa_arc80") if e.get("doa_arc80") is not None else float("nan")) for e in g]
    tot = sum(n for n, _ in arcs) or 1
    wide = [(n, a) for n, a in arcs if a == a and a > 30]
    w_pulses = sum(n for n, _ in wide)
    med = sorted(a for _, a in arcs if a == a)
    print(f"DOA {os.path.basename(r['file'])}: {len(g)} groups, {tot} pulses grouped, arc80 median "
          f"{med[len(med)//2] if med else float('nan'):.1f} deg, groups >30deg {len(wide)} "
          f"({100*w_pulses/tot:.0f}% of grouped pulses), max {max(med) if med else 0:.0f} deg")
    if "--groups" in sys.argv:
        for e in g:
            print(f"   n={e['n']:6d} arc80={(e.get("doa_arc80") or float("nan")):6.1f} f={e['f_q5_95'][0]:.0f}-{e['f_q5_95'][1]:.0f} {e['fmode']} lib={e['lib']}")
