import json, sys
for l in sys.stdin:
    if not l.startswith("{"): continue
    r = json.loads(l)
    for g in sorted(r["emitters"], key=lambda g: -g["n"]):
        d = g["doa_q10_50_90"]; f = g["f_q5_95"]
        print(f"  {g['n']:5d}p {g['fmode']:17s} doa {d[0]:6.1f} {d[1]:6.1f} {d[2]:6.1f}  (spread {d[2]-d[0]:5.1f})  f {f[0]:7.1f}-{f[1]:7.1f}")
