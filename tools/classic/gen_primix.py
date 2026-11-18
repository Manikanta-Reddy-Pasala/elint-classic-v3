"""Synthetic PDW CSVs with labelled PRI types (for pri_pattern scoring).

Each file: one target radar (emitter_id 1) with a known PRI type + 1-2 other
radars from gen_freqmodes (fixed carrier). Target PRI types:
  constant, jitter (uniform +-1..20 %), stagger (2-8 elements, some repeats),
  dwell_switch (2-5 PRIs, 16-128 pulses per dwell), sliding (saw / triangle).
Missing pulses 0-20 %; 30 % of files have every other pulse duplicated by a
second receiver (+0.1..0.6 us). Writes <out>/<name>.csv + manifest.json.
SEED=7 tune, SEED=99 holdout. The engine never reads emitter_id.
usage: gen_primix.py <outdir>
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_freqmodes as gf  # noqa: E402

TYPES = ["constant", "jitter", "stagger", "dwell_switch", "sliding"]


def target_gaps(rng, kind):
    """Gap sequence (us) for DUR_S seconds + label."""
    base = rng.uniform(300, 2500)
    n = int(gf.DUR_S * 1e6 / (base * 0.6)) + 50
    lab = dict(type=kind)
    if kind == "constant":
        g = np.full(n, base); lab["pri_us"] = base
    elif kind == "jitter":
        j = rng.uniform(0.01, 0.20)
        g = base * (1 + rng.uniform(-j, j, n)); lab.update(pri_us=base, jitter_pct=j * 100)
    elif kind == "stagger":
        k = int(rng.integers(2, 9))
        while True:
            el = list(base * rng.uniform(0.7, 1.4, k))
            if rng.random() < 0.2 and k >= 3:  # one repeated element
                el[int(rng.integers(1, k))] = el[0]
            vals = sorted(set(round(x, 6) for x in el))
            if all(b / a > 1.04 for a, b in zip(vals, vals[1:])):
                break
        g = np.tile(el, n // k + 1)[:n]
        i0 = int(np.argmin(el))
        lab.update(pri_us=sum(el) / k, period_us=sum(el), elements_us=el[i0:] + el[:i0])
    elif kind == "dwell_switch":
        k = int(rng.integers(2, 6))
        while True:
            pr = list(base * rng.uniform(0.6, 2.5, k))
            v = sorted(pr)
            if all(b / a > 1.08 for a, b in zip(v, v[1:])):
                break
        d = int(rng.integers(16, 129))
        g = np.concatenate([np.full(d, pr[i % k]) for i in range(n // d + 1)])[:n]
        lab.update(pri_us=float(np.mean(pr)), pris_us=sorted(pr), dwell_pulses=d)
    else:  # sliding
        hi = base * rng.uniform(1.1, 1.6); ramp = int(rng.integers(50, 401))
        ph = (np.arange(n) % ramp) / ramp
        if rng.random() < 0.5:  # triangle
            ph = 1 - np.abs(2 * ((np.arange(n) % (2 * ramp)) / (2 * ramp)) - 1)
        g = base + (hi - base) * ph
        lab.update(pri_us=(base + hi) / 2, min_us=base, max_us=hi, ramp_pulses=ramp)
    return g * (1 + rng.normal(0, 0.0005, n)), lab


def target(rng, kind):
    g, lab = target_gaps(rng, kind)
    t = np.cumsum(g) + rng.uniform(0, 1000)
    t = t[t < gf.DUR_S * 1e6]
    miss = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    t = t[rng.random(len(t)) >= miss]
    lab["missing_pct"] = miss * 100
    n = len(t)
    f0 = rng.uniform(2700, 3400)
    spec = dict(f=(f0, f0), pw=float(np.exp(rng.uniform(np.log(0.5), np.log(100)))), bw=rng.uniform(0.5, 5),
                mod=str(rng.choice(gf.MODS)), pri=("x", [lab["pri_us"]]), fmode="fixed")
    return dict(t=t, f=(f0 + rng.normal(0, 0.1, n)) * 1e6, pw=spec["pw"] * 1e3 * (1 + rng.normal(0, 0.01, n)),
                bw=np.maximum(spec["bw"] + rng.normal(0, 0.2, n), 0.05) * 1e6, doa=rng.uniform(0, 360, n),
                pwr=rng.uniform(-80, -60) + rng.normal(0, 0.4, n), mod=np.full(n, spec["mod"]),
                pri=np.full(n, lab["pri_us"] * 1e3), id=np.full(n, 1), src=np.full(n, "omni"), mode="fixed", spec=spec), lab


def dup(e, rng):
    """Second receiver: copy of every other pulse, +0.1..0.6 us."""
    sel = np.arange(len(e["t"])) % 2 == 0
    out = dict(e)
    for k in ["t", "f", "pw", "bw", "doa", "pwr", "mod", "pri", "id", "src"]:
        extra = e[k][sel] + rng.uniform(0.1, 0.6, sel.sum()) if k == "t" else e[k][sel]
        out[k] = np.concatenate([e[k], extra])
    return out


def main(out, n_per=12):
    seed = int(os.environ.get("SEED", 7))
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    man = {}
    for kind in TYPES:
        for k in range(n_per):
            name = f"pri_{kind}_{k}"
            tgt, lab = target(rng, kind)
            ems = [tgt]
            for j in range(int(rng.integers(1, 3))):
                ems.append(gf.emitter(rng, j + 2, "fixed", "omni"))
            if rng.random() < 0.3:
                ems = [dup(e, rng) for e in ems]; lab["dup"] = True
            gf.write(f"{out}/{name}.csv", ems, rng)
            others = [dict(id=j + 2, type={"fixed": "constant"}.get(e["spec"]["pri"][0], e["spec"]["pri"][0]),
                           pri=[float(x) for x in e["spec"]["pri"][1]]) for j, e in enumerate(ems[1:])]
            man[name] = dict(target=lab, others=others)
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1, default=float)
    print(len(man), "files ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
