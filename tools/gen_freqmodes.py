"""Synthetic PDW CSVs for frequency-mode tests (same columns as the corpus).

Modes: fixed, agile_discrete, agile_continuous, burst, sliding, n_modal.
PRI types: fixed, stagger, jitter. DOA: 'omni' (corpus-like, useless) or 'df'.
DFMIX=1: direction-finder test set instead: 2-4 radars of the SAME type
(modulation, PW, BW, carrier band) at different bearings, either a fixed
direction finder ("dfmixstatic", bearing noise 1.5 deg) or a rotating
antenna that hears a radar only while pointing within +-16 deg of it and
reports its own pointing angle ("dfmixrot", like real capture mf6).
Writes <out>/<name>.csv and <out>/manifest.json (truth per file: emitters,
modes, expected library names). The engine never reads emitter_id.
"""
import json, sys, os
import numpy as np

COLS = "time_of_arrival_ns,power_dbm,center_frequency_hz,bandwidth_hz,duration_ns,direction_of_arrival_deg,elevation_deg,type,pulse_repetition_interval_ns,emitter_id,source,dtoa_ns,id,snr_db,sidelobe_blanking_flag_as_int8"
MODS = ["lfm_up", "lfm_down", "unmodulated", "barker"]
DUR_S = 10.0

# synthetic KNOWN agile radars (for naming tests; library file lists them)
KNOWN = {
    2001: dict(name="AGX1", fmode="agile_continuous", f=(3150, 3300), pw=30.0, bw=2.0, mod="lfm_down", pri=("fixed", [1500.0])),
    2002: dict(name="AGX2", fmode="burst", f=(2950, 3050), pw=12.0, bw=1.0, mod="barker", pri=("stagger", [900.0, 1100.0])),
    2003: dict(name="AGX3", fmode="agile_discrete", f=(3350, 3550), pw=4.0, bw=3.0, mod="unmodulated", pri=("jitter", [2500.0])),
}


def toas(rng, pri):
    kind, vals = pri
    n = int(DUR_S * 1e6 / min(vals)) + 10
    if kind == "fixed":
        g = np.full(n, vals[0])
    elif kind == "stagger":
        g = np.tile(vals, n // len(vals) + 1)[:n]
    else:  # jitter 5 %
        g = rng.normal(vals[0], vals[0] * 0.05, n)
    g = g * (1 + rng.normal(0, 0.0005, n))
    t = np.cumsum(g) + rng.uniform(0, vals[0])
    return t[t < DUR_S * 1e6]  # us


def carrier(rng, t, mode, f0, f1):
    n = len(t)
    if mode == "fixed":
        return np.full(n, (f0 + f1) / 2)
    if mode == "agile_continuous":
        return rng.uniform(f0, f1, n)
    if mode == "agile_discrete":
        k = rng.choice([8, 16, 32])
        fs = np.sort(rng.uniform(f0, f1, k))
        return fs[rng.integers(0, k, n)]
    if mode == "n_modal":
        k = rng.choice([2, 3])
        fs = np.linspace(f0, f1, k)
        return np.tile(fs, n // k + 1)[:n]
    if mode == "burst":
        fs = np.linspace(f0, f1, 6)
        out = np.empty(n); i = 0
        while i < n:
            d = int(rng.integers(16, 65)); out[i:i + d] = fs[rng.integers(0, 6)]; i += d
        return out
    if mode == "sliding":
        period = rng.uniform(1e6, 3e6)  # us
        return f0 + (f1 - f0) * ((t % period) / period)
    raise ValueError(mode)


def emitter(rng, eid, mode, doa_kind, spec=None, bearing=None, rot=None):
    if spec is None:
        span = {"fixed": 0, "agile_continuous": rng.uniform(100, 300), "agile_discrete": rng.uniform(100, 250),
                "n_modal": rng.uniform(10, 60), "burst": rng.uniform(60, 150), "sliding": rng.uniform(40, 100)}[mode]
        f0 = rng.uniform(2700, 3400 - span); f1 = f0 + span
        kind = rng.choice(["fixed", "stagger", "jitter"])
        base = rng.uniform(300, 3000)
        vals = [base] if kind != "stagger" else list(base * np.array([1.0, 1.17, 0.86])[: rng.choice([2, 3])])
        spec = dict(f=(f0, f1), pw=float(np.exp(rng.uniform(np.log(0.5), np.log(100)))), bw=rng.uniform(0.5, 5),
                    mod=str(rng.choice(MODS)), pri=(kind, vals), fmode=mode)
    t = toas(rng, spec["pri"])
    n = len(t)
    fr = carrier(rng, t, spec["fmode"], *spec["f"]) + rng.normal(0, 0.1, n)
    doa0 = rng.uniform(0, 360) if bearing is None else bearing
    if rot is not None:  # rotating antenna: (period us, start angle, half beam)
        per, a0, hb = rot
        ant = (a0 + 360.0 * t / per) % 360
        keep = np.abs((ant - doa0 + 180) % 360 - 180) <= hb
        t, fr, ant = t[keep], fr[keep], ant[keep]
        n = len(t)
        doa = (ant + rng.normal(0, 0.05, n)) % 360
    elif doa_kind == "df":
        doa = (doa0 + rng.normal(0, 1.5, n)) % 360
    else:
        doa = rng.uniform(0, 360, n)
    return dict(t=t, f=fr * 1e6, pw=spec["pw"] * 1e3 * (1 + rng.normal(0, 0.01, n)),
                bw=np.maximum(spec["bw"] + rng.normal(0, 0.2, n), 0.05) * 1e6, doa=doa,
                pwr=rng.uniform(-80, -60) + rng.normal(0, 0.4, n), mod=np.full(n, spec["mod"]),
                pri=np.full(n, spec["pri"][1][0] * 1e3), id=np.full(n, eid), mode=spec["fmode"], spec=spec,
                src=np.full(n, "df" if doa_kind == "df" else "omni"))


def write(path, ems, rng):
    keys = ["t", "f", "pw", "bw", "doa", "pwr", "mod", "pri", "id", "src"]
    cat = {k: np.concatenate([e[k] for e in ems]) for k in keys}
    m = max(1, int(len(cat["t"]) * 0.03))  # false alarms
    noise = dict(t=rng.uniform(0, DUR_S * 1e6, m), f=rng.uniform(30, 40000, m) * 1e6, pw=rng.uniform(0.1, 50, m) * 1e3,
                 bw=rng.uniform(1, 20, m) * 1e6, doa=rng.uniform(0, 360, m), pwr=rng.uniform(-90, -60, m),
                 mod=np.full(m, "unknown"), pri=np.zeros(m), id=np.full(m, -1), src=np.full(m, ems[0]["src"][0]))
    cat = {k: np.concatenate([cat[k], noise[k]]) for k in keys}
    o = np.argsort(cat["t"], kind="stable")
    cat = {k: v[o] for k, v in cat.items()}
    toa_ns = np.round((cat["t"] - cat["t"][0]) * 1e3).astype(np.int64)
    dtoa = np.diff(toa_ns, prepend=toa_ns[0])
    with open(path, "w") as fh:
        fh.write(COLS + "\n")
        for i in range(len(toa_ns)):
            fh.write(f"{toa_ns[i]},{cat['pwr'][i]:.4f},{cat['f'][i]:.1f},{cat['bw'][i]:.1f},{cat['pw'][i]:.2f},"
                     f"{cat['doa'][i]:.3f},NaN,{cat['mod'][i]},{int(cat['pri'][i])},{int(cat['id'][i])},{cat['src'][i]},{dtoa[i]},{i+1},"
                     f"{cat['pwr'][i]+100:.3f},4\n")


def dfmix(out, rng, n_per=12):
    """Same-type radars at different bearings (only the bearing tells them apart)."""
    man = {}
    modes = ["fixed", "agile_discrete", "agile_continuous", "burst", "sliding", "n_modal"]
    for kind in ["dfmixstatic", "dfmixrot"]:
        for k in range(n_per):
            ns = int(rng.integers(2, 5))
            # bearings at least 20 deg apart
            while True:
                b = np.sort(rng.uniform(0, 360, ns))
                gaps = np.diff(np.r_[b, b[0] + 360])
                if gaps.min() >= 20:
                    break
            rot = None
            if kind == "dfmixrot":
                rot = (rng.uniform(0.8e6, 2.5e6), rng.uniform(0, 360), 16.0)
            mode = str(rng.choice(modes))
            base = None
            ems = []
            for j in range(ns):
                if base is None:
                    e = emitter(rng, j + 1, mode, "df", None, b[j], rot)
                    base = e["spec"]
                else:
                    spec = dict(base)
                    # same type, own PRI (half of the files: the very same PRI too)
                    if k % 2 == 0:
                        kind_p = rng.choice(["fixed", "stagger", "jitter"])
                        v = rng.uniform(300, 3000)
                        spec["pri"] = (kind_p, [v] if kind_p != "stagger" else list(v * np.array([1.0, 1.17, 0.86])[: rng.choice([2, 3])]))
                    else:
                        spec["pri"] = (base["pri"][0], list(base["pri"][1]))
                    e = emitter(rng, j + 1, mode, "df", spec, b[j], rot)
                ems.append(e)
            # 0-2 other radars of other types at their own bearings
            for j in range(int(rng.integers(0, 3))):
                ems.append(emitter(rng, ns + j + 1, str(rng.choice(modes)), "df", None, rng.uniform(0, 360), rot))
            name = f"{kind}_df_{k}"
            write(f"{out}/{name}.csv", ems, rng)
            man[name] = dict(emitters=len(ems), modes=[e["mode"] for e in ems], known={},
                             pri=[e["spec"]["pri"][0] for e in ems], pulses=[int(len(e["t"])) for e in ems],
                             bearings=[float(x) for x in b], same_type=ns)
    return man


def main(out, n_per=6, seed=7):
    seed = int(os.environ.get("SEED", seed))
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    if os.environ.get("DFMIX") == "1":
        man = dfmix(out, rng)
        json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
        print(len(man), "files ->", out)
        return
    man = {}
    modes = ["fixed", "agile_discrete", "agile_continuous", "burst", "sliding", "n_modal"]
    jobs = []
    for doa_kind in ["omni", "df"]:
        for mode in modes[1:]:
            for k in range(n_per):  # target mode + 1-3 others (any mode)
                others = list(rng.choice(modes, int(rng.integers(1, 4))))
                jobs.append((f"{mode}_{doa_kind}_{k}", [mode] + others, doa_kind, None))
        for kid in KNOWN:  # known agile radar + 1-2 unknown others
            for k in range(3):
                jobs.append((f"known{kid}_{doa_kind}_{k}", [KNOWN[kid]["fmode"]] + list(rng.choice(modes, int(rng.integers(1, 3)))), doa_kind, kid))
        for k in range(4):  # hard: two radars of the SAME type, both agile
            jobs.append((f"hard_sametype_{doa_kind}_{k}", ["agile_continuous", "agile_continuous"], doa_kind, "same"))
    for name, ms, doa_kind, special in jobs:
        ems = []
        for j, m in enumerate(ms):
            spec = None
            if isinstance(special, int) and j == 0:
                spec = dict(KNOWN[special]); spec["pri"] = (spec["pri"][0], list(spec["pri"][1]))
            if special == "same" and j == 1:
                spec = dict(ems[0]["spec"])
            ems.append(emitter(rng, j + 1, m, doa_kind, spec))
        write(f"{out}/{name}.csv", ems, rng)
        man[name] = dict(emitters=len(ems), modes=[e["mode"] for e in ems],
                         known={str(j + 1): special for j in range(1) if isinstance(special, int)},
                         pri=[e["spec"]["pri"][0] for e in ems], pulses=[int(len(e["t"])) for e in ems])
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    print(len(man), "files ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
