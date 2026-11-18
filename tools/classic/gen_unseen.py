"""Unseen-radar test set: real corpus files (2 known library radars each)
plus 1-2 synthetic radars of types that are NOT in the naming library.

Classes (file name prefix):
  far      - carrier outside every library band, any frequency mode
  carrier  - fixed carrier INSIDE a library radar's band; PW / BW /
             modulation / PRI all differ from that radar
  pri      - same carrier band, same modulation and BW as a library radar,
             PRI 15-45 % away from its PRI (only the pulse train differs)
  two      - one far + one carrier radar in the same file
New radars get emitter_id 101, 102. The engine never reads emitter_id.
usage: SEED=7 gen_unseen.py <corpus_known_dir> <outdir> [files_per_class]
Writes <out>/<name>.csv and <out>/manifest.json.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_freqmodes as gf

# Library radars as measured on the corpus (carrier MHz, PRI us, BW MHz,
# main modulation). Used only to place "close" look-alikes.
LIB = {
    1113: dict(name="MF03", f=(2898.0, 2905.0), pri=294.0, bw=1.7, mod="lfm_up"),
    1106: dict(name="PSR3", f=(2848.0, 2852.0), pri=75.0, bw=1.0, mod="unmodulated"),
    1104: dict(name="PSR1", f=(2748.0, 2752.0), pri=1105.0, bw=2.0, mod="lfm_down"),
    1116: dict(name="PSR5m2", f=(2798.0, 2822.0), pri=130.0, bw=5.0, mod="lfm_up"),
    1109: dict(name="MF04", f=(2858.0, 2892.0), pri=307.0, bw=4.0, mod="lfm_up"),
    1105: dict(name="PSR2", f=(2874.0, 2876.5), pri=116.0, bw=1.0, mod="unmodulated"),
    1108: dict(name="PSR5m1", f=(2858.0, 2892.0), pri=147.0, bw=4.0, mod="lfm_down"),
}
LIB_BANDS = [(2740.0, 2760.0), (2790.0, 2830.0), (2840.0, 2910.0)]
MODES = ["fixed", "agile_discrete", "agile_continuous", "burst", "sliding", "n_modal"]


def pri_spec(rng, base):
    kind = str(rng.choice(["fixed", "stagger", "jitter"]))
    vals = [base] if kind != "stagger" else list(base * np.array([1.0, 1.17, 0.86])[: int(rng.choice([2, 3]))])
    return (kind, [float(v) for v in vals])


def far_spec(rng):
    mode = str(rng.choice(MODES))
    span = {"fixed": 0, "agile_continuous": rng.uniform(100, 300), "agile_discrete": rng.uniform(100, 250),
            "n_modal": rng.uniform(10, 60), "burst": rng.uniform(60, 150), "sliding": rng.uniform(40, 100)}[mode]
    while True:  # whole span clear of every library band (+-10 MHz)
        f0 = rng.uniform(2400, 3800 - span)
        if all(f0 + span < lo - 10 or f0 > hi + 10 for lo, hi in LIB_BANDS):
            break
    return dict(f=(f0, f0 + span), pw=float(np.exp(rng.uniform(np.log(0.5), np.log(150)))), bw=float(rng.uniform(0.5, 5)),
                mod=str(rng.choice(gf.MODS)), pri=pri_spec(rng, rng.uniform(80, 3000)), fmode=mode)


def away(rng, pri):
    return pri * (rng.uniform(1.15, 1.45) if rng.random() < 0.5 else rng.uniform(0.55, 0.85))


def carrier_spec(rng, lib):
    r = LIB[lib]
    fc = rng.uniform(r["f"][0] + 0.5, r["f"][1] - 0.5)
    mods = [m for m in gf.MODS if m != r["mod"]]
    bw = r["bw"] + rng.choice([-1, 1]) * rng.uniform(1.0, 2.0)
    return dict(f=(fc, fc), pw=float(np.exp(rng.uniform(np.log(0.5), np.log(150)))), bw=float(max(bw, 0.4)),
                mod=str(rng.choice(mods)), pri=pri_spec(rng, away(rng, r["pri"])), fmode="fixed")


def pri_only_spec(rng, lib):
    r = LIB[lib]
    fc = rng.uniform(r["f"][0] + 0.5, r["f"][1] - 0.5)
    return dict(f=(fc, fc), pw=float(np.exp(rng.uniform(np.log(1.0), np.log(100)))), bw=r["bw"], mod=r["mod"],
                pri=("fixed", [away(rng, r["pri"])]), fmode="fixed")


def main(src, out, per=10):
    seed = int(os.environ.get("SEED", 7))
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.endswith(".csv"))
    man = {}
    for cls in ["far", "carrier", "pri", "two"]:
        for k in range(per):
            base = files[int(rng.integers(0, len(files)))]
            df = pd.read_csv(os.path.join(src, base))
            t0 = int(df.time_of_arrival_ns.min())
            gf.DUR_S = (df.time_of_arrival_ns.max() - t0) / 1e9
            lib = int(rng.choice(list(LIB)))
            specs = {"far": [far_spec(rng)], "carrier": [carrier_spec(rng, lib)], "pri": [pri_only_spec(rng, lib)],
                     "two": [far_spec(rng), carrier_spec(rng, lib)]}[cls]
            for sp in specs:  # physical duty cycle: pulse width <= 10 % of the shortest PRI
                sp["pw"] = float(min(sp["pw"], 0.1 * min(sp["pri"][1])))
            rows = []
            for j, sp in enumerate(specs):
                e = gf.emitter(rng, 101 + j, sp["fmode"], "omni", sp)
                rows.append(pd.DataFrame({
                    "time_of_arrival_ns": np.round(e["t"] * 1e3).astype(np.int64) + t0,
                    "power_dbm": e["pwr"], "center_frequency_hz": e["f"], "bandwidth_hz": e["bw"], "duration_ns": e["pw"],
                    "direction_of_arrival_deg": e["doa"], "elevation_deg": np.nan, "type": e["mod"],
                    "pulse_repetition_interval_ns": e["pri"].astype(np.int64), "emitter_id": e["id"], "source": "omni",
                    "dtoa_ns": 0, "id": 0, "snr_db": e["pwr"] + 100, "sidelobe_blanking_flag_as_int8": 4}))
            m = pd.concat([df] + rows, ignore_index=True).sort_values("time_of_arrival_ns", kind="stable")
            m["dtoa_ns"] = m.time_of_arrival_ns.diff().fillna(0).astype(np.int64)
            m["id"] = np.arange(1, len(m) + 1)
            name = f"{cls}_{k:02d}"
            m.to_csv(f"{out}/{name}.csv", index=False, na_rep="NaN")
            man[name] = dict(base=base, cls=cls, near_lib=lib if cls != "far" else None,
                             new=[dict(id=101 + j, fmode=s["fmode"], f=[round(x, 2) for x in s["f"]], pw=round(s["pw"], 2),
                                       bw=round(s["bw"], 2), mod=s["mod"], pri=s["pri"]) for j, s in enumerate(specs)])
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)
    print(len(man), "files ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
