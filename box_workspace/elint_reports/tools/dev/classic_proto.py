"""Prototype for the label-free classic engine. Labels are read ONLY to score.

usage: python classic_proto.py <expected_results.json> <csv>...
Prints per file: groups found, exact-count match, known names correct, purity.
"""
import json, sys, collections
import numpy as np, pandas as pd

P = dict(min_cell=20, f_cell=1e6, pw_rel=0.01, bw_cell=1e6, f_adj=2, merge_f=3e6, merge_bw=1.0e6, min_group=50, agile_pw_ns=100.0, agile_bw=0.3e6, merge_pw_rel=0.02, coh_f=40e6, coh_thr=0.13, coh_top=4, coh_min=0.04)

def decision_tree(g):
    """Library id from measured group stats (carrier mode, PRI from TOA, max PW, lfm_up share, BW mode)."""
    f = np.round(g.center_frequency_hz / 1e5); f_mhz = f.mode().iloc[0] / 10
    bw = np.round(g.bandwidth_hz / 1e5); bw_mhz = bw.mode().iloc[0] / 10
    pw_max = (g.duration_ns / 1e3).quantile(0.99)
    lfm = (g.type.str.lower().str.replace("_", "") == "lfmup").mean()
    pri = float(g.pri_toa.iloc[0])
    if bw_mhz >= 6.0: return None
    if 2898 <= f_mhz <= 2905: return "1113"
    if 2848 <= f_mhz <= 2852: return "1106"
    if 2748 <= f_mhz <= 2752: return "1107" if pw_max >= 86 else "1104"
    if 2798 <= f_mhz <= 2822: return "1116"
    if 2858 <= f_mhz <= 2892:
        if lfm >= 0.8: return "1109"
        if 2874 <= f_mhz <= 2876.5 and pri < 140: return "1105"
        return "1108"
    return None

def pri_from_toa(toa_ns):
    t = np.sort(toa_ns.to_numpy(dtype=np.float64))
    d = np.diff(t) / 1e3
    d = d[d > 0.5]
    if len(d) < 3: return 0.0
    h = np.round(d).astype(np.int64)
    vals, cnt = np.unique(h, return_counts=True)
    return float(vals[np.argmax(cnt)])

def coherence(ta, tb, n=3000):
    """Share of A pulses whose offset to the preceding B pulse falls in the top 1-us bin."""
    if len(ta) < 20 or len(tb) < 20: return 0.0
    s = ta[np.linspace(0, len(ta) - 1, min(n, len(ta))).astype(int)]
    j = np.searchsorted(tb, s) - 1
    ok = j >= 0
    off = np.round((s[ok] - tb[j[ok]]) / 1e3).astype(np.int64)
    if len(off) == 0: return 0.0
    _, c = np.unique(off, return_counts=True)
    return float(np.sort(c)[-P['coh_top']:].sum() / len(off))

def classic(d):
    d = d[~d.type.str.lower().str.startswith("unk")].copy()
    d["fc"] = np.round(d.center_frequency_hz / P["f_cell"]).astype(int)
    d["wc"] = np.round(np.log(np.maximum(d.duration_ns / 1e3, 0.05)) / np.log1p(P["pw_rel"])).astype(int)
    d["bc"] = np.round(d.bandwidth_hz / P["bw_cell"]).astype(int)
    d["mc"] = d.type.str.lower()
    cells = d.groupby(["mc", "wc", "bc", "fc"]).size()
    cells = cells[cells >= P["min_cell"]]
    keys = list(cells.index); idx = {k: i for i, k in enumerate(keys)}
    parent = list(range(len(keys)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for k in keys:
        m, w, b, f = k
        for dw in (-1, 0, 1):
            for db in (-1, 0, 1):
                for df in range(-P["f_adj"], P["f_adj"] + 1):
                    j = idx.get((m, w + dw, b + db, f + df))
                    if j is not None: parent[find(idx[k])] = find(j)
    comp = {k: find(idx[k]) for k in keys}
    d["comp"] = [comp.get(k, -1) for k in zip(d.mc, d.wc, d.bc, d.fc)]
    d = d[d.comp >= 0]
    # merge components of one multi-mode emitter: carrier band + BW overlap
    cs = d.groupby("comp").agg(f_lo=("center_frequency_hz", lambda x: x.quantile(.02)), f_hi=("center_frequency_hz", lambda x: x.quantile(.98)),
                               b_lo=("bandwidth_hz", lambda x: x.quantile(.02)), b_hi=("bandwidth_hz", lambda x: x.quantile(.98)), n=("comp", "size"),
                               pw=("duration_ns", "median"))
    ids = list(cs.index); par = {c: c for c in ids}
    def f2(x):
        while par[x] != x: x = par[x]
        return x
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            A, B = cs.loc[ids[a]], cs.loc[ids[b]]
            if A.f_lo - P["merge_f"] <= B.f_hi and B.f_lo - P["merge_f"] <= A.f_hi and \
               A.b_lo - P["merge_bw"] <= B.b_hi and B.b_lo - P["merge_bw"] <= A.b_hi and \
               abs(A.pw - B.pw) <= P["merge_pw_rel"] * min(A.pw, B.pw):
                par[f2(ids[a])] = f2(ids[b])
    d["grp"] = d.comp.map(lambda c: f2(c))
    sizes = d.groupby("grp").size(); d = d[d.grp.isin(sizes[sizes >= P["min_cell"]].index)]
    # multi-mode radars: modes of one radar keep a fixed TOA relation
    toas = {g: np.sort(x.time_of_arrival_ns.to_numpy(np.float64)) for g, x in d.groupby("grp")}
    gs = d.groupby("grp").agg(f=("center_frequency_hz", "median"), bw=("bandwidth_hz", "median"))
    gids = list(gs.index); par3 = {g: g for g in gids}
    def f4(x):
        while par3[x] != x: x = par3[x]
        return x
    for a in range(len(gids)):
        for b in range(a + 1, len(gids)):
            A, B = gs.loc[gids[a]], gs.loc[gids[b]]
            if True:
                c1, c2 = coherence(toas[gids[a]], toas[gids[b]]), coherence(toas[gids[b]], toas[gids[a]])
                if max(c1, c2) >= P["coh_thr"] and min(c1, c2) >= P["coh_min"]:
                    par3[f4(gids[a])] = f4(gids[b])
    d["grp"] = d.grp.map(f4)
    sizes = d.groupby("grp").size(); d = d[d.grp.isin(sizes[sizes >= P["min_group"]].index)]
    names = {}
    for gid, g in d.groupby("grp"):
        g = g.assign(pri_toa=pri_from_toa(g.time_of_arrival_ns))
        names[gid] = decision_tree(g)
    # one emitter per library id; every other group is its own unknown_ai emitter
    emit = {}
    for gid, nm in names.items():
        emit[gid] = nm if nm else f"unk{gid}"
    d["emitter"] = d.grp.map(emit)
    return d

def score(d, truth):
    em = d.emitter.unique()
    n_known = sum(1 for e in em if not e.startswith("unk"))
    n = n_known + sum(1 for e in em if e.startswith("unk"))
    # purity: share of each GT label's pulses in its majority emitter
    pur = d.groupby("emitter_id").emitter.agg(lambda x: x.value_counts().iloc[0] / len(x))
    return n, sorted(e for e in em if not e.startswith("unk")), float(pur.min())

def one(f):
    name = f.split("/")[-1]
    try:
        d = classic(pd.read_csv(f))
        return name, score(d, None)
    except Exception as ex:
        return name, (0, [], 0.0, repr(ex))

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--all":
    import multiprocessing as mp, glob
    exp = json.load(open(sys.argv[1]))
    files = sorted(glob.glob(sys.argv[3] + "/*/*.csv"))
    with mp.Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 16) as pool:
        res = pool.map(one, files, chunksize=1)
    tot = collections.Counter(); bad = []
    for name, r in res:
        e = exp[name]; n, known = r[0], r[1]
        names_ok = (e["kind"] != "known") or all(x in known for x in e["expected_ids"])
        cnt_ok = n == e["exp_em"]
        tot[e["kind"] + "_files"] += 1; tot[e["kind"] + "_names_ok"] += names_ok; tot[e["kind"] + "_count_ok"] += cnt_ok
        if not (names_ok and cnt_ok): bad.append(f"{name} exp={e['exp_em']} got={n} known={known} pur={r[2]:.2f} {r[3] if len(r)>3 else ''}")
    print("\n".join(bad)); print(dict(tot)); sys.exit(0)

exp = json.load(open(sys.argv[1]))
tot = collections.Counter()
for f in sys.argv[2:]:
    name = f.split("/")[-1]; e = exp[name]
    d = classic(pd.read_csv(f))
    n, known, pur = score(d, e)
    names_ok = (e["kind"] != "known") or all(x in known for x in e["expected_ids"])
    cnt_ok = n == e["exp_em"]
    tot["files"] += 1; tot["names_ok"] += names_ok; tot["count_ok"] += cnt_ok
    print(f"{name:40s} exp={e['exp_em']} got={n} names_ok={names_ok} known={known} purity_min={pur:.2f}")
print(dict(tot))

def debug(f):
    d = classic(pd.read_csv(f))
    g = d.groupby(["emitter", "emitter_id"]).agg(n=("grp", "size"), grps=("grp", lambda x: sorted(x.unique())),
        f=("center_frequency_hz", lambda x: f"{x.min()/1e6:.1f}-{x.max()/1e6:.1f}"),
        pw=("duration_ns", lambda x: f"{x.min()/1e3:.1f}-{x.max()/1e3:.1f}"),
        bw=("bandwidth_hz", lambda x: f"{x.min()/1e6:.2f}-{x.max()/1e6:.2f}"), mc=("mc", lambda x: ",".join(sorted(x.unique()))))
    print(g.to_string())
