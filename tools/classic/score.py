"""Score classic_eval JSON lines. usage: score.py results.jsonl [manifest.json] [--detail]"""
import json, sys, csv, collections, os
res = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
man = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2].endswith(".json") else {}
detail = "--detail" in sys.argv
exp = {}
# Expected results: the test repo (EXP) when present, else the copy frozen in this repo.
EXP_DIR = os.environ.get("EXP", "")
_csvs = [f"{EXP_DIR}/tests/test_data/known_emitters_db_emitter_level.csv", f"{EXP_DIR}/tests/test_data/unknown_emitters_db.csv"]
if EXP_DIR and all(os.path.isfile(p) for p in _csvs):
    for p in _csvs:
        for r in csv.DictReader(open(p)):
            exp[os.path.basename(r["csv_file_path"])] = r
else:
    exp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_608.json")))
tot = collections.Counter(); bymode = collections.defaultdict(collections.Counter); bad = []
for r in res:
    name = os.path.basename(r["file"]); stem = name[:-4]
    truth = {int(k): v for k, v in r["truth_counts"].items() if int(k) > 0 and v >= 50}
    ems = r["emitters"]
    # recovery
    rec = {}
    for e, n in truth.items():
        best = max(ems, key=lambda g: g["truth"].get(str(e), 0), default=None)
        if best is None: rec[e] = False; continue
        have = best["truth"].get(str(e), 0)
        rec[e] = have >= 0.8 * n and have >= 0.8 * best["n"]
    if stem in man:
        m = man[stem]; want = m["emitters"]; lo = hi = want
        for i, e in enumerate(sorted(truth)):
            bymode[m["modes"][i]]["n"] += 1; bymode[m["modes"][i]]["ok"] += rec[e]
        name_ok = None
        if m["known"]:
            kid = int(list(m["known"].values())[0])
            best = max(ems, key=lambda g: g["truth"].get("1", 0))
            name_ok = best["lib"] == kid
            # a known id given to a group that is NOT that radar = wrong name too
            name_ok = name_ok and all(g["lib"] != kid or g["truth"].get("1", 0) >= 0.8 * g["n"] for g in ems)
        grp = stem.split("_")[0] if not stem.startswith(("agile", "hard")) else "_".join(stem.split("_")[:2])
        grp = grp + "_" + ("df" if "_df_" in stem else "omni")
    elif name not in exp:
        want = len(truth); lo = hi = want; name_ok = None
        grp = os.path.basename(os.path.dirname(r["file"]))
        if not truth:  # unlabeled real capture: report only
            print(f"UNLABELED {name}: {r['pulses']} pulses -> {len(ems)} emitters, {r['ms']:.0f} ms;",
                  ", ".join(f"{g['n']}p {g['fmode']} lib={g['lib']}" for g in sorted(ems, key=lambda g: -g['n'])[:12]))
            continue
    else:
        x = exp[name]; want = int(x.get("no_emitters") or x.get("number_emitters")); sd = int(float(x["std_no_emitters"] or 0))
        lo, hi = want - sd, want + sd
        name_ok = None
        if "emitter_id_1" in x:
            libs = {}
            for k in ["1", "2"]:
                best = max(ems, key=lambda g: g["truth"].get(x[f"emitter_{k}_id_csv"], 0))
                libs[k] = best["lib"]
            name_ok = str(libs["1"]) == x["emitter_id_1"] and str(libs["2"]) == x["emitter_id_2"]
        grp = "corpus_known" if "emitter_id_1" in x else "corpus_unknown"
    cnt_ok = lo <= len(ems) <= hi
    all_ok = cnt_ok and all(rec.values()) and name_ok is not False
    for k, v in [("files", 1), ("count_ok", cnt_ok), ("recovered_all", all(rec.values())), ("ok", all_ok)]:
        tot[(grp, k)] += v
    if name_ok is not None: tot[(grp, "names_ok")] += name_ok; tot[(grp, "names_n")] += 1
    tot[(grp, "ms")] += r["ms"]
    if not all_ok: bad.append((stem, len(ems), f"{lo}-{hi}", {e: rec[e] for e in rec}, name_ok))
groups = sorted({g for g, _ in tot})
print(f"{'group':28s} files  ok  count  recov  names   avg_ms")
T = collections.Counter()
for g in groups:
    f = tot[(g, "files")]
    nm = f"{tot[(g,'names_ok')]}/{tot[(g,'names_n')]}" if tot[(g, 'names_n')] else "-"
    print(f"{g:28s} {f:5d} {tot[(g,'ok')]:3d} {tot[(g,'count_ok')]:6d} {tot[(g,'recovered_all')]:6d}  {nm:6s} {tot[(g,'ms')]/f:8.1f}")
    for k in ["files", "ok"]: T[k] += tot[(g, k)]
print(f"{'TOTAL':28s} {T['files']:5d} {T['ok']:3d}")
if bymode:
    print("per true emitter, by frequency mode: recovered / total")
    for m, c in sorted(bymode.items()): print(f"  {m:18s} {c['ok']:4d}/{c['n']:<4d}")
if detail:
    for b in bad: print("FAIL", b)
