"""Build a per-file report for one engine run of the pytest harness.

Inputs
  --report-dir  RUST_REPORT_DIR used for the run (<file>.<engine>.json + .log)
  --junit       pytest --junitxml file of the same run
  --expected    tests/test_data/expected_results.json
  --engine      classic | onnx
  --out         output prefix  (writes <out>.csv and <out>.html)

Timing: the engine prints `LAYER|<stage>|<ms>` lines. CSV parsing and loading
(`wipe_tables`, `load_csv`) is test setup, not processing; the pipeline time is
everything from reading the pulses back out of ClickHouse (`db_read`) to the
last result write:  pipeline_ms = engine_total - wipe_tables - load_csv.
"""
import argparse, csv, html, json, os, re, statistics, xml.etree.ElementTree as ET

LAYERS = {
    "classic": ["db_read", "classic.cells", "classic.components", "classic.split_repair", "classic.coherence",
                "classic.naming", "classic.detect_total", "classic.db_writes", "classic.total_wall"],
    "onnx": ["db_read", "onnx.features_cpu_sum", "onnx.model_service_sum", "onnx.model_per_window",
             "onnx.postprocess_inorder", "onnx.db_writes", "onnx.total_wall"],
}
SETUP = ["wipe_tables", "load_csv"]


def junit_results(path):
    out = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        m = re.search(r"/([^/\[\]]+\.csv)-", tc.get("name", ""))
        if not m:
            continue
        fail = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        skip = tc.find("skipped")
        if fail is not None:
            msg = (fail.get("message") or "").replace("\n", " ")
            mm = re.search(r"\[(.*?)\]", msg)
            out[m.group(1)] = ("FAIL", (mm.group(1) if mm else msg)[:200])
        elif skip is not None:
            # a file also appears in skipped test variants — keep the real result
            out.setdefault(m.group(1), ("SKIP", (skip.get("message") or "")[:200]))
        else:
            out[m.group(1)] = ("PASS", "")
    return out


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return {}
    s = sorted(v)
    return {"mean": statistics.fmean(s), "median": statistics.median(s),
            "p95": s[min(len(s) - 1, int(0.95 * len(s)))], "max": s[-1], "sum": sum(s)}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--report-dir", "--junit", "--expected", "--engine", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    exp = json.load(open(a.expected))
    jr = junit_results(a.junit)
    layers = SETUP + LAYERS[a.engine] + ["engine_total"]
    rows = []
    for name in sorted(exp):
        e = exp[name]
        jp = os.path.join(a.report_dir, f"{os.path.splitext(name)[0]}.{a.engine}.json")
        s = json.load(open(jp)) if os.path.exists(jp) else {}
        lay = s.get("layers", {})
        names = s.get("emitters", [])
        known_found = sorted(n for n in names if n != "unknown_ai")
        exp_ids = e.get("expected_ids") or []
        r = {
            "file": name, "kind": e["kind"],
            "expected_names": "/".join(e.get("expected_names") or []),
            "expected_ids": "/".join(exp_ids),
            "expected_emitters": e["exp_em"],
            "found_emitters": len(names) if s else None,
            "found_known": "/".join(known_found),
            "found_unknown_ai": sum(1 for n in names if n == "unknown_ai"),
            "count_exact": (len(names) == e["exp_em"]) if s else None,
            "known_names_ok": (all(i in names for i in exp_ids) if e["kind"] == "known" else None) if s else None,
            "signals": s.get("signals"), "clusters": s.get("clusters"),
            "clusters_without_signal": s.get("clusters_without_signal"),
            "csv_rows": int(lay["csv_rows"]) if "csv_rows" in lay else None,
            "pulses_read": int(lay["pulses_read"]) if "pulses_read" in lay else None,
            "pulses_in_clusters": s.get("pulses_in_clusters"),
            "pytest": jr.get(name, ("NOT RUN", ""))[0],
            "pytest_reason": jr.get(name, ("", ""))[1],
            "rc": s.get("rc"),
        }
        if r["pulses_read"] and r["pulses_in_clusters"] is not None:
            r["cluster_coverage_pct"] = round(100.0 * r["pulses_in_clusters"] / r["pulses_read"], 1)
        for k in layers:
            r[f"ms.{k}"] = round(lay[k], 2) if k in lay else None
        if "engine_total" in lay:
            r["ms.pipeline"] = round(lay["engine_total"] - sum(lay.get(k, 0) for k in SETUP), 2)
            if r["pulses_read"]:
                r["pulses_per_s"] = round(r["pulses_read"] / max(r["ms.pipeline"], 1e-3) * 1e3)
        if e["kind"] == "unknown":
            r["per_emitter"] = "; ".join(
                f"{p['name']}:pri={p['pri_us'] and round(p['pri_us'],1)} pw={p['pw_us'] and round(p['pw_us'],2)} "
                f"f={p['freq_hz'] and round(p['freq_hz']/1e6,1)}MHz n={p['pulses']}" for p in s.get("per_emitter", []))
        rows.append(r)

    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def rate(key, kind=None):
        sel = [r for r in rows if (kind is None or r["kind"] == kind) and r[key] is not None]
        ok = sum(1 for r in sel if r[key] in (True, "PASS"))
        return ok, len(sel)

    summ = {
        "pytest pass (all)": rate("pytest"),
        "pytest pass (known)": rate("pytest", "known"),
        "pytest pass (unknown)": rate("pytest", "unknown"),
        "known names found": rate("known_names_ok", "known"),
        "exact emitter count (known)": rate("count_exact", "known"),
        "exact emitter count (unknown)": rate("count_exact", "unknown"),
    }
    tstats = {k: stats([r.get(f"ms.{k}") for r in rows]) for k in layers + ["pipeline"]}
    pps = stats([r.get("pulses_per_s") for r in rows])
    with open(a.out + ".summary.json", "w") as fh:
        json.dump({"engine": a.engine, "files": len(rows), "rates": summ, "timing_ms": tstats,
                   "pulses_per_s": pps}, fh, indent=1)

    # HTML
    def td(v):
        if v is True:
            return '<td class="ok">yes</td>'
        if v is False:
            return '<td class="bad">no</td>'
        if v in ("PASS",):
            return '<td class="ok">PASS</td>'
        if v in ("FAIL", "NOT RUN"):
            return f'<td class="bad">{v}</td>'
        return f"<td>{html.escape('' if v is None else str(v))}</td>"
    h = [f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ELINT {a.engine} report</title><style>
:root{{--bg:#fff;--fg:#1d2433;--mut:#5d6677;--line:#e3e6ec;--ok:#1a7f37;--bad:#c62828;--hd:#f5f7fa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171c;--fg:#e6e9ee;--mut:#9aa3b2;--line:#2a2f38;--ok:#4ac26b;--bad:#ff6b6b;--hd:#1b1f26}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px}}
h1{{font-size:20px}} table{{border-collapse:collapse;font-size:12px}} th,td{{border:1px solid var(--line);padding:3px 6px;white-space:nowrap}}
th{{background:var(--hd);position:sticky;top:0;cursor:pointer}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
.wrap{{overflow-x:auto}} .muted{{color:var(--mut)}}</style></head><body>
<h1>ELINT {a.engine} engine — {len(rows)} files</h1><p class="muted">pipeline time = DB read → result writes (CSV parse/load is setup and excluded)</p>
<h2>Summary</h2><table>"""]
    for k, (ok, n) in summ.items():
        h.append(f"<tr><th>{k}</th><td>{ok}/{n}</td><td>{100*ok/max(n,1):.1f}%</td></tr>")
    h.append("</table><h2>Time per layer (ms)</h2><table><tr><th>layer</th><th>mean</th><th>median</th><th>p95</th><th>max</th><th>sum</th></tr>")
    for k, st in tstats.items():
        if st:
            h.append(f"<tr><th>{k}</th>" + "".join(f"<td>{st[x]:.1f}</td>" for x in ("mean", "median", "p95", "max", "sum")) + "</tr>")
    if pps:
        h.append(f"<tr><th>pulses/s (pipeline)</th><td>{pps['mean']:.0f}</td><td>{pps['median']:.0f}</td><td></td><td>{pps['max']:.0f}</td><td></td></tr>")
    h.append('</table><h2>Per file</h2><div class="wrap"><table id="t"><tr>' + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>")
    for r in rows:
        h.append("<tr>" + "".join(td(r.get(c)) for c in cols) + "</tr>")
    h.append("""</table></div><script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{const t=th.closest('table'),rs=[...t.rows].slice(1),
d=th.dataset.d=th.dataset.d==='a'?'d':'a';rs.sort((x,y)=>{const a=x.cells[i].innerText,b=y.cells[i].innerText,na=parseFloat(a),nb=parseFloat(b);
const c=(!isNaN(na)&&!isNaN(nb))?na-nb:a.localeCompare(b);return d==='a'?c:-c});rs.forEach(r=>t.appendChild(r))});</script></body></html>""")
    open(a.out + ".html", "w").write("\n".join(h))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
