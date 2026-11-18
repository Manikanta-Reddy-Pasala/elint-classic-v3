"""Combined per-file time + result report: Python master vs Rust ONNX vs Rust classic.

usage: time_report.py --classic <report.csv> --onnx <report.csv> --python-dir <dir of py json>
                      --expected expected_results.json --out <prefix>

Pipeline time per engine = processing only (CSV parse/load excluded):
  python  : features + model + post-processing over all windows (reads the CSV directly)
  rust    : engine_total - wipe_tables - load_csv  (DB read -> results written)
"""
import argparse, csv, html, json, os, statistics


def load_csv(p):
    return {r["file"]: r for r in csv.DictReader(open(p))} if p and os.path.exists(p) else {}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic"); ap.add_argument("--onnx"); ap.add_argument("--python-dir")
    ap.add_argument("--expected", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp = json.load(open(a.expected))
    C, O = load_csv(a.classic), load_csv(a.onnx)
    rows = []
    for name in sorted(exp):
        e = exp[name]
        c, o = C.get(name, {}), O.get(name, {})
        pj = os.path.join(a.python_dir or "", name[:-4] + ".python.json")
        p = json.load(open(pj)) if a.python_dir and os.path.exists(pj) else {}
        pl = p.get("layers_ms", {})
        r = {
            "file": name, "kind": e["kind"], "expected": e["exp_em"],
            "pulses": f(o.get("pulses_read")) or f(c.get("pulses_read")) or pl.get("pulses"),
            "py_emitters": len(p["emitters"]) if p.get("emitters") is not None else None,
            "onnx_emitters": f(o.get("found_emitters")), "classic_emitters": f(c.get("found_emitters")),
            "onnx_pytest": o.get("pytest"), "classic_pytest": c.get("pytest"),
            "py_signals": p.get("signals"), "onnx_signals": f(o.get("signals")), "classic_signals": f(c.get("signals")),
            "py_clusters": p.get("clusters"), "onnx_clusters": f(o.get("clusters")), "classic_clusters": f(c.get("clusters")),
            # python stages
            "py.features": pl.get("py.features"), "py.model": pl.get("py.model"), "py.postprocess": pl.get("py.postprocess"),
            "py.pipeline": pl.get("py.pipeline"),
            # rust onnx stages
            "onnx.db_read": f(o.get("ms.db_read")), "onnx.features_cpu_sum": f(o.get("ms.onnx.features_cpu_sum")),
            "onnx.model_service_sum": f(o.get("ms.onnx.model_service_sum")),
            "onnx.postprocess_inorder": f(o.get("ms.onnx.postprocess_inorder")), "onnx.db_writes": f(o.get("ms.onnx.db_writes")),
            "onnx.pipeline": f(o.get("ms.pipeline")),
            # rust classic stages
            "classic.db_read": f(c.get("ms.db_read")), "classic.detect": f(c.get("ms.classic.detect_total")),
            "classic.db_writes": f(c.get("ms.classic.db_writes")), "classic.pipeline": f(c.get("ms.pipeline")),
        }
        if r["py.pipeline"] and r["onnx.pipeline"]:
            r["speedup_onnx_vs_py"] = round(r["py.pipeline"] / r["onnx.pipeline"], 1)
        if r["py.pipeline"] and r["classic.pipeline"]:
            r["speedup_classic_vs_py"] = round(r["py.pipeline"] / r["classic.pipeline"], 1)
        rows.append(r)

    cols = list(rows[0].keys()) + ["speedup_onnx_vs_py", "speedup_classic_vs_py"]
    cols = list(dict.fromkeys(cols))
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    def st(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        if not v:
            return None
        v = sorted(v)
        return {"files": len(v), "sum_s": sum(v) / 1e3 if not k.startswith("speedup") else None,
                "mean": statistics.fmean(v), "median": statistics.median(v), "p95": v[min(len(v) - 1, int(.95 * len(v)))], "max": v[-1]}
    keys = [k for k in cols if "." in k or k.startswith("speedup")]
    summ = {k: st(k) for k in keys}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)

    h = ["""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ELINT time report</title><style>
:root{--bg:#fff;--fg:#1d2433;--mut:#5d6677;--line:#e3e6ec;--hd:#f5f7fa}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ee;--mut:#9aa3b2;--line:#2a2f38;--hd:#1b1f26}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px}
table{border-collapse:collapse;font-size:12px} th,td{border:1px solid var(--line);padding:3px 6px;white-space:nowrap;text-align:right}
th{background:var(--hd);position:sticky;top:0;cursor:pointer} td:first-child{text-align:left} .wrap{overflow-x:auto}
.mut{color:var(--mut)}</style></head><body><h1>Time per stage — Python master vs Rust ONNX vs Rust classic</h1>
<p class="mut">ms per file; pipeline excludes CSV parse/load (Python reads the CSV directly; Rust reads pulses back from ClickHouse).
Rust ONNX features/model are summed over the parallel workers, so they exceed the wall-clock pipeline.</p>
<h2>Summary (ms)</h2><table><tr><th>stage</th><th>files</th><th>total s</th><th>mean</th><th>median</th><th>p95</th><th>max</th></tr>"""]
    for k, s in summ.items():
        if s:
            tot = f"{s['sum_s']:.1f}" if s["sum_s"] is not None else ""
            h.append(f"<tr><td>{k}</td><td>{s['files']}</td><td>{tot}</td>" + "".join(f"<td>{s[x]:.1f}</td>" for x in ("mean", "median", "p95", "max")) + "</tr>")
    h.append('</table><h2>Per file</h2><div class="wrap"><table id="t"><tr>' + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>")
    for r in rows:
        h.append("<tr>" + "".join(f"<td>{html.escape('' if r.get(c) is None else str(r.get(c)))}</td>" for c in cols) + "</tr>")
    h.append("""</table></div><script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{const t=th.closest('table'),rs=[...t.rows].slice(1),
d=th.dataset.d=th.dataset.d==='a'?'d':'a';rs.sort((x,y)=>{const a=x.cells[i].innerText,b=y.cells[i].innerText,na=parseFloat(a),nb=parseFloat(b);
const c=(!isNaN(na)&&!isNaN(nb))?na-nb:a.localeCompare(b);return d==='a'?c:-c});rs.forEach(r=>t.appendChild(r))});</script></body></html>""")
    open(a.out + ".html", "w").write("\n".join(h))
    print(json.dumps({k: (round(v["median"], 1) if v else None) for k, v in summ.items()}))


if __name__ == "__main__":
    main()
