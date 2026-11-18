"""Per-file difference report: Rust ONNX vs Rust classic (same 608 files).

usage: compare_engines.py --onnx <report.csv> --classic <report.csv> --out <prefix>
Writes <prefix>.csv, <prefix>.html, <prefix>.summary.json.
"""
import argparse, csv, html, json, statistics


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--classic", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O = {r["file"]: r for r in csv.DictReader(open(a.onnx))}
    C = {r["file"]: r for r in csv.DictReader(open(a.classic))}
    rows = []
    for name in sorted(set(O) | set(C)):
        o, c = O.get(name, {}), C.get(name, {})
        ok_ = lambda r: r.get("pytest") == "PASS"
        r = {
            "file": name, "kind": o.get("kind") or c.get("kind"),
            "expected_names": o.get("expected_names") or c.get("expected_names"),
            "expected_emitters": f(o.get("expected_emitters") or c.get("expected_emitters")),
            "pulses": f(o.get("pulses_read") or c.get("pulses_read")),
            "onnx_pytest": o.get("pytest"), "classic_pytest": c.get("pytest"),
            "pytest_same": o.get("pytest") == c.get("pytest"),
            "onnx_emitters": f(o.get("found_emitters")), "classic_emitters": f(c.get("found_emitters")),
            "onnx_known": o.get("found_known"), "classic_known": c.get("found_known"),
            "known_same": o.get("found_known") == c.get("found_known"),
            "onnx_signals": f(o.get("signals")), "classic_signals": f(c.get("signals")),
            "onnx_clusters": f(o.get("clusters")), "classic_clusters": f(c.get("clusters")),
            "onnx_pulses_in_clusters": f(o.get("pulses_in_clusters")), "classic_pulses_in_clusters": f(c.get("pulses_in_clusters")),
            "onnx_coverage_pct": f(o.get("cluster_coverage_pct")), "classic_coverage_pct": f(c.get("cluster_coverage_pct")),
            "onnx_pipeline_ms": f(o.get("ms.pipeline")), "classic_pipeline_ms": f(c.get("ms.pipeline")),
            "onnx_db_read_ms": f(o.get("ms.db_read")), "classic_db_read_ms": f(c.get("ms.db_read")),
            "onnx_db_writes_ms": f(o.get("ms.onnx.db_writes")), "classic_db_writes_ms": f(c.get("ms.classic.db_writes")),
            "onnx_model_sum_ms": f(o.get("ms.onnx.model_service_sum")), "classic_detect_ms": f(c.get("ms.classic.detect_total")),
        }
        for k in ("emitters", "signals", "clusters", "pulses_in_clusters"):
            if r[f"onnx_{k}"] is not None and r[f"classic_{k}"] is not None:
                r[f"diff_{k}"] = r[f"onnx_{k}"] - r[f"classic_{k}"]
        if r["expected_emitters"] is not None:
            for e in ("onnx", "classic"):
                if r[f"{e}_emitters"] is not None:
                    r[f"{e}_count_error"] = r[f"{e}_emitters"] - r["expected_emitters"]
        if r["onnx_pipeline_ms"] and r["classic_pipeline_ms"]:
            r["diff_pipeline_ms"] = round(r["onnx_pipeline_ms"] - r["classic_pipeline_ms"], 1)
            r["onnx_over_classic_time"] = round(r["onnx_pipeline_ms"] / r["classic_pipeline_ms"], 1)
        rows.append(r)

    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    def tot(k):
        return sum(r[k] for r in rows if r.get(k) is not None)

    def med(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return statistics.median(v) if v else None
    n = len(rows)
    summ = {
        "files": n,
        "pytest pass onnx": sum(r["onnx_pytest"] == "PASS" for r in rows),
        "pytest pass classic": sum(r["classic_pytest"] == "PASS" for r in rows),
        "files where pytest result differs": sum(not r["pytest_same"] for r in rows),
        "files where known names differ": sum(not r["known_same"] for r in rows),
        "files with same emitter count": sum(r.get("diff_emitters") == 0 for r in rows),
        "exact count onnx": sum(r.get("onnx_count_error") == 0 for r in rows),
        "exact count classic": sum(r.get("classic_count_error") == 0 for r in rows),
        "total pulses": tot("pulses"),
        "total emitters onnx / classic / expected": [tot("onnx_emitters"), tot("classic_emitters"), tot("expected_emitters")],
        "total signals onnx / classic": [tot("onnx_signals"), tot("classic_signals")],
        "total clusters onnx / classic": [tot("onnx_clusters"), tot("classic_clusters")],
        "total pulses in clusters onnx / classic": [tot("onnx_pulses_in_clusters"), tot("classic_pulses_in_clusters")],
        "total pipeline s onnx / classic": [round(tot("onnx_pipeline_ms") / 1e3, 1), round(tot("classic_pipeline_ms") / 1e3, 1)],
        "median pipeline ms onnx / classic": [med("onnx_pipeline_ms"), med("classic_pipeline_ms")],
        "median onnx/classic time ratio": med("onnx_over_classic_time"),
        "total db_read s onnx / classic": [round(tot("onnx_db_read_ms") / 1e3, 2), round(tot("classic_db_read_ms") / 1e3, 2)],
        "total db_writes s onnx / classic": [round(tot("onnx_db_writes_ms") / 1e3, 2), round(tot("classic_db_writes_ms") / 1e3, 2)],
    }
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)

    def td(k, v):
        if isinstance(v, bool):
            return f'<td class="{"ok" if v else "bad"}">{"yes" if v else "no"}</td>'
        if v in ("PASS",):
            return '<td class="ok">PASS</td>'
        if v in ("FAIL", "NOT RUN"):
            return f'<td class="bad">{v}</td>'
        cls = ' class="num"' if isinstance(v, (int, float)) else ""
        if k.startswith("diff_") and isinstance(v, (int, float)) and v != 0:
            cls = ' class="num diff"'
        return f"<td{cls}>{html.escape('' if v is None else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))}</td>"
    h = ["""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ONNX vs Classic</title><style>
:root{--bg:#fff;--fg:#1d2433;--mut:#5d6677;--line:#e3e6ec;--ok:#1a7f37;--bad:#c62828;--hd:#f5f7fa;--hl:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ee;--mut:#9aa3b2;--line:#2a2f38;--ok:#4ac26b;--bad:#ff6b6b;--hd:#1b1f26;--hl:#3a3218}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px}
table{border-collapse:collapse;font-size:12px} th,td{border:1px solid var(--line);padding:3px 6px;white-space:nowrap}
th{background:var(--hd);position:sticky;top:0;cursor:pointer} .num{text-align:right} .diff{background:var(--hl)}
.ok{color:var(--ok)} .bad{color:var(--bad)} .wrap{overflow-x:auto} .mut{color:var(--mut)}</style></head><body>
<h1>Rust ONNX vs Rust classic — per file</h1><p class="mut">diff_* = ONNX − classic (highlighted when not 0). Times in ms; pipeline = DB read → results written.</p>
<h2>Totals</h2><table>"""]
    for k, v in summ.items():
        h.append(f"<tr><th style='text-align:left'>{html.escape(k)}</th><td class='num'>{html.escape(str(v))}</td></tr>")
    h.append('</table><h2>Per file</h2><div class="wrap"><table id="t"><tr>' + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>")
    for r in rows:
        h.append("<tr>" + "".join(td(c, r.get(c)) for c in cols) + "</tr>")
    h.append("""</table></div><script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{const t=th.closest('table'),rs=[...t.rows].slice(1),
d=th.dataset.d=th.dataset.d==='a'?'d':'a';rs.sort((x,y)=>{const a=x.cells[i].innerText,b=y.cells[i].innerText,na=parseFloat(a),nb=parseFloat(b);
const c=(!isNaN(na)&&!isNaN(nb))?na-nb:a.localeCompare(b);return d==='a'?c:-c});rs.forEach(r=>t.appendChild(r))});</script></body></html>""")
    open(a.out + ".html", "w").write("\n".join(h))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
