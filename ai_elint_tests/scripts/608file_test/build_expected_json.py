#!/usr/bin/env python3
"""Build expected_results_608.json from classic + ONNX result files.

Usage: python3 build_expected_json.py <classic_results.txt> <onnx_results.txt> <output.json>

Result file format (one line per file):
RESULT|<filename.csv>|engine=classic|rc=0|t=1.50s|pg_em=2|pg_sig=2|pg_clu=83|ch_pdw=105181|ch_clu=83|...

Output JSON format:
{
  "generated": "2026-11-11T08:00:00Z",
  "total_files": 608,
  "classic": {"pass": 608, "fail": 0},
  "onnx": {"pass": 159, "fail": 449},
  "results": [
    {
      "file": "mf03_mf04_best-0_plus0.csv",
      "classic": {"rc": 0, "pass": true, "t": "1.50s", "pg_em": 2, "pg_sig": 2, "pg_clu": 83, "ch_pdw": 105181, "ch_clu": 83},
      "onnx": {"rc": 0, "pass": true, "t": "45.34s", "pg_em": 1, "pg_sig": 1, "pg_clu": 1, "ch_pdw": 105181, "ch_clu": 1}
    },
    ...
  ]
}
"""
import json
import re
import sys
from datetime import datetime, timezone


def parse_result_line(line: str) -> dict | None:
    """Parse a RESULT| line into a dict."""
    line = line.strip()
    if not line.startswith("RESULT|"):
        return None
    parts = line.split("|")
    if len(parts) < 3:
        return None
    filename = parts[1]
    # Find engine= in parts
    engine = None
    rc = None
    t = None
    pg_em = None
    pg_sig = None
    pg_clu = None
    ch_pdw = None
    ch_clu = None
    for p in parts:
        if p.startswith("engine="):
            engine = p.split("=", 1)[1]
        elif p.startswith("rc="):
            try:
                rc = int(p.split("=", 1)[1])
            except ValueError:
                rc = None
        elif p.startswith("t="):
            t = p.split("=", 1)[1]
        elif p.startswith("pg_em="):
            try:
                pg_em = int(p.split("=", 1)[1])
            except ValueError:
                pg_em = None
        elif p.startswith("pg_sig="):
            try:
                pg_sig = int(p.split("=", 1)[1])
            except ValueError:
                pg_sig = None
        elif p.startswith("pg_clu="):
            try:
                pg_clu = int(p.split("=", 1)[1])
            except ValueError:
                pg_clu = None
        elif p.startswith("ch_pdw="):
            try:
                ch_pdw = int(p.split("=", 1)[1])
            except ValueError:
                ch_pdw = None
        elif p.startswith("ch_clu="):
            try:
                ch_clu = int(p.split("=", 1)[1])
            except ValueError:
                ch_clu = None
    if engine is None:
        return None
    return {
        "file": filename,
        "engine": engine,
        "rc": rc,
        "pass": (rc == 0 and pg_em is not None and pg_em > 0),
        "t": t,
        "pg_em": pg_em,
        "pg_sig": pg_sig,
        "pg_clu": pg_clu,
        "ch_pdw": ch_pdw,
        "ch_clu": ch_clu,
    }


def load_results(path: str) -> dict:
    """Load a results file, returning {filename: result_dict}."""
    results = {}
    try:
        with open(path) as f:
            for line in f:
                parsed = parse_result_line(line)
                if parsed and parsed["engine"]:
                    results[parsed["file"]] = parsed
    except FileNotFoundError:
        pass
    return results


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    classic_path, onnx_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    classic = load_results(classic_path)
    onnx = load_results(onnx_path)

    # Union of all filenames, sorted
    all_files = sorted(set(list(classic.keys()) + list(onnx.keys())))

    results = []
    for fn in all_files:
        c = classic.get(fn)
        o = onnx.get(fn)
        entry = {"file": fn}
        if c:
            entry["classic"] = {
                "rc": c["rc"],
                "pass": c["pass"],
                "t": c["t"],
                "pg_em": c["pg_em"],
                "pg_sig": c["pg_sig"],
                "pg_clu": c["pg_clu"],
                "ch_pdw": c["ch_pdw"],
                "ch_clu": c["ch_clu"],
            }
        if o:
            entry["onnx"] = {
                "rc": o["rc"],
                "pass": o["pass"],
                "t": o["t"],
                "pg_em": o["pg_em"],
                "pg_sig": o["pg_sig"],
                "pg_clu": o["pg_clu"],
                "ch_pdw": o["ch_pdw"],
                "ch_clu": o["ch_clu"],
            }
        results.append(entry)

    c_pass = sum(1 for r in results if r.get("classic", {}).get("pass"))
    c_fail = sum(1 for r in results if r.get("classic") and not r["classic"].get("pass"))
    o_pass = sum(1 for r in results if r.get("onnx", {}).get("pass"))
    o_fail = sum(1 for r in results if r.get("onnx") and not r["onnx"].get("pass"))

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_files": len(results),
        "classic": {"pass": c_pass, "fail": c_fail, "total": len(classic)},
        "onnx": {"pass": o_pass, "fail": o_fail, "total": len(onnx)},
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Total files: {len(results)}")
    print(f"Classic: {c_pass} pass, {c_fail} fail, {len(classic)} total")
    print(f"ONNX:    {o_pass} pass, {o_fail} fail, {len(onnx)} total")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
