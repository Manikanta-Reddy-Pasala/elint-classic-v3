#!/usr/bin/env python3
"""Build a real, repo-accurate expected-results file for the 608-file sweep.

The ai_elint_tests pytest harness validates each CSV by DB queries, using the
per-file *parameter* CSVs as the source of truth:

  known  tests/test_data/known_emitters_db_emitter_level.csv   (448 rows)
  unknown tests/test_data/unknown_emitters_db.csv              (160 rows)

For every file the harness asserts (tests/test_pdw_generator.py):
  * known   -> each named emitter exists:
               SELECT count(*) FROM emitters WHERE name = '<emitter_id_N>'  > 0
               and  SELECT count(*) FROM clusters WHERE signal_id IS NULL  == 0
  * unknown -> at least `number_emitters` emitters detected, and each emitter's
               average PRI / PW / frequency falls inside its spec range
               (helpers/emitter_specifications/unknown_emitters.py)

This script turns those parameter CSVs into ONE authoritative JSON that a
validation pass can compare the actual DB results against.  The expected
count for a file is the parameter CSV's `no_emitters` / `number_emitters`
value.  `pass` for an engine is true when the engine ran (rc==0) and the
number of emitters it actually produced equals that expected count.

Usage:
  python3 build_expected_from_params.py \
      --repo /path/to/ai_elint_tests \
      --classic /home/sysadm/master_608_classic_results.txt \
      --onnx    /home/sysadm/master_608_onnx_results.txt \
      --out     tests/test_data/expected_results.json

The classic/onnx result files may be omitted; when omitted the JSON holds the
expected values only (no measured engine results, no pass/fail per engine).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

KNOWN_PARAM = "tests/test_data/known_emitters_db_emitter_level.csv"
UNKNOWN_PARAM = "tests/test_data/unknown_emitters_db.csv"

# Unknown-emitter spec ranges live here in the repo.
SPEC_RANGES = {
    "CTA1": {"frequency": [3000000000, 3175000000], "pri": [129, 253], "pw": [21.5, 22.5]},
    "CTA2": {"frequency": [2930000000, 3110000000], "pri": [1875, 2271], "pw": [6.0, 7.0]},
    "NAV1": {"frequency": [2950000000, 3051000000], "pri": [101, 176], "pw": [0.51, 0.61]},
    "NAV2": {"frequency": [3045000000, 3055000000], "pri": [217, 221], "pw": [0.65, 0.75]},
    "NAV3": {"frequency": [3000000000, 3100000000], "pri": [168, 208], "pw": [22, 23]},
}


def load_param_map(repo: Path) -> dict:
    """Return {filename: expected_entry} from the two main parameter CSVs."""
    expected = {}

    known_csv = repo / KNOWN_PARAM
    with known_csv.open() as f:
        for row in csv.DictReader(f):
            fn = row["csv_file_path"].split("/")[-1]
            expected[fn] = {
                "file": fn,
                "type": "known",
                "emitters": [row["emitter_1_name"], row["emitter_2_name"]],
                "emitter_ids": [row["emitter_id_1"], row["emitter_id_2"]],
                "expected_emitters": int(row["no_emitters"]),
                "std_no_emitters": int(row["std_no_emitters"]),
            }

    unknown_csv = repo / UNKNOWN_PARAM
    with unknown_csv.open() as f:
        for row in csv.DictReader(f):
            fn = row["csv_file_path"].split("/")[-1]
            expected[fn] = {
                "file": fn,
                "type": "unknown",
                "emitters": [row["emitter_1_name"], row["emitter_2_name"]],
                "emitter_ids": None,  # unknown emitters are matched by spec, not id
                "expected_emitters": int(row["number_emitters"]),
                "std_no_emitters": int(row["std_no_emitters"]),
            }

    return expected


def parse_result_line(line: str) -> dict | None:
    """Parse one RESULT| line from a batch result file."""
    line = line.strip()
    if not line.startswith("RESULT|"):
        return None
    parts = line.split("|")
    if len(parts) < 3:
        return None
    filename = parts[1]
    d = {"file": filename}
    for p in parts:
        if p.startswith("engine="):
            d["engine"] = p.split("=", 1)[1]
        elif p.startswith("rc="):
            try:
                d["rc"] = int(p.split("=", 1)[1])
            except ValueError:
                d["rc"] = None
        elif p.startswith("t="):
            d["t"] = p.split("=", 1)[1]
        elif p.startswith("pg_em="):
            try:
                d["pg_em"] = int(p.split("=", 1)[1])
            except ValueError:
                d["pg_em"] = None
        elif p.startswith("pg_sig="):
            try:
                d["pg_sig"] = int(p.split("=", 1)[1])
            except ValueError:
                d["pg_sig"] = None
        elif p.startswith("pg_clu="):
            try:
                d["pg_clu"] = int(p.split("=", 1)[1])
            except ValueError:
                d["pg_clu"] = None
        elif p.startswith("ch_pdw="):
            try:
                d["ch_pdw"] = int(p.split("=", 1)[1])
            except ValueError:
                d["ch_pdw"] = None
        elif p.startswith("ch_clu="):
            try:
                d["ch_clu"] = int(p.split("=", 1)[1])
            except ValueError:
                d["ch_clu"] = None
    return d if "engine" in d else None


def load_engine_results(path: str | None) -> dict:
    """Return {filename: {rc, pg_em, ...}} for one engine result file."""
    out = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        print(f"warning: engine result file not found: {path}", file=sys.stderr)
        return out
    with p.open() as f:
        for line in f:
            parsed = parse_result_line(line)
            if parsed and "file" in parsed:
                out[parsed["file"]] = parsed
    return out


def build_engine_block(expected_entry: dict, eng: dict | None) -> dict | None:
    """Build the per-engine block, computing pass against the expected count."""
    if eng is None:
        return None
    expected_count = expected_entry["expected_emitters"]
    pg_em = eng.get("pg_em")
    rc = eng.get("rc")
    block = {
        "rc": rc,
        "t": eng.get("t"),
        "pg_em": pg_em,
        "pg_sig": eng.get("pg_sig"),
        "pg_clu": eng.get("pg_clu"),
        "ch_pdw": eng.get("ch_pdw"),
        "ch_clu": eng.get("ch_clu"),
        "expected_emitters": expected_count,
    }
    if rc is not None and pg_em is not None:
        block["pass"] = (rc == 0) and (pg_em == expected_count)
    else:
        block["pass"] = False
    return block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="path to the ai_elint_tests repo root")
    ap.add_argument("--classic", default=None, help="classic batch result file (optional)")
    ap.add_argument("--onnx", default=None, help="onnx batch result file (optional)")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    repo = Path(args.repo)
    expected = load_param_map(repo)
    classic = load_engine_results(args.classic)
    onnx = load_engine_results(args.onnx)

    results = []
    for fn in sorted(expected):
        entry = expected[fn]
        e = {
            "file": fn,
            "type": entry["type"],
            "emitters": entry["emitters"],
            "emitter_ids": entry["emitter_ids"],
            "expected_emitters": entry["expected_emitters"],
            "std_no_emitters": entry["std_no_emitters"],
        }
        if entry["type"] == "unknown":
            e["spec_ranges"] = {name: SPEC_RANGES[name] for name in entry["emitters"] if name in SPEC_RANGES}
        c = build_engine_block(entry, classic.get(fn))
        o = build_engine_block(entry, onnx.get(fn))
        if c:
            e["classic"] = c
        if o:
            e["onnx"] = o
        results.append(e)

    def _stat(results, key):
        have = [r[key] for r in results if key in r]
        return {"pass": sum(1 for r in have if r["pass"]),
                "fail": sum(1 for r in have if not r["pass"]),
                "total": len(have)}

    output = {
        "source": {
            "known_param_csv": KNOWN_PARAM,
            "unknown_param_csv": UNKNOWN_PARAM,
            "note": ("Expected emitter counts/names come from the parameter CSVs the "
                     "pytest harness itself reads. pass = engine ran (rc==0) and the "
                     "detected emitter count equals expected_emitters."),
        },
        "total_files": len(results),
        "known": {"files": sum(1 for r in results if r["type"] == "known")},
        "unknown": {"files": sum(1 for r in results if r["type"] == "unknown")},
        "classic": _stat(results, "classic"),
        "onnx": _stat(results, "onnx"),
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Total files:   {len(results)} (known {output['known']['files']}, unknown {output['unknown']['files']})")
    print(f"Classic: {output['classic']['pass']} pass / {output['classic']['fail']} fail / {output['classic']['total']} measured")
    print(f"ONNX:    {output['onnx']['pass']} pass / {output['onnx']['fail']} fail / {output['onnx']['total']} measured")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
