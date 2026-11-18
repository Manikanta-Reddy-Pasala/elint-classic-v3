#!/usr/bin/env python3
"""Build expected_results.json from the param CSVs + measured engine results.

Writes TWO files:
  tests/test_data/expected_results.json
      Flat dict: {filename: {kind, expected_names, expected_ids, exp_em, std}}
      Used by test_and_validate_all.sh to validate pg_em after each run.

  scripts/608file_test/expected_results_full.json
      Rich report: per-file expected + measured classic/onnx results + pass flag.
      Used for review / comparison.

Usage (on the VM or locally with result files available):
  python3 build_expected_results.py \
      --repo /path/to/ai_elint_tests \
      [--classic /path/to/classic_results.txt] \
      [--onnx    /path/to/onnx_results.txt]

When --classic / --onnx are omitted, only the expected values are written
(no measured results, no pass/fail per engine).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

KNOWN_PARAM = "tests/test_data/known_emitters_db_emitter_level.csv"
UNKNOWN_PARAM = "tests/test_data/unknown_emitters_db.csv"

SPEC_RANGES = {
    "CTA1": {"frequency": [3000000000, 3175000000], "pri": [129, 253], "pw": [21.5, 22.5]},
    "CTA2": {"frequency": [2930000000, 3110000000], "pri": [1875, 2271], "pw": [6.0, 7.0]},
    "NAV1": {"frequency": [2950000000, 3051000000], "pri": [101, 176], "pw": [0.51, 0.61]},
    "NAV2": {"frequency": [3045000000, 3055000000], "pri": [217, 221], "pw": [0.65, 0.75]},
    "NAV3": {"frequency": [3000000000, 3100000000], "pri": [168, 208], "pw": [22, 23]},
}


def load_param_map(repo: Path) -> dict:
    """Return {filename: {kind, expected_names, expected_ids, exp_em, std}}."""
    m = {}
    with (repo / KNOWN_PARAM).open() as f:
        for row in csv.DictReader(f):
            fn = row["csv_file_path"].split("/")[-1]
            m[fn] = {
                "kind": "known",
                "expected_names": [row["emitter_1_name"], row["emitter_2_name"]],
                "expected_ids": [row["emitter_id_1"], row["emitter_id_2"]],
                "exp_em": int(row["no_emitters"]),
                "std": int(row["std_no_emitters"]),
            }
    with (repo / UNKNOWN_PARAM).open() as f:
        for row in csv.DictReader(f):
            fn = row["csv_file_path"].split("/")[-1]
            m[fn] = {
                "kind": "unknown",
                "expected_names": [row["emitter_1_name"], row["emitter_2_name"]],
                "expected_ids": None,
                "exp_em": int(row["number_emitters"]),
                "std": int(row["std_no_emitters"]),
            }
    return m


def parse_result_line(line: str):
    line = line.strip()
    if not line.startswith("RESULT|"):
        return None
    parts = line.split("|")
    if len(parts) < 3:
        return None
    d = {"file": parts[1]}
    for p in parts:
        if p.startswith("engine="):
            d["engine"] = p.split("=", 1)[1]
        elif p.startswith("rc="):
            try: d["rc"] = int(p.split("=", 1)[1])
            except ValueError: d["rc"] = None
        elif p.startswith("t="):
            d["t"] = p.split("=", 1)[1]
        elif p.startswith("pg_em="):
            try: d["pg_em"] = int(p.split("=", 1)[1])
            except ValueError: d["pg_em"] = None
        elif p.startswith("pg_sig="):
            try: d["pg_sig"] = int(p.split("=", 1)[1])
            except ValueError: d["pg_sig"] = None
        elif p.startswith("pg_clu="):
            try: d["pg_clu"] = int(p.split("=", 1)[1])
            except ValueError: d["pg_clu"] = None
        elif p.startswith("ch_pdw="):
            try: d["ch_pdw"] = int(p.split("=", 1)[1])
            except ValueError: d["ch_pdw"] = None
        elif p.startswith("ch_clu="):
            try: d["ch_clu"] = int(p.split("=", 1)[1])
            except ValueError: d["ch_clu"] = None
    return d if "engine" in d else None


def load_engine(path):
    out = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        p = parse_result_line(line)
        if p:
            out[p["file"]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--classic", default=None)
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()

    repo = Path(args.repo)
    expected = load_param_map(repo)
    classic = load_engine(args.classic)
    onnx = load_engine(args.onnx)

    # --- Flat file: used by test_and_validate_all.sh ---
    flat_path = repo / "tests" / "test_data" / "expected_results.json"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    with flat_path.open("w") as f:
        json.dump(expected, f, indent=2)

    # --- Rich report: for review / comparison ---
    rich = {
        "source": {
            "known_param_csv": KNOWN_PARAM,
            "unknown_param_csv": UNKNOWN_PARAM,
            "note": (
                "exp_em comes from the param CSVs the pytest harness reads. "
                "pass = rc==0 AND pg_em == exp_em."
            ),
        },
        "total_files": len(expected),
        "known_files": sum(1 for v in expected.values() if v["kind"] == "known"),
        "unknown_files": sum(1 for v in expected.values() if v["kind"] == "unknown"),
        "results": [],
    }
    c_pass = c_fail = o_pass = o_fail = 0
    for fn in sorted(expected):
        e = expected[fn]
        entry = {
            "file": fn,
            "kind": e["kind"],
            "expected_names": e["expected_names"],
            "expected_ids": e["expected_ids"],
            "exp_em": e["exp_em"],
            "std": e["std"],
        }
        if e["kind"] == "unknown":
            entry["spec_ranges"] = {
                n: SPEC_RANGES[n] for n in e["expected_names"] if n in SPEC_RANGES
            }
        for key, eng in (("classic", classic), ("onnx", onnx)):
            if fn in eng:
                g = eng[fn]
                block = {
                    "rc": g.get("rc"), "t": g.get("t"),
                    "pg_em": g.get("pg_em"), "pg_sig": g.get("pg_sig"),
                    "pg_clu": g.get("pg_clu"), "ch_pdw": g.get("ch_pdw"),
                    "ch_clu": g.get("ch_clu"), "exp_em": e["exp_em"],
                    "pass": (g.get("rc") == 0 and g.get("pg_em") == e["exp_em"]),
                }
                entry[key] = block
                if key == "classic":
                    if block["pass"]: c_pass += 1
                    else: c_fail += 1
                else:
                    if block["pass"]: o_pass += 1
                    else: o_fail += 1
        rich["results"].append(entry)

    rich["classic"] = {"pass": c_pass, "fail": c_fail, "total": c_pass + c_fail}
    rich["onnx"] = {"pass": o_pass, "fail": o_fail, "total": o_pass + o_fail}

    rich_path = repo / "scripts" / "608file_test" / "expected_results_full.json"
    rich_path.parent.mkdir(parents=True, exist_ok=True)
    with rich_path.open("w") as f:
        json.dump(rich, f, indent=2)

    print(f"Flat  (test_and_validate_all.sh): {flat_path}  [{len(expected)} files]")
    print(f"Rich  (review report):            {rich_path}")
    print(f"  classic: {c_pass} pass / {c_fail} fail / {c_pass+c_fail} measured")
    print(f"  onnx:    {o_pass} pass / {o_fail} fail / {o_pass+o_fail} measured")


if __name__ == "__main__":
    main()
