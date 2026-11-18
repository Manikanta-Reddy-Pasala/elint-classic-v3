"""Run the fresh (holdout) files through several engine setups and score them.

usage: run_holdout.py <holdout_dir> <out.jsonl> [setup ...]
setups: classic_old, classic_new, onnx, onnx_merge
Score per file: exact radar count vs truth, all known library ids found.
"""
import json
import os
import subprocess
import sys
import time

import psycopg2

BIN = os.path.expanduser("~/coderepo/ai_elint/target/release/ai_processor")
PG_URL = "postgresql://elint:elint@127.0.0.1:15432/sigint"
BASE = {"PG_URL": PG_URL, "CH_URL": "http://127.0.0.1:18123", "CLICKHOUSE_DB": "signals",
        "CLICKHOUSE_USER": "elint", "CLICKHOUSE_PASSWORD": "elint", "ONNX_URL": "http://127.0.0.1:8000",
        "ELINT_LOG_DIR": "/tmp/elint_holdout_logs"}
SETUPS = {
    # classic exactly as before 2026-11-17 (both clock-test fixes off)
    "classic_old": {"CLASSIC": "1", "EMBED": "classic", "CLASSIC_COH_MIN_PAIRS": "0", "CLASSIC_REPAIR_CLOCK_VETO": "0"},
    "classic_new": {"CLASSIC": "1", "EMBED": "classic"},
    "onnx": {"CLASSIC": "0"},
    "onnx_merge": {"CLASSIC": "0", "ONNX_CLOCK_MERGE": "1"},
}

hold, out = sys.argv[1], sys.argv[2]
setups = sys.argv[3:] or list(SETUPS)
truth = json.load(open(os.path.join(hold, "truth.json")))
os.makedirs(BASE["ELINT_LOG_DIR"], exist_ok=True)


def layers(stdout):
    lay = {}
    for line in stdout.splitlines():
        if line.startswith("LAYER|"):
            _, k, v = line.strip().split("|", 2)
            try:
                lay[k] = float(v)
            except ValueError:
                pass
    return lay


with open(out, "a") as fo:
    for setup in setups:
        for name, t in truth.items():
            env = dict(os.environ, **BASE, **SETUPS[setup])
            env.pop("EMBED", None) if setup.startswith("onnx") else None
            t0 = time.perf_counter()
            p = subprocess.run([BIN, "--csv", os.path.join(hold, name)], env=env, capture_output=True, text=True, timeout=3600)
            wall = time.perf_counter() - t0
            lay = layers(p.stdout)
            conn = psycopg2.connect(PG_URL)
            cur = conn.cursor()
            cur.execute("select name from emitters order by id")
            names = [r[0] for r in cur.fetchall()]
            conn.close()
            pipe = lay.get("engine_total", 0) - lay.get("wipe_tables", 0) - lay.get("load_csv", 0)
            rec = {"setup": setup, "file": name, "rc": p.returncode, "expected": t["emitters"], "found": len(names),
                   "count_ok": len(names) == t["emitters"], "known_ok": all(k in names for k in t["known_ids"]),
                   "names": names, "pipeline_ms": round(pipe, 1), "wall_s": round(wall, 2)}
            if p.returncode != 0:
                rec["error"] = (p.stderr or p.stdout)[-500:]
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(setup, name, rec["found"], "/", rec["expected"], "count_ok", rec["count_ok"], "known_ok", rec["known_ok"], flush=True)
