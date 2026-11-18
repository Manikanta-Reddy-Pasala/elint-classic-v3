#!/usr/bin/env python3
"""Drop-in for the ai_processor binary that also records a report.

The test harness (ai_elint_tests/tests/conftest_rust.py) runs whatever
RUST_AI_PROCESSOR points at. Point it here: this runs the real binary
(REAL_AI_PROCESSOR) with the same arguments and environment, passes its
output and exit code through, and — when RUST_REPORT_DIR is set — saves
  <file>.<engine>.log   engine output (incl. LAYER|<stage>|<ms> lines)
  <file>.<engine>.json  result-table summary right after the run
so nothing in the test repo has to change for reporting.
"""
import json, os, subprocess, sys, time

real = os.environ.get("REAL_AI_PROCESSOR", os.path.expanduser("~/coderepo/ai_elint/target/release/ai_processor"))
t0 = time.perf_counter()
p = subprocess.run([real] + sys.argv[1:], capture_output=True, text=True)
wall = time.perf_counter() - t0
sys.stdout.write(p.stdout)
sys.stderr.write(p.stderr)

rd = os.environ.get("RUST_REPORT_DIR")
if rd and "--csv" in sys.argv:
    name = os.path.basename(sys.argv[sys.argv.index("--csv") + 1])
    engine = "classic" if os.environ.get("CLASSIC", "0") in ("1", "true") else "onnx"
    os.makedirs(rd, exist_ok=True)
    base = os.path.join(rd, f"{os.path.splitext(name)[0]}.{engine}")
    open(base + ".log", "w").write(p.stdout + p.stderr)
    layers = {}
    for line in p.stdout.splitlines():
        if line.startswith("LAYER|"):
            _, k, v = line.strip().split("|", 2)
            try:
                layers[k] = float(v)
            except ValueError:
                pass
    s = {"file": name, "engine": engine, "rc": p.returncode, "process_wall_s": round(wall, 3), "layers": layers}
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ["PG_URL"])
        cur = conn.cursor()
        def q(sql):
            cur.execute(sql)
            return cur.fetchall()
        s["emitters"] = [r[0] for r in q("select name from emitters order by id")]
        s["signals"] = q("select count(*) from signals")[0][0]
        s["clusters"] = q("select count(*) from clusters")[0][0]
        s["clusters_without_signal"] = q("select count(*) from clusters where signal_id is null")[0][0]
        s["pulses_in_clusters"] = int(q("select coalesce(sum(pdw_count),0) from clusters")[0][0])
        s["per_emitter"] = [
            {"emitter_id": r[0], "name": r[1], "clusters": r[2], "pulses": int(r[3] or 0),
             "pri_us": r[4], "pw_us": r[5], "freq_hz": float(r[6]) if r[6] is not None else None}
            for r in q("""select e.id, e.name, count(c.id), sum(c.pdw_count),
                                 avg(c.pulse_repetition_interval_us), avg(c.pulse_width_us), avg(c.frequency_hz)
                          from emitters e left join signals s on s.emitter_id = e.id
                          left join clusters c on c.signal_id = s.id
                          group by e.id, e.name order by e.id""")]
        conn.close()
    except Exception as e:
        s["db_error"] = repr(e)
    json.dump(s, open(base + ".json", "w"), indent=1, default=str)
sys.exit(p.returncode)
