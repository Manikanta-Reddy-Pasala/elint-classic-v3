"""ELINT runner UI — small web front end for the ai_elint engines.

Pick a source (one CSV, a folder of CSVs, or the pulses already in
ClickHouse), pick engines (classic, onnx, or both), run, and see per run:
emitters, signals, clusters, pulses, expected result (when the file is in
the test repo's expected_results.json) and time per layer.

Runs are executed one at a time (they share the databases) by a worker
thread that calls the `ai_processor` binary:
  file/folder:  ai_processor --csv <file>        (load into CH, then process)
  load only:    ai_processor --csv <file> --load-only
  database:     ai_processor --from-db           (process what is in signals.pdw)

Standard library + psycopg2 only. Settings via environment (see README).
"""
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BIN = os.environ.get("AI_PROCESSOR", os.path.join(REPO, "target", "release", "ai_processor"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO, "data"))
EXPECTED = os.environ.get("EXPECTED_JSON", os.path.join(HERE, "expected_results.json"))
PG_URL = os.environ.get("PG_URL", "postgresql://elint:elint@127.0.0.1:15432/sigint")
CH_URL = os.environ.get("CH_URL", "http://127.0.0.1:18123")
ONNX_URL = os.environ.get("ONNX_URL", "http://127.0.0.1:8000")
PORT = int(os.environ.get("UI_PORT", "8800"))
RESULTS_LOG = os.environ.get("RESULTS_LOG", os.path.join(HERE, "results.jsonl"))

try:
    EXPECTED_MAP = json.load(open(EXPECTED))
except Exception:
    EXPECTED_MAP = {}

JOBS = {}            # job id -> job dict
QUEUE = queue.Queue()
LOCK = threading.Lock()


def engine_env(engine):
    env = dict(os.environ)
    env.update({"PG_URL": PG_URL, "CH_URL": CH_URL, "CLICKHOUSE_DB": "signals",
                "CLICKHOUSE_USER": env.get("CLICKHOUSE_USER", "elint"),
                "CLICKHOUSE_PASSWORD": env.get("CLICKHOUSE_PASSWORD", "elint"),
                "ONNX_URL": ONNX_URL, "ELINT_LOG_DIR": os.path.join(HERE, "logs")})
    if engine == "onnx":
        env["CLASSIC"] = "0"
        env.pop("EMBED", None)
    else:
        env["CLASSIC"] = "1"
        env["EMBED"] = "classic"
    return env


def layers(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("LAYER|"):
            _, k, v = line.strip().split("|", 2)
            try:
                out[k] = round(float(v), 2)
            except ValueError:
                pass
    return out


def snapshot(pg_url=PG_URL):
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()

    def q(sql):
        cur.execute(sql)
        return cur.fetchall()
    snap = {
        "emitters": [r[0] for r in q("select name from emitters order by id")],
        "signals": q("select count(*) from signals")[0][0],
        "clusters": q("select count(*) from clusters")[0][0],
        "pulses_in_clusters": int(q("select coalesce(sum(pdw_count),0) from clusters")[0][0]),
        "per_emitter": [
            {"id": r[0], "name": r[1], "clusters": r[2], "pulses": int(r[3] or 0),
             "pri_us": r[4] and round(r[4], 2), "pw_us": r[5] and round(r[5], 3),
             "freq_mhz": r[6] and round(float(r[6]) / 1e6, 3), "doa_deg": r[7] and round(r[7], 1)}
            for r in q("""select e.id, e.name, count(c.id), sum(c.pdw_count),
                          avg(c.pulse_repetition_interval_us), avg(c.pulse_width_us), avg(c.frequency_hz), max(s.doa_deg)
                          from emitters e left join signals s on s.emitter_id = e.id
                          left join clusters c on c.signal_id = s.id group by e.id, e.name order by e.id""")
        ],
    }
    conn.close()
    return snap



def _run_one(job, item):
    engine, path, action = item["engine"], item.get("path"), item["action"]
    args = [BIN]
    if action == "db":
        args.append("--from-db")
    else:
        args += ["--csv", path]
        if action == "load":
            args.append("--load-only")
    t0 = time.perf_counter()
    p = subprocess.run(args, env=engine_env(engine), capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    lay = layers(p.stdout)
    res = {"file": os.path.basename(path) if path else "(database)", "path": path, "engine": engine,
           "action": action, "rc": p.returncode, "wall_s": round(wall, 2), "layers": lay,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if "engine_total" in lay:
        res["pipeline_ms"] = round(lay["engine_total"] - lay.get("wipe_tables", 0) - lay.get("load_csv", 0), 1)
    if p.returncode != 0:
        res["error"] = (p.stderr or p.stdout)[-1500:]
    if action != "load" and p.returncode == 0:
        res.update(snapshot())
        e = EXPECTED_MAP.get(res["file"])
        if e:
            res["expected"] = {"emitters": e["exp_em"], "names": e.get("expected_names"), "ids": e.get("expected_ids")}
            res["count_ok"] = len(res["emitters"]) == e["exp_em"]
            if e["kind"] == "known":
                res["names_ok"] = all(i in res["emitters"] for i in e["expected_ids"])
    return res


# ---- hardware monitor: CPU / GPU / memory, sampled every 0.1 s -------------------------
GPU_DEV = os.environ.get("GPU_SYSFS", "/sys/class/drm/card0/device")


def _rd(path, default=None):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def _cpu_name():
    for l in (_rd("/proc/cpuinfo", "") or "").splitlines():
        if l.startswith("model name"):
            return l.split(":", 1)[1].strip()
    return "CPU"


_CPU_FULL = _cpu_name()
HW_STATIC = {"cpu": _CPU_FULL.split(" w/ ")[0], "threads": os.cpu_count(),
             "gpu": ("AMD " + _CPU_FULL.split(" w/ ")[1]) if " w/ " in _CPU_FULL else "GPU"}
HW = {"cpu_pct": 0.0, "busy_threads": 0, "gpu_pct": None}
_TRACK = {"on": False, "n": 0, "cpu_sum": 0.0, "gpu_sum": 0.0, "cpu_peak": 0.0, "gpu_peak": 0, "thr_peak": 0}


def _cpu_times():
    out = []
    for l in (_rd("/proc/stat", "") or "").splitlines():
        if l.startswith("cpu"):
            v = [int(x) for x in l.split()[1:]]
            out.append((v[3] + v[4], sum(v)))   # idle+iowait, total
    return out


def hw_sampler():
    prev = _cpu_times()
    while True:
        time.sleep(0.1)
        cur = _cpu_times()
        pct = []
        for (i0, t0), (i1, t1) in zip(prev, cur):
            dt = t1 - t0
            pct.append(100.0 * (1 - (i1 - i0) / dt) if dt > 0 else 0.0)
        prev = cur
        if not pct:
            continue
        g = _rd(GPU_DEV + "/gpu_busy_percent")
        HW["cpu_pct"] = round(pct[0], 1)
        HW["busy_threads"] = sum(1 for x in pct[1:] if x > 50)
        HW["gpu_pct"] = int(g) if g and g.isdigit() else None
        if _TRACK["on"]:
            _TRACK["n"] += 1
            _TRACK["cpu_sum"] += pct[0]
            _TRACK["gpu_sum"] += HW["gpu_pct"] or 0
            _TRACK["cpu_peak"] = max(_TRACK["cpu_peak"], pct[0])
            _TRACK["gpu_peak"] = max(_TRACK["gpu_peak"], HW["gpu_pct"] or 0)
            _TRACK["thr_peak"] = max(_TRACK["thr_peak"], HW["busy_threads"])


def hw_now():
    mem = {}
    for l in (_rd("/proc/meminfo", "") or "").splitlines():
        k, v = l.split(":", 1)
        mem[k] = int(v.split()[0]) * 1024

    def gb(x):
        return round(x / 2**30, 1) if x is not None else None

    def num(name):
        v = _rd(GPU_DEV + "/" + name)
        return int(v) if v and v.isdigit() else None
    try:
        import urllib.request
        workers = json.load(urllib.request.urlopen(ONNX_URL + "/health", timeout=1)).get("workers", [])
    except Exception:
        workers = []
    return dict(HW_STATIC, **HW,
                ram_used_gb=gb(mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)), ram_total_gb=gb(mem.get("MemTotal")),
                vram_used_gb=gb(num("mem_info_vram_used")), vram_total_gb=gb(num("mem_info_vram_total")),
                gtt_used_gb=gb(num("mem_info_gtt_used")),
                kfd=os.path.exists("/dev/kfd"), workers=workers)


# ---- compare demo ---------------------------------------------------------------------
KNOWN_DIR = os.path.join(DATA_DIR, "pdw_generator", "known_emitters") + "/"
PRESETS = [
    {"id": 1, "title": "Compare 1 · big file", "file": KNOWN_DIR + "mf03_psr2_best-3_plus3.csv",
     "engines": ["onnx", "classic"],
     "story": "185k pulses, 2 known radars (MF03, PSR2) + 3 unknown radars. Rust ONNX vs Rust classic."},
    {"id": 2, "title": "Compare 2 · medium file", "file": KNOWN_DIR + "mf03_mf04_best-10_plus2.csv",
     "engines": ["onnx", "classic"],
     "story": "24k pulses, 2 known radars (MF03, MF04) + 2 unknown. Rust ONNX vs Rust classic."},
    {"id": 3, "title": "Compare 3 · easy file", "file": KNOWN_DIR + "mf03_mf04_best-0_plus0.csv",
     "engines": ["onnx", "classic"],
     "story": "105k pulses, 2 known radars only. Both engines are right; only the time differs."},
]
PLOT_N = int(os.environ.get("PLOT_N", "6000"))
_SAMPLE_CACHE = {}


def csv_sample(path):
    """Every k-th pulse of the CSV: row number, CSV id, time s, freq MHz, PW us, DOA deg, true emitter."""
    import csv
    st = os.stat(path)
    key = (path, st.st_mtime)
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]
    with open(path, newline="") as fh:
        n = sum(1 for _ in fh) - 1
    step = max(1, n // PLOT_N)
    rows = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=1):
            if (i - 1) % step:
                continue
            def f(k, scale=1.0):
                try:
                    return round(float(r.get(k) or "nan") / scale, 4)
                except ValueError:
                    return None
            rows.append([i, int(float(r.get("id") or i)), f("time_of_arrival_ns", 1e9), f("center_frequency_hz", 1e6),
                         f("duration_ns", 1e3), f("direction_of_arrival_deg"), r.get("emitter_id") or "?"])
    e = EXPECTED_MAP.get(os.path.basename(path)) or {}
    out = {"path": path, "file": os.path.basename(path), "pulses": n, "step": step, "rows": rows,
           "expected": {"emitters": e.get("exp_em"), "names": e.get("expected_names"), "ids": e.get("expected_ids")}}
    _SAMPLE_CACHE.clear()
    _SAMPLE_CACHE[key] = out
    return out


def _ch(sql):
    import urllib.request
    req = urllib.request.Request(CH_URL + "/", data=sql.encode())
    return urllib.request.urlopen(req, timeout=30).read().decode()


def plot_labels(engine, path):
    """Label the sampled pulses with the emitter the engine put them in (read right after the run)."""
    smp = csv_sample(path)
    keys = [r[0] for r in smp["rows"]]
    ch_db, pg = "signals", PG_URL
    pulse_cluster = {}
    uniq = sorted(set(keys))
    for i in range(0, len(uniq), 3000):
        part = ",".join(str(k) for k in uniq[i:i + 3000])
        txt = _ch(f"select toUInt64(a) as k, any(cluster_id) from {ch_db}.clusters array join activity_ids as a "
                  f"where k in ({part}) group by k format TabSeparated")
        for line in txt.splitlines():
            k, c = line.split("\t")
            pulse_cluster[int(k)] = int(c)
    conn = psycopg2.connect(pg)
    cur = conn.cursor()
    cur.execute("select c.id, e.id, e.name from clusters c join signals s on s.id = c.signal_id "
                "join emitters e on e.id = s.emitter_id")
    cl_em = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    conn.close()
    order, legend, labels = {}, [], []
    for k in keys:
        em = cl_em.get(pulse_cluster.get(k))
        if em is None:
            labels.append(-1)
            continue
        if em[0] not in order:
            order[em[0]] = len(legend)
            legend.append({"emitter_id": em[0], "name": em[1], "n": 0})
        legend[order[em[0]]]["n"] += 1
        labels.append(order[em[0]])
    return {"labels": labels, "legend": legend, "unassigned": labels.count(-1)}


def run_one(job, item):
    """Run one item while the sampler records CPU/GPU load; log the result with its load."""
    _TRACK.update(on=True, n=0, cpu_sum=0.0, gpu_sum=0.0, cpu_peak=0.0, gpu_peak=0, thr_peak=0)
    try:
        res = _run_one(job, item)
    finally:
        _TRACK["on"] = False
    n = max(_TRACK["n"], 1)
    res["hw"] = {"cpu_avg": round(_TRACK["cpu_sum"] / n, 1), "cpu_peak": round(_TRACK["cpu_peak"], 1),
                 "gpu_avg": round(_TRACK["gpu_sum"] / n, 1), "gpu_peak": _TRACK["gpu_peak"],
                 "threads_peak": _TRACK["thr_peak"], "samples": _TRACK["n"]}
    if res.get("rc") == 0 and item.get("action") == "run" and item.get("path"):
        try:
            pid = uuid.uuid4().hex[:10]
            os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
            with open(os.path.join(HERE, "logs", f"plot_{pid}.json"), "w") as fh:
                json.dump(plot_labels(item["engine"], item["path"]), fh)
            res["plot_id"] = pid
        except Exception as e:
            res["plot_error"] = repr(e)
    with open(RESULTS_LOG, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


def worker():
    while True:
        jid = QUEUE.get()
        job = JOBS[jid]
        job["status"] = "running"
        for k, item in enumerate(job["items"]):
            job["current"] = f"{item['engine']} · {os.path.basename(item.get('path') or 'database')}"
            try:
                job["results"].append(run_one(job, item))
            except Exception as e:
                job["results"].append({"file": item.get("path"), "engine": item["engine"], "rc": -1, "error": repr(e)})
            job["done"] = k + 1
        job["status"] = "finished"
        job["current"] = ""


def list_csvs(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(".csv")]
    return sorted(out)


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        if u.path == "/api/config":
            try:
                import urllib.request
                health = json.load(urllib.request.urlopen(ONNX_URL + "/health", timeout=2))
            except Exception as e:
                health = {"status": "down", "error": str(e)}
            return self._json({"bin": BIN, "data_dir": DATA_DIR, "onnx": health, "expected_files": len(EXPECTED_MAP)})
        if u.path == "/api/presets":
            return self._json([dict(p, exists=os.path.exists(p["file"])) for p in PRESETS])
        if u.path == "/api/sample":
            p = qs.get("path", [""])[0]
            if not os.path.isfile(p):
                return self._json({"error": f"not a file: {p}"}, 400)
            return self._json(csv_sample(p))
        if u.path.startswith("/api/plot/"):
            pid = u.path.rsplit("/", 1)[1]
            f = os.path.join(HERE, "logs", f"plot_{pid}.json")
            if not pid.isalnum() or not os.path.exists(f):
                return self._json({"error": "no such plot"}, 404)
            return self._json(json.load(open(f)))
        if u.path == "/api/hw":
            return self._json(hw_now())
        if u.path == "/api/browse":
            p = qs.get("path", [DATA_DIR])[0]
            if not os.path.isdir(p):
                return self._json({"error": f"not a folder: {p}"}, 400)
            dirs = sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))
            files = sorted(f for f in os.listdir(p) if f.lower().endswith(".csv"))
            return self._json({"path": os.path.abspath(p), "parent": os.path.dirname(os.path.abspath(p)),
                               "dirs": dirs, "files": [{"name": f, "mb": round(os.path.getsize(os.path.join(p, f)) / 1e6, 1)} for f in files]})
        if u.path.startswith("/api/jobs/"):
            j = JOBS.get(u.path.rsplit("/", 1)[1])
            return self._json(j) if j else self._json({"error": "no such job"}, 404)
        if u.path == "/api/history":
            rows = []
            if os.path.exists(RESULTS_LOG):
                rows = [json.loads(l) for l in open(RESULTS_LOG).read().splitlines()[-200:]]
            return self._json(rows[::-1])
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._json({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        engines = [e for e in body.get("engines", []) if e in ("classic", "onnx")]
        action = body.get("action", "run")          # run | load | db
        if action == "db":
            items = [{"engine": e, "action": "db"} for e in engines]
        else:
            path = body.get("path", "")
            if not os.path.exists(path):
                return self._json({"error": f"path not found: {path}"}, 400)
            files = list_csvs(path)[: int(body.get("limit") or 100000)]
            if action == "load":
                items = [{"engine": "classic", "path": f, "action": "load"} for f in files]
            else:
                items = [{"engine": e, "path": f, "action": "run"} for f in files for e in engines]
        if not items:
            return self._json({"error": "nothing to run (choose an engine and a source)"}, 400)
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"id": jid, "status": "queued", "items": items, "done": 0, "results": [], "current": "",
                     "created": time.strftime("%H:%M:%S")}
        QUEUE.put(jid)
        self._json({"job": jid, "items": len(items)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=hw_sampler, daemon=True).start()
    print(f"ELINT UI on http://0.0.0.0:{PORT}  (binary {BIN})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
