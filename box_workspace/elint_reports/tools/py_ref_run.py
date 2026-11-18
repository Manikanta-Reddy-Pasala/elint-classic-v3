import os, sys, time, logging
import numpy as np
import pandas as pd

os.environ["POSTGRES_DB_HOST"] = "127.0.0.1"
os.environ["POSTGRES_DB_PORT"] = "15432"
os.environ["POSTGRES_DB_NAME"] = "sigint"
os.environ["POSTGRES_DB_USER"] = "elint"
os.environ["POSTGRES_DB_PASS"] = "elint"
os.environ["CLICKHOUSE_DB_HOST"] = "127.0.0.1"
os.environ["CLICKHOUSE_DB_PORT"] = "18123"
os.environ["CLICKHOUSE_DB_NAME"] = "signals"
os.environ["CLICKHOUSE_DB_USER"] = "elint"
os.environ["CLICKHOUSE_DB_PASS"] = "elint"
os.environ["CLICKHOUSE_SECURE"] = "0"
os.environ["CLICKHOUSE_VERIFY_TLS"] = "0"
os.environ["CLICKHOUSE_DB_TABLE"] = "signals.pdw"
os.environ["MODEL_PATH"] = "model/transformer_model_known_v16.onnx"

APP = "/home/sysadm/coderepo/ai_elint_python_master/integration/app"
CSV = sys.argv[1]
sys.path.insert(0, APP)
os.chdir(APP)
import config
import logger  # patches logging.Logger.verbose for master
from database import connect_clickhouse, connect_postgres
from cluster_handling import (handle_embedding, load_tracking_cache,
    load_cluster_centers_cache, load_medoids_cache,
    CACHE_FILE_PATH, CENTERS_FILE_PATH, MEDOIDS_FILE_PATH)
from inference import init_onnx_session, run_onnx_inference
from pathlib import Path

def wipe(pg, ch):
    with pg.cursor() as cur:
        for t in ("clusters", "signals", "emitters"):
            try: cur.execute(f"TRUNCATE {t} CASCADE")
            except Exception: cur.execute(f"DELETE FROM {t}")
    pg.commit()
    try: ch.command("TRUNCATE TABLE clusters")
    except Exception as e: print("ch truncate skipped:", e)

def count(pg, t):
    with pg.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {t}")
        return cur.fetchone()[0]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
t0 = time.perf_counter()
onnx_session = init_onnx_session(config.ONNX_MODEL_PATH)
ch = connect_clickhouse(); pg = connect_postgres()
for p in (CACHE_FILE_PATH, CENTERS_FILE_PATH, MEDOIDS_FILE_PATH):
    for cand in (Path(APP)/p, Path(p), Path("/home/sysadm/coderepo/ai_elint_python_master/integration")/p):
        try:
            if cand.exists(): cand.unlink()
        except Exception: pass
wipe(pg, ch)
load_tracking_cache(); load_cluster_centers_cache(); load_medoids_cache()
df = pd.read_csv(CSV)
df["id"] = df["id"].astype(str)
if "emitter_id" not in df.columns: df["emitter_id"] = 0
if "modulation" not in df.columns: df["modulation"] = 0
df["direction_of_arrival_deg"] = pd.to_numeric(df["direction_of_arrival_deg"], errors="coerce")
df["power_dbm"] = pd.to_numeric(df["power_dbm"], errors="coerce")
sel = df[["id","power_dbm","center_frequency_hz","time_of_arrival_ns","duration_ns",
          "bandwidth_hz","snr_db","direction_of_arrival_deg","modulation","emitter_id"]
         ].copy().reset_index(drop=True)
sel["duration_us"] = sel["duration_ns"] / 1000.0
sel["TOA"] = pd.to_datetime(sel["time_of_arrival_ns"]).dt.tz_localize(None).astype("int64").astype(np.float64).to_numpy()
power_raw = sel["power_dbm"].values.astype(np.float32)
freq_raw  = sel["center_frequency_hz"].values.astype(np.float32)
pw_raw    = sel["duration_us"].values.astype(np.float32)
dt = pd.to_datetime(sel["time_of_arrival_ns"]).dt.tz_localize(None)
toa_raw = dt.astype("int64").astype(np.float64).to_numpy()
n = len(sel) // config.PDW_CHUNK_SIZE * config.PDW_CHUNK_SIZE
sel = sel.iloc[:n].reset_index(drop=True)
power_raw, freq_raw, pw_raw, toa_raw = power_raw[:n], freq_raw[:n], pw_raw[:n], toa_raw[:n]
print(f"pulses={n} chunks={n//config.PDW_CHUNK_SIZE}")
for i in range(n // config.PDW_CHUNK_SIZE):
    s = i * config.PDW_CHUNK_SIZE; e = s + config.PDW_CHUNK_SIZE
    out = run_onnx_inference(onnx_session,
        freq_raw[s:e]/10e5, power_raw[s:e], pw_raw[s:e], toa_raw[s:e]/1000.0)
    handle_embedding(out, sel.iloc[s:e].copy().reset_index(drop=True), pg, ch)
n_sig = count(pg, "signals"); n_em = count(pg, "emitters"); n_cl = count(pg, "clusters")
print("="*50)
print(f"MASTER E2E done in {time.perf_counter()-t0:.1f}s")
print(f"signals={n_sig} emitters={n_em} clusters_pg={n_cl}")
print("="*50)
with pg.cursor() as cur:
    cur.execute("select name, count(*) from emitters group by name order by name")
    print("names", cur.fetchall())
