from typing import TypeVar
import os

# config.py
def cfg_var(var: str, fallback: str) -> str:
    return os.environ.get(var, fallback)


# === CLICKHOUSE DATABASE CONFIGURATION ===
CLICKHOUSE_CONFIG = {
    'host': 'db.ai.elint.local',
    'port': 8444,
    'database': 'signals',
    'user': 'default',
    'table': 'signals.pdw'
}
CLICKHOUSE_SECURE = bool(int(cfg_var('CLICKHOUSE_SECURE', '1')))
CLICKHOUSE_VERIFY_TLS = bool(int(cfg_var('CLICKHOUSE_VERIFY_TLS', '0')))

# === DATA FILTERING CRITERIA ===
FREQ_MIN_HZ = 0
FREQ_MAX_HZ = 9957191680
FETCH_LIMIT = 1024*20
POLLING_INTERVAL_SECONDS = 10
PDW_CHUNK_SIZE = 512



# === UPDATE THIS ONNX MODEL PATH ===
ONNX_MODEL_PATH = cfg_var('MODEL_PATH', 'model/transformer_model_known_v16.onnx')

EMBEDDING_COLS = [f"dim_{i}" for i in range(256)]


# config.py

# === SYSTEM TRACKING FILE ===
WATERMARK_FILE_PATH = "last_processed_timestamp.txt"

# === INITIALIZATION MODE STRATEGY ===
# Options:
# "WATERMARK" -> Reads text file tracking placeholder (Crash Safety)
# "CURRENT"   -> Dynamically binds to the current system clock time on boot
# "SPECIFIC"  -> Forces the pipeline to start exactly at the SPECIFIC_START_TIME below
INIT_MODE = "CURRENT"

# Manual fallback string utilized ONLY if INIT_MODE is set to "SPECIFIC"
SPECIFIC_START_TIME = "2026-09-17 09:29:33.849960"

#Similarity threshold for known emitters
SIMILARITY_THRESHOLD = 0.7
SIMILARITY_THRESHOLD_1117 = 0.5
MIN_CLUSTER_SIZE = 20
UNKNOWN_MERGED_MIN_CLUSTER_SIZE = 10
UNKNOWN_DOT_PRODUCT_THRESHOLD = 0.8
UNKNOWN_MERGED_MIN_FINAL_SCORE = 0.7
UNKNOWN_MERGED_MAX_ANGLE_SPREAD = 40
UNKNOWN_MERGED_CLUSTER_MIN_SCORE = 0.3
UNKNONW_MERGED_SNR_DELTA_FILTER_DB = 10
UNKNONW_MERGED_SIMILARITY_FILTER_THRESHOLD = 0.8
UNKNONW_BASE_MATCHED_EMITTER_ID_FOR_SPLIT = 4000

UNKNOWN_ACCUMULATION_TIME_SEC = 200
UNKNOWN_MIN_SIGNAL_SIZE = 20
KNOWN_MIN_SIGNAL_SIZE = 20

KNOWN_DOT_PRODUCT_THRESHOLD = 0.6
BUFFER_THRESHOLD = 512*1


# === POSTGRESQL (POSTGRES) CONFIGURATION ===

PG_CONFIG = {
    # 'host': 'localhost',
    'host': 'db.ai.elint.local',
    'port': 5432,
    'database': 'sigint',
    'user': 'postgres',
    'password': 'postgres'
}
PG_TABLE_CLUSTERS = cfg_var('POSTGRES_TABLE_CLUSTERS', 'clusters')
PG_TABLE_SIGNALS = cfg_var('POSTGRES_TABLE_SIGNALS', 'signals')
PG_TABLE_EMITTERS = cfg_var('POSTGRES_TABLE_EMITTERS', 'emitters')

UNKNOWN_CLUSTER_FILENAME = "unknown_clusters"

unknown_cluster_csv_desired_order = [
    'id','time_of_arrival_ns', 'power_dbm', 'center_frequency_hz', 'bandwidth_hz',
    'duration_ns', 'direction_of_arrival_deg', 'emitter_id',
    'snr_db', 'matched_emitter_id', 'similarities','cluster_id']

DEBUG_UN_INFO = False
DEBUG_UN_SAVE_CSV = False
DEBUG_UN_PLOTS = False




