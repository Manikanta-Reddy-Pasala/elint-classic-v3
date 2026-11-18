# config.py

# === CLICKHOUSE DATABASE CONFIGURATION ===
CLICKHOUSE_CONFIG = {
    #'host': '10.10.180.233',
    'host': 'db.ai.elint.local',
    'port': 8444,
    'database': 'signals',
    'user': 'default',
    'table': 'signals.pdw'
}
CLICKHOUSE_SECURE = True
CLICKHOUSE_VERIFY_TLS = False

# === DATA FILTERING CRITERIA ===
FREQ_MIN_HZ = 0
FREQ_MAX_HZ = 9957191680
FETCH_LIMIT = 5120
POLLING_INTERVAL_SECONDS = 10



# === UPDATE THIS ONNX MODEL PATH ===
ONNX_MODEL_PATH = "model/transformer_model_known_v14.onnx"

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
SPECIFIC_START_TIME = "2026-08-26 10:24:08.000000000"

#Similarity threshold for known emitters
SIMILARITY_THRESHOLD = 0.80
MIN_CLUSTER_SIZE = 20
UNKNOWN_DOT_PRODUCT_THRESHOLD = 0.8




# === POSTGRESQL (POSTGRES) CONFIGURATION ===
PG_CONFIG = {
    'host': 'db.ai.elint.local',
    'port': 5432,
    'database': 'sigint',
    'user': 'postgres',
    'password': 'postgres'
}

PG_TABLE_CLUSTERS = 'clusters'
PG_TABLE_SIGNALS = 'signals'
PG_TABLE_EMITTERS = 'emitters'





