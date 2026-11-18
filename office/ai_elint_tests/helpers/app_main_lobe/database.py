import os
import sqlite3
import pandas as pd
import clickhouse_connect
import os
import logging
log = logging.getLogger(__name__)

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from config import PG_CONFIG, PG_TABLE_SIGNALS, PG_TABLE_CLUSTERS


from datetime import datetime, timezone
from config import WATERMARK_FILE_PATH, INIT_MODE, SPECIFIC_START_TIME


import typing
import numpy as np
import pandas as pd
import psycopg2
from config import PG_TABLE_CLUSTERS
import psycopg2.extras as pg_extras
import typing
import pandas as pd
import psycopg2
from config import PG_TABLE_SIGNALS
from config import PG_TABLE_EMITTERS
import time
# Import configurations explicitly
from config import (
    CLICKHOUSE_CONFIG,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_VERIFY_TLS,
    SPECIFIC_START_TIME,
    FREQ_MIN_HZ,
    FREQ_MAX_HZ,
    FETCH_LIMIT
)


# Exact translation of your MATLAB ModulationTypeEnum
MODULATION_MAPPING = {
    'UNKNOWN': 1,
    'UNMODULATED': 2,
    'LFM_UP': 3,
    'NLFM_UP': 4,
    'LFM_DOWN': 5,
    'NLFM_DOWN': 6,
    'FMCW': 7,
    'BARKER_2A': 8,
    'BARKER_2B': 9,
    'BARKER_3': 10,
    'BARKER_4A': 11,
    'BARKER_4B': 12,
    'BARKER_5': 13,
    'BARKER_7': 14,
    'BARKER_11': 15,
    'BARKER_13': 16,
    'AM': 17
}


def connect_clickhouse():
    """Establishes a secure connection to the ClickHouse database."""
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_CONFIG['host'],
        port=CLICKHOUSE_CONFIG['port'],
        username=CLICKHOUSE_CONFIG['user'],
        database=CLICKHOUSE_CONFIG['database'],
        secure=CLICKHOUSE_SECURE,
        verify=CLICKHOUSE_VERIFY_TLS
    )
    print("Connected securely to ClickHouse database.")
    return client


def init_sqlite_database() -> tuple[sqlite3.Connection, str]:
    """Removes legacy DB, opens a fresh SQLite connection with WAL mode, and finds the watermark."""
    if os.path.exists(SQLITE_DB_PATH):
        try:
            os.remove(SQLITE_DB_PATH)
            print(f"Removed legacy database file: {SQLITE_DB_PATH}")
        except OSError as e:
            print(f"Warning: Could not remove {SQLITE_DB_PATH} ({e}). It might be locked.")

    conn = sqlite3.connect(SQLITE_DB_PATH, timeout=SQLITE_TIMEOUT, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Essential for streaming concurrency

    last_processed_time = DEFAULT_START_WATERMARK
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(time_of_arrival_ns) FROM emitter_signals")
        res = cursor.fetchone()
        if res and res[0] is not None:
            last_processed_time = str(res[0])
            print(f"Resuming pipeline from last known SQLite timestamp: {last_processed_time}")
    except sqlite3.OperationalError:
        print(f"Starting fresh pipeline from baseline time: {last_processed_time}")

    return conn, last_processed_time


def fetch_new_data(client, last_processed_time: str, side_lobe_flag = 1) -> pd.DataFrame:
    """Queries ClickHouse for new signal records based on the last processed timestamp."""
    query = f"""
        SELECT id, power_dbm, center_frequency_hz, time_of_arrival_ns, duration_ns,bandwidth_hz,direction_of_arrival_deg ,snr_db,modulation
        FROM {CLICKHOUSE_CONFIG['table']}
        WHERE time_of_arrival_ns > '{last_processed_time}'
          AND center_frequency_hz > {FREQ_MIN_HZ} 
          AND center_frequency_hz < {FREQ_MAX_HZ} 
          AND sidelobe_blanking_flag = 1
          AND is_cw = False
        ORDER BY time_of_arrival_ns ASC
        LIMIT {FETCH_LIMIT}
    """
    return client.query_df(query)





def load_last_processed_time() -> str:
    """Evaluates INIT_MODE configuration rules to return the correct starting watermark."""
    mode = INIT_MODE.upper()

    # Strategy 1: Dynamic system clock time extraction
    if mode == "CURRENT":
        # Generates a string like: "2026-07-28 15:20:00.000000000" matching ClickHouse schema
        current_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f") + "000"
        print(f" LIVE STREAMING: Bypassing history. Starting loop from current clock runtime: {current_utc_str}")
        return current_utc_str

    # Strategy 2: Manual hardcoded baseline specification
    if mode == "SPECIFIC":
        print(f" FORCED OVERRIDE: Starting pipeline tracking at specified baseline: {SPECIFIC_START_TIME}")
        return SPECIFIC_START_TIME

    # Strategy 3: Standard text file watermark recovery (Default fallback)
    if mode == "WATERMARK":
        if not os.path.exists(WATERMARK_FILE_PATH):
            print(f"🌱 No tracking file found. Falling back to SPECIFIC baseline configuration: {SPECIFIC_START_TIME}")
            return SPECIFIC_START_TIME

        try:
            with open(WATERMARK_FILE_PATH, "r") as f:
                timestamp = f.read().strip()
                if timestamp:
                    print(f"CRASH RECOVERY: Resuming pipeline from saved watermark file: {timestamp}")
                    return timestamp
        except Exception as e:
            print(f"Warning: Could not read tracking file ({e}). Falling back to specified baseline.")

    return SPECIFIC_START_TIME


def save_last_processed_time(timestamp: str):
    """Atomically updates the text file with the latest processed timestamp."""
    # We only care about updating the file if the user intends to track states
    if INIT_MODE.upper() == "WATERMARK":
        try:
            with open(WATERMARK_FILE_PATH, "w") as f:
                f.write(str(timestamp))
        except Exception as e:
            print(f"Error: Failed to write tracking watermark to disk: {e}")








def connect_postgres():
    """Establishes a connection to the Postgres database."""
    conn = psycopg2.connect(
        host=PG_CONFIG['host'],
        port=PG_CONFIG['port'],
        database=PG_CONFIG['database'],
        user=PG_CONFIG['user'],
        password=PG_CONFIG['password']
    )
    conn.autocommit = True  # Ensures transactions commit automatically
    print("Connected successfully to Postgres database.")
    return conn


def insert_cluster_to_postgres(pg_conn, cluster_df: pd.DataFrame, signal_id: int) -> typing.Optional[int]:
    """
    Calculates macro signal metrics from a dataframe chunk, logs a single
    aggregated statistical summary row into the clusters table, and RETURNS the new row ID.
    """
    if cluster_df.empty:
        return None
    print("insert_cluster_to_postgres")

    # 1. Compute statistical aggregations
    pdw_count = int(len(cluster_df))

    # Force arrays to explicit numeric types to prevent hidden Timestamp/object issues
    freq_vals = pd.to_numeric(cluster_df['center_frequency_hz'], errors='coerce').to_numpy()
    mean_frequency = int(np.nanmean(freq_vals)) if len(freq_vals) > 0 else 0
    min_frequency  = int(np.nanmin(freq_vals)) if len(freq_vals) > 0 else 0
    max_frequency  = int(np.nanmax(freq_vals)) if len(freq_vals) > 0 else 0
    std_frequency  = float(np.nanstd(freq_vals, ddof=0)) if len(freq_vals) > 0 else 0.0

    power_vals = pd.to_numeric(cluster_df['power_dbm'], errors='coerce').to_numpy()
    mean_power = float(np.nanmean(power_vals)) if len(power_vals) > 0 else 0.0
    min_power  = float(np.nanmin(power_vals)) if len(power_vals) > 0 else 0.0
    max_power  = float(np.nanmax(power_vals)) if len(power_vals) > 0 else 0.0
    std_power  = float(np.nanstd(power_vals, ddof=0)) if len(power_vals) > 0 else 0.0

    pw_vals = pd.to_numeric(cluster_df['duration_us'], errors='coerce').to_numpy()
    mean_pw = float(np.nanmean(pw_vals)) if len(pw_vals) > 0 else 0.0
    min_pw  = float(np.nanmin(pw_vals)) if len(pw_vals) > 0 else 0.0
    max_pw  = float(np.nanmax(pw_vals)) if len(pw_vals) > 0 else 0.0
    std_pw  = float(np.nanstd(pw_vals, ddof=0)) if len(pw_vals) > 0 else 0.0

    # Explicitly convert the datetime series to Unix Epoch Nanoseconds (Int64)
    # This safely protects you from Timestamp exceptions entirely
    toa_series = pd.to_datetime(cluster_df['time_of_arrival_ns'], errors='coerce').astype('int64')
    toa_vals = toa_series.to_numpy()

    # Note: pandas assigns a specific large negative value (-9223372036854775808) to missing/NaT datetimes.
    # We filter them out to ensure accuracy.
    valid_toa = toa_vals[toa_vals > 0]
    first_toa_ns = int(np.min(valid_toa)) if len(valid_toa) > 0 else 0

    # Calculate Pulse Repetition Interval (PRI) metrics in microseconds
    pri_vals = toa_series.diff().to_numpy() / 1000.0
    valid_pri = pri_vals[~np.isnan(pri_vals)]

    # --- Outlier Removal using IQR ---
    if len(valid_pri) >= 4:  # Requires a few points to calculate quartiles reliably
        q25, q75 = np.percentile(valid_pri, [25, 75])
        iqr = q75 - q25

        # Define thresholds (1.5 is standard, use 3.0 to only catch extreme values)
        lower_bound = q25 - (1.5 * iqr)
        upper_bound = q75 + (1.5 * iqr)

        # Filter the array
        valid_pri = valid_pri[(valid_pri >= lower_bound) & (valid_pri <= upper_bound)]
    # ---------------------------------

    mean_pri = float(np.mean(valid_pri)) if len(valid_pri) > 0 else 0.0
    min_pri  = float(np.min(valid_pri)) if len(valid_pri) > 0 else 0.0
    max_pri  = float(np.max(valid_pri)) if len(valid_pri) > 0 else 0.0
    std_pri  = float(np.std(valid_pri, ddof=0)) if len(valid_pri) > 0 else 0.0


    # 2. Build the query string with RETURNING id appended at the very end
    query = f"""
        INSERT INTO {PG_TABLE_CLUSTERS} (
            name, pdw_count, toa_ns, signal_id,
            frequency_hz, min_frequency_hz, max_frequency_hz, std_frequency_hz,
            power_dbm, min_power_dbm, max_power_dbm, std_power_dbm,
            pulse_width_us, min_pulse_width_us, max_pulse_width_us, std_pulse_width_us,
            pulse_repetition_interval_us, min_pulse_repetition_interval_us, max_pulse_repetition_interval_us, std_pulse_repetition_interval_us,
            created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            NOW(), NOW()
        )
        RETURNING id;
    """

    params = (
        f"emitter_{signal_id}_summary",
        pdw_count,
        first_toa_ns,
        int(signal_id),
        mean_frequency, min_frequency, max_frequency, std_frequency,
        mean_power, min_power, max_power, std_power,
        mean_pw, min_pw, max_pw, std_pw,
        mean_pri, min_pri, max_pri, std_pri
    )

    # 3. Commit values and fetch the returned ID
    try:
        with pg_conn.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result:
                generated_summary_id = result[0]
                print(f"Summary Logged: Saved stats for emitter {signal_id}. Generated Row ID: {generated_summary_id}")
                return generated_summary_id

    except Exception as e:
        print(f"Error executing Postgres statistical summary insertion query: {e}")

    return None



def insert_cluster_to_postgres_(pg_conn, cluster_df: pd.DataFrame, signal_id: int) -> typing.Optional[int]:
    """
    Calculates macro signal metrics from a dataframe chunk, logs a single
    aggregated statistical summary row into the clusters table, and RETURNS the new row ID.
    """
    if cluster_df.empty:
        return None
    print("insert_cluster_to_postgres")

    # 1. Compute statistical aggregations
    pdw_count = int(len(cluster_df))

    # Force arrays to explicit numeric types to prevent hidden Timestamp/object issues
    freq_vals = pd.to_numeric(cluster_df['center_frequency_hz'], errors='coerce').to_numpy()
    mean_frequency = int(np.nanmean(freq_vals)) if len(freq_vals) > 0 else 0
    min_frequency  = int(np.nanmin(freq_vals)) if len(freq_vals) > 0 else 0
    max_frequency  = int(np.nanmax(freq_vals)) if len(freq_vals) > 0 else 0
    std_frequency  = float(np.nanstd(freq_vals, ddof=0)) if len(freq_vals) > 0 else 0.0

    power_vals = pd.to_numeric(cluster_df['power_dbm'], errors='coerce').to_numpy()
    mean_power = float(np.nanmean(power_vals)) if len(power_vals) > 0 else 0.0
    min_power  = float(np.nanmin(power_vals)) if len(power_vals) > 0 else 0.0
    max_power  = float(np.nanmax(power_vals)) if len(power_vals) > 0 else 0.0
    std_power  = float(np.nanstd(power_vals, ddof=0)) if len(power_vals) > 0 else 0.0

    pw_vals = pd.to_numeric(cluster_df['duration_us'], errors='coerce').to_numpy()
    mean_pw = float(np.nanmean(pw_vals)) if len(pw_vals) > 0 else 0.0
    min_pw  = float(np.nanmin(pw_vals)) if len(pw_vals) > 0 else 0.0
    max_pw  = float(np.nanmax(pw_vals)) if len(pw_vals) > 0 else 0.0
    std_pw  = float(np.nanstd(pw_vals, ddof=0)) if len(pw_vals) > 0 else 0.0

    # Explicitly convert the datetime series to Unix Epoch Nanoseconds (Int64)
    # This safely protects you from Timestamp exceptions entirely
    toa_vals = pd.to_datetime(cluster_df['time_of_arrival_ns'], errors='coerce').astype('int64').to_numpy()

    # Note: pandas assigns a specific large negative value (-9223372036854775808) to missing/NaT datetimes.
    # We filter them out to ensure accuracy.
    valid_toa = toa_vals[toa_vals > 0]
    first_toa_ns = int(np.min(valid_toa)) if len(valid_toa) > 0 else 0

    # 2. Build the query string with RETURNING id appended at the very end
    query = f"""
        INSERT INTO {PG_TABLE_CLUSTERS} (
            name, pdw_count, toa_ns, signal_id,
            frequency_hz, min_frequency_hz, max_frequency_hz, std_frequency_hz,
            power_dbm, min_power_dbm, max_power_dbm, std_power_dbm,
            pulse_width_us, min_pulse_width_us, max_pulse_width_us, std_pulse_width_us,
            created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            NOW(), NOW()
        )
        RETURNING id;
    """

    params = (
        f"emitter_{signal_id}_summary",
        pdw_count,
        first_toa_ns,
        int(signal_id),
        mean_frequency, min_frequency, max_frequency, std_frequency,
        mean_power, min_power, max_power, std_power,
        mean_pw, min_pw, max_pw, std_pw
    )

    # 3. Commit values and fetch the returned ID
    try:
        with pg_conn.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result:
                generated_summary_id = result[0]
                print(f"Summary Logged: Saved stats for emitter {signal_id}. Generated Row ID: {generated_summary_id}")
                return generated_summary_id

    except Exception as e:
        print(f"Error executing Postgres statistical summary insertion query: {e}")

    return None




def add_pdws_to_clusters_table_ch(
    cluster_id: int,
    cluster_df: pd.DataFrame,
    ch_client: clickhouse_connect.driver.Client
):
    """
    Inserts a single structural row mapping a Postgres cluster tracking ID
    to an array of incoming raw pulse activity UUID strings into ClickHouse.
    """
    # 1. Edge guard protection for empty batches
    if cluster_df.empty or cluster_id is None:
        return

    try:
        # 2. Extract and cast all the unique pulse string IDs from your dataframe
        # Cleans and forces them to standard strings to avoid numpy conversion errors
        raw_ids = cluster_df['id'].dropna().unique()
        activity_uuids = [str(uid) for uid in raw_ids]
        if not activity_uuids:
            print("Skipping ClickHouse insertion: No valid pulse IDs found in the window.")
            return

        # 3. Assemble parameters exactly matching ClickHouse client structural data formats
        # Outer list encapsulates the single row sequence matrix: [ [cluster_id, [uuid1, uuid2...]] ]
        data_to_insert = [
            [int(cluster_id), activity_uuids]
        ]
        # 4. Stream data directly to ClickHouse via high-speed native protocol
        # Omitting 'created_at' lets the table apply 'DEFAULT now()' automatically
        ch_client.insert(
            table='clusters',
            data=data_to_insert,
            column_names=['cluster_id', 'activity_ids']
        )

        print(f" ClickHouse Logged: Mapped cluster_id {cluster_id} to {len(activity_uuids)} activity UUIDs.")

    except Exception as e:
        print(f"Error executing ClickHouse bulk insertion sequence query: {e}")




def update_existing_signal(pg_conn, signal_id):
    """
    Updates the duration_ms of a signal based on the difference between
    the current database time and its Time of Arrival (toa_seconds).
    """
    query = """
        UPDATE signals
        SET duration_ms = EXTRACT(EPOCH FROM (NOW()::timestamp - toa_seconds)) * 1000.0
        WHERE id = %s;
    """

    with pg_conn.cursor() as cursor:
        cursor.execute(query, (signal_id,))
        pg_conn.commit()


def update_doa_in_signal(pg_conn, signal_id, angle):
    query = """
        UPDATE signals
        SET doa_deg = %s
        WHERE id = %s;
    """

    with pg_conn.cursor() as cursor:
        cursor.execute(query, (float(angle), int(signal_id)))
        pg_conn.commit()


PG_TABLE_EMITTERS = "emitters"

SIGNATURE_MAP = {
    1104: 22,   # PSR 1
    1105: 23,   # PSR 2
    1106: 24,   # PSR 3
    1107: 25,   # PSR 4
    1108: 26,   # PSR 5 Mod 1
    1116: 28,   # PSR 5 Mod 2
    1113: 32,   # MF 03
    1115: 33,   # MF 05
    1109: 29,   # EW 03 (Old MF 04)
    1117: 38    # MF 06
}

def insert_emitter(pg_conn, name: typing.Union[int, str]) -> typing.Optional[int]:
    if name == "unknown_ai":
        query = f"""
            INSERT INTO {PG_TABLE_EMITTERS}
                (name, created_at, updated_at)
            VALUES
                (%s, NOW(), NOW())
            RETURNING id;
        """
        params = (name,)
    else:
        signature_matched_id = SIGNATURE_MAP.get(name)
        if signature_matched_id is None:
            print(f"Unknown emitter name '{name}', skipping insert.")
            return None

        query = f"""
            INSERT INTO {PG_TABLE_EMITTERS}
                (name, created_at, updated_at, signature_matched_id)
            VALUES
                (%s, NOW(), NOW(), %s)
            RETURNING id;
        """
        params = (name, signature_matched_id)

    try:
        with pg_conn.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"Error inserting emitter '{name}': {e}")
        return None



def insert_signal(pg_conn, cluster_df: pd.DataFrame, emitter_id, angle) -> typing.Optional[int]:
    """
    Directly appends a new signal record containing frequency metadata into Postgres.
    Allows 'id' to auto-increment naturally.

    Returns:
        The auto-incremented integer ID assigned by the database, or None if it fails.
    """
    if cluster_df.empty:
        return None

    # Helper function to safely extract the first row and fall back if missing/NaN
    def get_safe_int(col_name: str, fallback: int) -> int:
        if col_name not in cluster_df.columns:
            return fallback
        val = cluster_df[col_name].iloc[0]
        if pd.isna(val) or val is None:
            return fallback
        return int(val)

    # 1. Extract and explicitly cast DataFrame values to native Python ints
    freq          = get_safe_int('center_frequency_hz', 0)
    bw            = get_safe_int('bandwidth_hz', 0)
    receiver      = get_safe_int('receiver_id', 1)
    power         = get_safe_int('power_dbm', -40)
    snr_db        = get_safe_int('snr_db', 0)
    time_ns       = time.time_ns()

    # Extract modulation (handles both text names and raw fallback integer IDs)
    raw_mod = cluster_df['modulation'].iloc[0] if 'modulation' in cluster_df.columns else None

    if pd.isna(raw_mod) or raw_mod is None:
        modulation = 1  # Map to UNKNOWN (1) as a logical default
    else:
        # Convert string to uppercase and strip whitespace (e.g. 'unmodulated ' -> 'UNMODULATED')
        mod_str = str(raw_mod).strip().upper()

        # Look up the ID from your Enum table, or check if it's already a digit string
        if mod_str in MODULATION_MAPPING:
            modulation = MODULATION_MAPPING[mod_str]
        elif mod_str.isdigit():
            modulation = int(mod_str)
        else:
            print(f"Warning: Unknown modulation string '{raw_mod}'. Defaulting to UNKNOWN (1)")
            modulation = 1


    # ADDED: Defined emission type string for custom DB enum mapping
    emission_type = "PULSE"

    # 2. SANITIZE EXTERNAL INPUTS: Convert NumPy types to native Python float/int/None
    doa_deg = float(angle) if (angle is not None and not pd.isna(angle)) else None

    if emitter_id is not None and not pd.isna(emitter_id):
        clean_emitter_id = int(emitter_id)
    else:
        clean_emitter_id = None

    # Fixed operator precedence bug
    raw_duration = (cluster_df['TOA'].max() - cluster_df['TOA'].min()) / 1000000
    duration = float(raw_duration)
    # print(f"duration ::::: {duration}")

    is_ai = True
    print(f"Sanitized values to insert -> SNR: {snr_db}, Freq: {freq}, DOA: {doa_deg}, Emitter: {clean_emitter_id} modulation: {modulation}")

    # UPDATED: Added emission_type and an extra %s placeholder to the query structure
    query = f"""
    INSERT INTO {PG_TABLE_SIGNALS} 
        (frequency_hz, bandwidth_hz, receiver_id, power_dbm, snr_db, doa_deg, emitter_id, is_ai, modulation_type_id, toa_nanoseconds, duration_ms, emission_type, toa_seconds) 
    VALUES 
        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()) 
    RETURNING id;
    """

    try:
        with pg_conn.cursor() as cursor:
            # UPDATED: Included emission_type variable in the execution tuple
            cursor.execute(query, (
                freq, bw, receiver, power, snr_db, doa_deg,
                clean_emitter_id, is_ai, modulation, time_ns, duration, emission_type
            ))
            result = cursor.fetchone()

            if result:
                new_id = result[0]
                pg_conn.commit()
                print(f"Signals Database: Appended fresh row with Auto-Incremented ID {new_id}")
                return new_id

    except Exception as e:
        pg_conn.rollback()
        print(f"Error executing raw Postgres signal insertion query: {e}")

    return None





    import psycopg2.extras as pg_extras
from datetime import datetime

# --- POSTGRES AT-ONCE BATCH LOADER ---
def bulk_insert_clusters_to_postgres(pg_conn, metrics_records: list) -> dict:
    """
    Sends all summary records across all clusters to Postgres
    in a single packed batch operation.
    """
    query = f"""
    INSERT INTO clusters (
        name, pdw_count, toa_ns, signal_id, frequency_hz, min_frequency_hz, max_frequency_hz, std_frequency_hz,
        power_dbm, min_power_dbm, max_power_dbm, std_power_dbm, pulse_width_us, min_pulse_width_us, max_pulse_width_us, std_pulse_width_us,
        pulse_repetition_interval_us, min_pulse_repetition_interval_us, max_pulse_repetition_interval_us, std_pulse_repetition_interval_us,
        bandwidth_hz, min_bandwidth_hz, max_bandwidth_hz, std_bandwidth_hz,
        doa_deg, min_doa_deg, max_doa_deg, std_doa_deg, 
        is_ai, created_at, updated_at
    ) VALUES %s RETURNING signal_id, id;
    """

    params_list = []
    # Fetch current time once to keep it uniform across all rows
    current_time = datetime.now()

    for r in metrics_records:
        params_list.append((
            f"emitter_{int(r['target_signal_id'])}_summary",
            int(r['pdw_count']),
            int(r['toa_ns']),
            int(r['target_signal_id']),
            int(r['freq_mean']),
            int(r['freq_min']),
            int(r['freq_max']),
            float(r['freq_std']),
            float(r['power_mean']),
            float(r['power_min']),
            float(r['power_max']),
            float(r['power_std']),
            float(r['pw_mean']),
            float(r['pw_min']),
            float(r['pw_max']),
            float(r['pw_std']),
            float(r['pri_mean']),
            float(r['pri_min']),
            float(r['pri_max']),
            float(r['pri_std']),
            float(r['bw_mean']),
            float(r['bw_min']),
            float(r['bw_max']),
            float(r['bw_std']),
            float(r['doa_mean']),
            float(r['doa_min']),
            float(r['doa_max']),
            float(r['doa_std']),
            True,
            current_time, # FIXED: Changed from pd.Timestamp
            current_time  # FIXED: Changed from pd.Timestamp
        ))

    signal_to_pg_id_map = {}
    try:
        with pg_conn.cursor() as cursor:
            pg_extras.execute_values(
                cursor,
                query,
                params_list,
                template=None,
                page_size=100
            )

            results = cursor.fetchall()
            for signal_id, generated_id in results:
                signal_to_pg_id_map[int(signal_id)] = generated_id

    except Exception as e:
        print(f"Database insertion failed: {e}")
        raise e

    return signal_to_pg_id_map

