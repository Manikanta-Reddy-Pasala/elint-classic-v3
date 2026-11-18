import os
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import time
import logger
import config

# Local configuration and modular pipeline utility imports
from config import ONNX_MODEL_PATH
from cluster_handling import CACHE_FILE_PATH, CENTERS_FILE_PATH, MEDOIDS_FILE_PATH
from pathlib import Path

from database import (
    connect_clickhouse,
    connect_postgres,
    fetch_new_data,
    load_last_processed_time
)
from inference import init_onnx_session, run_onnx_inference
from cluster_handling import (
    handle_embedding,
    load_tracking_cache,
    load_cluster_centers_cache,
    load_medoids_cache
)

if __name__ == "__main__":
    # 1. Initialize machine learning sessions and connect drivers
    onnx_session = init_onnx_session(ONNX_MODEL_PATH)
    client = connect_clickhouse()
    pg_conn = connect_postgres()  # Fix: Added import reference in header
    p = Path(CACHE_FILE_PATH)
    if p.exists():
        p.unlink()
    p = Path(CENTERS_FILE_PATH)
    if p.exists():
        p.unlink()
    p = Path(MEDOIDS_FILE_PATH)
    if p.exists():
        p.unlink()
    logger.set_logger_level(logger.LOG_INFO)

    # 2. Boot up local tracking array and cluster centers matrices from disk
    load_tracking_cache()
    load_cluster_centers_cache()
    load_medoids_cache()

    # 3. Establish initial watermark timestamp based on configured rule selection
    last_processed_time = load_last_processed_time()


    logger.LOG_DBG(" Pipeline successfully initiated. Entering ClickHouse polling loop...")

    ############################
    # Test Configuration
    window_global_count = 0
    streaming_data = False
    if "--online" in sys.argv:
        streaming_data = True
    side_lobe_flag = 1
    ############################

    if  streaming_data == False:
        input_csv_filepath = os.getenv('INPUT_CSV_PATH', 'MF-05_1.csv')
        file_path = input_csv_filepath


    if config.DEBUG_UN_SAVE_CSV:
        config.UNKNOWN_CLUSTER_FILENAME = file_path + config.UNKNOWN_CLUSTER_FILENAME + '_' + filename
        if os.path.exists(config.UNKNOWN_CLUSTER_FILENAME):
            os.remove(config.UNKNOWN_CLUSTER_FILENAME)

    # --- Infinite Stream Processing Loop ---
    while True:
    # while window_global_count <1:
        try:
            if streaming_data == True:
                # Query incoming streaming telemetry frames using active watermark
                df_full = fetch_new_data(client, last_processed_time, side_lobe_flag)
                logger.LOG_INF(f"👉 Processing next set of streaming data")
            else:
                df_full = pd.read_csv(file_path)
                logger.LOG_INF(f"👉 Processing CSV file from: {file_path}")

            if len(df_full) == 0:
                logger.LOG_WRN("No new streaming data found in ClickHouse. Sleeping for 5 seconds...")
                time.sleep(5)
                continue

            if ('sidelobe_blanking_flag_as_int8' in df_full.columns) and side_lobe_flag == 1:
                mask = df_full['sidelobe_blanking_flag_as_int8'] == 1
                df = df_full[mask]
                logger.LOG_INF("Using only Main lobe pulses to feed to AI ---")
            else:
                df = df_full

            logger.LOG_DBG(f"Retrieved {len(df)} incoming rows from ClickHouse.")
            df['id'] = df['id'].astype(str)

            if 'emitter_id' not in df.columns:
                df['emitter_id'] = 0

            if 'modulation' not in df.columns:
                df['modulation'] = 0

            df['direction_of_arrival_deg'] = pd.to_numeric(df['direction_of_arrival_deg'], errors='coerce')
            df['power_dbm'] = pd.to_numeric(df['power_dbm'], errors='coerce')
            # Map columns cleanly to avoid modifying the source DataFrame array dimensions
            selected_df = df[['id', 'power_dbm', 'center_frequency_hz', 'time_of_arrival_ns', 'duration_ns','bandwidth_hz','snr_db','direction_of_arrival_deg','modulation', 'emitter_id']].copy().reset_index(drop=True)
            selected_df['duration_us'] = selected_df['duration_ns'] / 1000.0


            power_raw = selected_df['power_dbm'].values.astype(np.float32)
            freq_raw  = selected_df['center_frequency_hz'].values.astype(np.float32)
            pw_raw    = selected_df['duration_us'].values.astype(np.float32)

            # Convert raw Time of Arrival to clean float evaluation targets
            dt_series = pd.to_datetime(selected_df['time_of_arrival_ns']).dt.tz_localize(None)
            toa_raw = dt_series.astype('int64').astype(np.float64).to_numpy()
            selected_df['TOA'] = toa_raw
            total_windows = len(selected_df) // config.PDW_CHUNK_SIZE
            logger.LOG_DBG(f"total_windows{total_windows}")
            if total_windows == 0:
                logger.LOG_WRN(f"Insufficient rows (< {config.PDW_CHUNK_SIZE}) to formulate a processing window. Waiting for data backlog...")
                time.sleep(5)
                continue
            # total_windows = 11
            # Process data windows sequentially in chunks
            for count in range(total_windows):
                start_idx = config.PDW_CHUNK_SIZE * count
                end_idx = config.PDW_CHUNK_SIZE * (count + 1)
                logger.LOG_INF(f"🔍 Chunk [{count}] :: {start_idx} -> {end_idx} of total {len(selected_df)}")

                raw_toa = toa_raw[start_idx:end_idx] / 1000.0
                freq = freq_raw[start_idx:end_idx]/10e5
                pw = pw_raw[start_idx:end_idx]
                power = power_raw[start_idx:end_idx]

                start = time.perf_counter()
                # Execute feature calculations and ONNX graph inference steps
                out = run_onnx_inference(onnx_session, freq, power, pw, raw_toa)

                # Reconstruct matching metadata slice for this chunk
                pulses_batch_df = selected_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)


                # Process known profiles and direct anomalies down to Agglomerative Clustering
                # Fix: Passed 'client' (ClickHouse client engine) instead of undefined 'cl_client'
                handle_embedding(out, pulses_batch_df, pg_conn, client)

                logger.LOG_DBG((time.perf_counter() - start) * 1000)
                '''
                # Advance and update watermarks progressively 
                last_processed_time = str(selected_df['time_of_arrival_ns'].iloc[end_idx - 1])
                save_last_processed_time(last_processed_time)
                '''
                ts = selected_df['time_of_arrival_ns'].iloc[end_idx - 1]
                if not hasattr(ts, 'strftime'):
                    ts = pd.to_datetime(ts)

                last_processed_time = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + f"{ts.nanosecond:03d}"

                #save_last_processed_time(last_processed_time)

                window_global_count += 1

            logger.LOG_INF(f"🟢 Batch completed successfully. Synced with ClickHouse up to timestamp: {last_processed_time}")

            if streaming_data == False:
                break

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.LOG_ERR(f"An unexpected error occurred in the execution loop: {e}")
            time.sleep(5)
