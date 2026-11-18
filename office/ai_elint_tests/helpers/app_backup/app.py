import os
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import time
# Local configuration and modular pipeline utility imports
from config import EMBEDDING_COLS, ONNX_MODEL_PATH
from database import (
    connect_clickhouse,
    connect_postgres,
    fetch_new_data,
    load_last_processed_time,
    save_last_processed_time
)
from inference import init_onnx_session, run_onnx_inference
from cluster_handling import (
    handle_embedding,
    load_tracking_cache,
    load_cluster_centers_cache
)
from db_utils import (truncate_tables, create_populate_csv_input_table, create_csv_input_table)

if __name__ == "__main__":

    known_clusters_file = "known_cluster_centers.json"

    # 1. Initialize machine learning sessions and connect drivers
    onnx_session = init_onnx_session(ONNX_MODEL_PATH)
    client = connect_clickhouse()
    pg_conn = connect_postgres()  # Fix: Added import reference in header

    # 2. Boot up local tracking array and cluster centers matrices from disk
    # load_tracking_cache()
    # load_cluster_centers_cache()

    # 3. Establish initial watermark timestamp based on configured rule selection
    last_processed_time = load_last_processed_time()

    window_global_count = 0
    print(" Pipeline successfully initiated. Entering ClickHouse polling loop...")
    input_csv_filepath = os.getenv('INPUT_CSV_PATH', 'MF-05_1.csv')
    file_path = input_csv_filepath
    #create_csv_input_table(client)
    # truncate_tables(client, pg_conn)
    # create_populate_csv_input_table(file_path,client)

    # --- Infinite Stream Processing Loop ---
    #while True:
    while window_global_count <1:
        try:
            if os.path.exists(known_clusters_file):
                os.remove(known_clusters_file)
            df_full = pd.read_csv(file_path)

            df   = df_full[(df_full["source"] == "spinning-DF") & (df_full["sidelobe_blanking_flag_as_int8"] == 1)]
            #df   = df_full.copy()
            if len(df) == 0:
                print("No new streaming data found in ClickHouse. Sleeping for 5 seconds...")
                time.sleep(5)
                continue



            print(f"Retrieved {len(df)} incoming rows from ClickHouse.")
            df['id'] = df['id'].astype(str)


            # Map columns cleanly to avoid modifying the source DataFrame array dimensions
            selected_df = df[['id', 'power_dbm', 'center_frequency_hz', 'time_of_arrival_ns', 'duration_ns','bandwidth_hz','snr_db','direction_of_arrival_deg']].copy().reset_index(drop=True)
            selected_df['duration_us'] = selected_df['duration_ns'] / 1000.0



            power_raw = selected_df['power_dbm'].values.astype(np.float32)
            freq_raw  = selected_df['center_frequency_hz'].values.astype(np.float32)
            pw_raw    = selected_df['duration_us'].values.astype(np.float32)

            # Convert raw Time of Arrival to clean float evaluation targets
            dt_series = pd.to_datetime(selected_df['time_of_arrival_ns']).dt.tz_localize(None)
            toa_raw = dt_series.astype('int64').astype(np.float64).to_numpy()
            selected_df['TOA'] = toa_raw
            total_windows = len(selected_df) // 512
            print(f"total_windows{total_windows}")
            if total_windows == 0:
                print("Insufficient rows (< 512) to formulate a processing window. Waiting for data backlog...")
                time.sleep(5)
                continue
            #total_windows = 3
            # Process data windows sequentially in 512-row chunks
            for count in range(total_windows):
                start_idx = 512 * count
                end_idx = 512 * (count + 1)
                raw_toa = toa_raw[start_idx:end_idx] / 1000.0
                freq = freq_raw[start_idx:end_idx]/10e5
                pw = pw_raw[start_idx:end_idx]
                power = power_raw[start_idx:end_idx]

                start = time.perf_counter()
                # Execute feature calculations and ONNX graph inference steps
                out = run_onnx_inference(onnx_session, freq, power, pw, raw_toa)

                # Reconstruct matching metadata slice for these 512 pulses
                pulses_batch_df = selected_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)


                # Process known profiles and direct anomalies down to Agglomerative Clustering
                # Fix: Passed 'client' (ClickHouse client engine) instead of undefined 'cl_client'
                handle_embedding(out, pulses_batch_df, pg_conn, client)

                print( (time.perf_counter() - start) * 1000)
                '''
                # Advance and update watermarks progressively 
                last_processed_time = str(selected_df['time_of_arrival_ns'].iloc[end_idx - 1])
                save_last_processed_time(last_processed_time)
                '''
                # ts = selected_df['time_of_arrival_ns'].iloc[end_idx - 1]
                # last_processed_time = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + f"{ts.nanosecond:03d}"
                #
                # save_last_processed_time(last_processed_time)

                window_global_count += 1

            print(f"Batch completed successfully. Synced with ClickHouse up to timestamp: {last_processed_time}")

        except Exception as e:
            print(f"An unexpected error occurred in the execution loop: {e}")
            time.sleep(5)
    if os.path.exists(known_clusters_file):
        os.remove(known_clusters_file)
