import os
import json
import typing
import numpy as np
import pandas as pd
import logger
from create_features import generate_features_per_window
from sklearn.cluster import HDBSCAN
from cluster_refinement import get_cluster_score, get_cluster_angle_score, get_cluster_angle
from cluster_engine import EmitterTrack
# Centralized configurations
from config import SIMILARITY_THRESHOLD,SIMILARITY_THRESHOLD_1117, MIN_CLUSTER_SIZE, UNKNOWN_DOT_PRODUCT_THRESHOLD, BUFFER_THRESHOLD, KNOWN_DOT_PRODUCT_THRESHOLD, KNOWN_MIN_SIGNAL_SIZE, UNKNOWN_ACCUMULATION_TIME_SEC
from database import update_doa_in_signal
from custom_clustering import cluster_based_on_embedding_similarities, merge_clusters

# Transactional database access utilities
from database import (
    insert_signal,
    insert_cluster_to_postgres,
    add_pdws_to_clusters_table_ch,
    bulk_insert_clusters_to_postgres,
    insert_emitter,
    update_existing_signal
)
from pipeline import handle_unknown_pulses

# === CACHE AND STORAGE CONFIGURATIONS ===
CACHE_FILE_PATH = "emitter_signal_map.npy"
CENTERS_FILE_PATH = "known_cluster_centers.json"
MEDOIDS_FILE_PATH = "known_medoids.npy"
MAX_EMITTERS = 10000

# In-Memory Lookups
EMITTER_SIGNAL_CACHE = np.zeros(MAX_EMITTERS, dtype=np.uint64)
from cluster_engine import EmitterTrack

KNOWN_CENTERS_CACHE: dict[int, EmitterTrack] = {}


# =============================================================================
# --- 1. BOOT COMPONENT STATE UTILITIES ---
# =============================================================================

def load_tracking_cache():
    """Restores the fast one-to-one pointer locations on boot."""
    global EMITTER_SIGNAL_CACHE
    if os.path.exists(CACHE_FILE_PATH):
        try:
            EMITTER_SIGNAL_CACHE = np.load(CACHE_FILE_PATH)
            print(" NumPy tracking array loaded from disk.")
        except Exception as e:
            print(f"Warning: Failed to load NumPy array cache ({e}).")

def track_signal_id_for_emitter(emitter_id: int, signal_id: int):
    """Updates structural tracking arrays in raw binary format instantly."""
    if emitter_id is None or signal_id is None:
        return
    EMITTER_SIGNAL_CACHE[int(emitter_id)] = np.uint64(signal_id)
    try:
        np.save(CACHE_FILE_PATH, EMITTER_SIGNAL_CACHE)
    except Exception as e:
        print(f"Error flushing tracking matrix to disk: {e}")

def get_signal_id_for_emitter(emitter_id: int) -> typing.Optional[int]:
    """Evaluates assignments via native high-speed C array slicing indexes."""
    val = EMITTER_SIGNAL_CACHE[int(emitter_id)]
    return int(val) if val != 0 else None



def load_cluster_centers_cache():
    """Loads pre-saved vector center clusters and metadata from disk upon initialization."""

    global KNOWN_CENTERS_CACHE
    if os.path.exists(CENTERS_FILE_PATH):
        try:
            with open(CENTERS_FILE_PATH, "r") as f:
                raw_data = json.load(f)

                # Reconstruct the structured NamedTuple dictionary schema
                KNOWN_CENTERS_CACHE = {}
                for k, v in raw_data.items():
                    # If loading legacy data format (just a raw list vector), handle gracefully
                    if isinstance(v, list):
                        vector = np.array(v, dtype=np.float32)
                        angle = 0.0
                        modulation = 1.0
                        cardinality = 1
                        signal_id = 0
                    else:
                        # Standard structured parsing block
                        vector = np.array(v['vector'], dtype=np.float32)
                        angle = float(v.get('angle', 0.0))
                        modulation = float(v.get('modulation', 1.0))
                        cardinality = int(v.get('cardinality', 1))
                        signal_id = int(v.get('signal_id', 0))

                    # Pack directly into your verified type signature definition
                    KNOWN_CENTERS_CACHE[int(k)] = EmitterTrack(
                        vector=vector,
                        angle=angle,
                        modulation=modulation,
                        cardinality=cardinality,
                        signal_id=signal_id
                    )

                print(f" Centers Loaded: Restored {len(KNOWN_CENTERS_CACHE)} historical signal tracks into metadata cache.")
        except Exception as e:
            print(f"Warning: Failed to load cluster centers json ({e}).")

def load_medoids_cache():
    """Restores the medoids dictionary from disk upon initialization."""

    global MEDOIDS
    if os.path.exists(MEDOIDS_FILE_PATH):
        try:
            # Load with allow_pickle=True because we saved a dict of arrays
            loaded_data = np.load(MEDOIDS_FILE_PATH, allow_pickle=True)

            # Extract the dictionary from the 0-dim numpy array wrapper
            raw_medoids = loaded_data.item()

            # Reconstruct to ensure keys are integers and values are arrays
            MEDOIDS = {}
            for k, v in raw_medoids.items():
                # Ensure key is int
                key = int(k)

                # Ensure value is a numpy array (handle cases where it might be a list)
                if isinstance(v, list):
                    MEDOIDS[key] = np.array(v, dtype=np.float32)
                elif isinstance(v, np.ndarray):
                    MEDOIDS[key] = v.astype(np.float32)
                else:
                    # Fallback if data is corrupted
                    continue

            print(f" Medoids Loaded: Restored {len(MEDOIDS)} medoid entries from cache.")

        except Exception as e:
            print(f"Warning: Failed to load medoids cache ({e}).")
            MEDOIDS = {} # Reset on failure
    else:
        MEDOIDS = {}
        print(" No medoids cache found. Starting fresh.")

def save_cluster_centers_cache():
    global KNOWN_CENTERS_CACHE
    try:
        # Convert internal complex schema into basic JSON-safe structures
        serializable_data = {}
        for k, track in KNOWN_CENTERS_CACHE.items():
            serializable_data[int(k)] = {
                'vector': track.vector.tolist(),  # Convert NumPy matrix to standard list
                'angle': float(track.angle),
                'modulation': float(track.modulation),
                'cardinality': int(track.cardinality),
                'signal_id': int(track.signal_id)
            }

        with open(CENTERS_FILE_PATH, "w") as f:
            json.dump(serializable_data, f, indent=2)
        logger.LOG_INF(f" Centers Saved: Backup successful for {len(KNOWN_CENTERS_CACHE)} signal profiles.")
    except Exception as e:
        print(f"Warning: Failed to persist cluster centers database to disk ({e}).")

def save_medoids_cache():
    global MEDOIDS
    try:
        np.save(MEDOIDS_FILE_PATH, MEDOIDS)
        print(f" Medoids Saved (NPY): Backup successful for {len(MEDOIDS)} medoid profiles.")
    except Exception as e:
        print(f"Warning: Failed to persist medoids database to disk ({e}).")

# =============================================================================
# --- 2. PIPELINE CORE PROCESSING ENGINES ---
# =============================================================================

def handle_known_emitters(emitter_ids: np.ndarray, similarities: np.ndarray, pulses_df: pd.DataFrame):
    """
    Identifies known signals matching pre-existing library profiles.
    Returns a dictionary of structured dataframes and a leftover slice mask.
    """
    known_mask = similarities >= SIMILARITY_THRESHOLD
    known_clusters = {}
    print("handle_known_emitters")
    if np.any(known_mask):
        matched_ids = emitter_ids[known_mask]
        matched_pulses = pulses_df.iloc[known_mask]
        unique_emitters = np.unique(matched_ids)

        for e_id in unique_emitters:
            print("found id")
            print(e_id)
            emitter_pulse_mask = (matched_ids == e_id)
            known_clusters[int(e_id)] = matched_pulses.iloc[emitter_pulse_mask]

    remaining_pulses_mask = ~known_mask
    return known_clusters, remaining_pulses_mask





# =============================================================================
# --- 3. CORE MANAGEMENT PIPELINE ORCHESTRATOR ---
# =============================================================================

def handle_embedding(out: tuple, pulses_df: pd.DataFrame, pg_conn, ch_client):
    """
    Main entry point separating signal windows into known libraries
    and cascading anomalies straight down to the discovery loop.
    """
    raw_embeddings = np.squeeze(out[0], axis=0)
    norm = np.linalg.norm(raw_embeddings, ord=2, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    # print(np.dot(raw_embeddings[203],raw_embeddings[205]))
    # Divide the raw matrix by the lengths to normalize them perfectly to unit length
    normalized_embeddings = raw_embeddings #/ norm

    emitter_ids = out[1]
    similarities = out[2]
    logger.LOG_DBG(raw_embeddings.shape)
    embedding_cols = [f"dim_{i}" for i in range(1024)]
    embeddings_df = pd.DataFrame(normalized_embeddings, columns=embedding_cols)


    # Horizontally stack metadata with its matching embedding dimensions
    combined_pulses_df = pd.concat([pulses_df.reset_index(drop=True), embeddings_df], axis=1)

    flat_similarities = np.squeeze(similarities)
    flat_emitter_ids = np.squeeze(emitter_ids)

    combined_pulses_df['matched_emitter_id'] = flat_emitter_ids.astype(int)
    combined_pulses_df['similarities'] = flat_similarities.astype(float)

    # 1. Run primary profile signature matching checks vectorially
    known_df, unknown_pulses = handle_known_emitters_vectorized(combined_pulses_df)


    if not known_df.empty:
        # 2. Map existing signal tracking registrations via a high-speed cached mapping layer
        # For any new registrations, insert_signal runs only once per unique emitter profile group
        emitter_to_signal_map = batch_resolve_signal_ids(pg_conn, known_df)

        if emitter_to_signal_map:
            # Map the resolved signal IDs vectorially into our DataFrame columns
            known_df['target_signal_id'] = known_df['matched_emitter_id'].map(emitter_to_signal_map)

            # Strip out any rows where signal registration completely failed
            clean_known_df = known_df.dropna(subset=['target_signal_id']).copy()

            if not clean_known_df.empty:
                # 3. Vector-calculate macro metrics across all known clusters simultaneously
                known_metrics = calculate_all_cluster_metrics_vectorized(clean_known_df)
                #set doa to signal

                if known_metrics:
                    # 4. BULK INSERT all summaries into Postgres in one single packet round-trip
                    pg_cluster_id_map = bulk_insert_clusters_to_postgres(pg_conn, known_metrics)

                    # 5. Compile structural arrays and BULK INSERT into ClickHouse in one operation
                    ch_bulk_payload = []
                    for sig_id, pg_id in pg_cluster_id_map.items():
                        grp_mask = clean_known_df['target_signal_id'] == sig_id
                        uuids = clean_known_df.loc[grp_mask, 'id'].dropna().astype(str).unique().tolist()
                        if uuids:
                            ch_bulk_payload.append([int(pg_id), uuids])

                    if ch_bulk_payload:
                        ch_client.insert(
                            table='clusters',
                            data=ch_bulk_payload,
                            column_names=['cluster_id', 'activity_ids']
                        )
                        print(f" ClickHouse Logged: Bulk-saved {len(ch_bulk_payload)} known library tracking rows.")
                        df_out = pd.DataFrame(known_metrics)
                        signal_to_emitter = {v: k for k, v in emitter_to_signal_map.items()}
                        # emitter_key = signal_to_emitter[int(sig_id)]
                        df_out["Emitter_ID"] = df_out['target_signal_id'].map(signal_to_emitter)
                        header = not os.path.exists("known_metrics.csv")   # write header only on first run
                        df_out.to_csv("known_metrics.csv", mode="a", header=header, index=False)


    # Route leftovers into the dynamic discovery layer
    #leftover_noise_backlog = handle_unknown_pulses(unknown_pulses, pg_conn, ch_client)
    handle_unknown_pulses_buffered(unknown_pulses, pg_conn, ch_client)








def handle_unknown_pulses_working(unknown_pulses_df: pd.DataFrame, pg_conn, ch_client) -> pd.DataFrame:
    """
    Optimized high-throughput tracking engine.
    1. Vector-maps incoming pulses to pre-saved centers via matrix multiplication.
    2. Runs HDBSCAN on leftovers to isolate new emitters and bulk-inserts all logs.
    """
    if unknown_pulses_df.empty:
        return unknown_pulses_df

    embedding_cols = [f"dim_{i}" for i in range(1024)]
    if not all(col in unknown_pulses_df.columns for col in embedding_cols):
        print(" Skipping unknown processing: Dataframe missing vector embedding columns.")
        return unknown_pulses_df

    pulse_embeddings = unknown_pulses_df[embedding_cols].to_numpy().astype(np.float32)
    matched_indices = np.zeros(len(unknown_pulses_df), dtype=bool)
    # --- STEP 1: VECTORIZED SIMILARITY & BULK MATCH INSERTION ---
    if KNOWN_CENTERS_CACHE:
        signal_ids_list = list(KNOWN_CENTERS_CACHE.keys())
        centers_matrix = np.vstack([KNOWN_CENTERS_CACHE[sid] for sid in signal_ids_list])

        similarity_scores = np.dot(pulse_embeddings, centers_matrix.T)
        best_center_indices = np.argmax(similarity_scores, axis=1)
        best_similarity_values = np.max(similarity_scores, axis=1)
        threshold_mask = best_similarity_values >= UNKNOWN_DOT_PRODUCT_THRESHOLD
        if np.any(threshold_mask):
            matched_indices = threshold_mask
            print(f" Center Matching: Routing {np.sum(threshold_mask)} pulses to existing centers.")

            # Slice matched data frame and perform fast low-level NumPy index assignment
            matched_df = unknown_pulses_df[threshold_mask].copy()
            signal_ids_arr = np.array(signal_ids_list)
            matched_df['target_signal_id'] = signal_ids_arr[best_center_indices[threshold_mask]]

            # Vector-calculate summaries and bulk-insert everything at once
            matched_metrics = calculate_all_cluster_metrics_vectorized(matched_df)
            if matched_metrics:
                pg_matched_id_map = bulk_insert_clusters_to_postgres(pg_conn, matched_metrics)

                # Stream structural mappings directly to ClickHouse at once
                ch_matched_payload = []
                for sig_id, pg_id in pg_matched_id_map.items():
                    grp_mask = matched_df['target_signal_id'] == sig_id
                    uuids = matched_df.loc[grp_mask, 'id'].dropna().astype(str).unique().tolist()
                    if uuids:
                        ch_matched_payload.append([int(pg_id), uuids])

                if ch_matched_payload:
                    ch_client.insert(table='clusters', data=ch_matched_payload, column_names=['cluster_id', 'activity_ids'])
                    print(f" ClickHouse Logged: Saved {len(ch_matched_payload)} matched tracking clusters instantly.")

    # --- STEP 2: CLUSTERING FOR NEW PATTERN RECOGNITION ---
    remaining_pulses_df = unknown_pulses_df.iloc[~matched_indices].copy().reset_index(drop=True)

    if len(remaining_pulses_df) < max(2, MIN_CLUSTER_SIZE):
        print(f" Remaining queue ({len(remaining_pulses_df)} pulses) too small for clustering. Holding for backlog...")
        return remaining_pulses_df

    print(f" Running HDBSCAN Clustering on {len(remaining_pulses_df)} unmapped vectors...")
    leftover_embeddings = remaining_pulses_df[embedding_cols].to_numpy().astype(np.float32)

    cosine_similarity = np.dot(leftover_embeddings, leftover_embeddings.T)
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_CLUSTER_SIZE // 2, metric='cosine', copy=True)
    labels = clusterer.fit_predict(leftover_embeddings)
    unique_labels = np.unique(labels)
    print(f"Discovered labels array: {list(unique_labels)}")

    # Store standard unclustered noise elements immediately
    noise_row_indices = list(np.where(labels == -1)[0])
    start_idx = 0
    # Isolate newly discovered valid emitter clusters
    valid_discoveries_mask = labels != -1
    '''
    @vidya : Please add your cluster merge , scroing etc logic here 
    '''
    if np.any(valid_discoveries_mask):
        discovered_df = remaining_pulses_df[valid_discoveries_mask].copy()
        discovered_df['hdbscan_label'] = labels[valid_discoveries_mask]

        # Track if cache was modified to avoid redundant file operations
        cache_was_updated = False
        new_signal_records = []
        label_to_signal_id_map = {}
        # 1. Register new signals one by one to get new unique Signal IDs from Postgres
        # (This remains sequential since signal registration creates distinct new system rows)
        for label in unique_labels:
            if label == -1:
                continue
            lbl_mask = discovered_df['hdbscan_label'] == label
            new_cluster_df = discovered_df[lbl_mask].drop(columns=['hdbscan_label'])

            # Isolate local indices inside the active pool matching your template logic
            local_indices = np.where(labels == label)[0]
            global_indices = local_indices + start_idx
            num_elements = len(local_indices)

            # Isolate clean pairwise similarity space for this specific cluster block
            sub_matrix = cosine_similarity[local_indices[:, None], local_indices]

            if num_elements > 1:
                mean_sim = (np.sum(sub_matrix) - num_elements) / (num_elements * (num_elements - 1))

                # 1. Find the medoid's positional index *inside* this sub_matrix
                medoid_sub_idx = np.argmax(np.sum(sub_matrix, axis=1))
                medoid_local_idx = local_indices[medoid_sub_idx]

                # 2. Extract the medoid's row from sub_matrix and average it cleanly
                avg_medoid_sim = np.mean(sub_matrix[medoid_sub_idx].ravel())
                new_center_vector = leftover_embeddings[medoid_local_idx]
            else:
                mean_sim = 1.0
                avg_medoid_sim = 1.0
                medoid_local_idx = local_indices[0] if num_elements > 0 else 0
                new_center_vector = leftover_embeddings[medoid_local_idx] if num_elements > 0 else np.zeros(leftover_embeddings.shape[1])

            global_medoid_idx = medoid_local_idx + start_idx
            # for now only store emitter and signal only if similarity is very high
            # TODO :: enable this , once we have moduation and AOA @anurag
            #if(get_cluster_score(new_cluster_df,avg_medoid_sim) > some_threshlod)
            if(avg_medoid_sim > UNKNOWN_DOT_PRODUCT_THRESHOLD):
                # Execute transactional database insertion
                emitter_id = insert_emitter(pg_conn, "unknown")
                new_signal_id = insert_signal(pg_conn, new_cluster_df, emitter_id)

                if new_signal_id is not None:
                    label_to_signal_id_map[label] = int(new_signal_id)

                    # Format the output prints to match your metrics design exactly
                    label_name = f"Cluster {label:02d}"
                    print(f"     {label_name} | Postgres Signal ID: {new_signal_id} | Size: {num_elements:03d} pulses")
                    print(f"       Avg Internal Pairwise Sim: {mean_sim:.4f}")
                    print(f"       Avg Sim to Medoid Vector:  {avg_medoid_sim:.4f}")
                    print(f"       Chosen Medoid Global Index: {global_medoid_idx}")
                    print(f"       Indices: {global_indices.tolist()}\n")

                    # Push back securely into the volatile cache dictionary
                    KNOWN_CENTERS_CACHE[int(new_signal_id)] = new_center_vector
                    cache_was_updated = True


        # 2. Vector-map new tracking assignments using the generated keys
        if label_to_signal_id_map:
            discovered_df['target_signal_id'] = discovered_df['hdbscan_label'].map(label_to_signal_id_map)
            clean_discovered_df = discovered_df.drop(columns=['hdbscan_label'])

            # Vector-calculate metrics across all discovered blocks simultaneously
            discovered_metrics = calculate_all_cluster_metrics_vectorized(clean_discovered_df)

            if discovered_metrics:
                # Bulk-insert stats to Postgres
                pg_discovered_id_map = bulk_insert_clusters_to_postgres(pg_conn, discovered_metrics)

                # Bulk-compile raw pulse sequences for ClickHouse
                ch_discovered_payload = []
                for sig_id, pg_id in pg_discovered_id_map.items():
                    grp_mask = clean_discovered_df['target_signal_id'] == sig_id
                    uuids = clean_discovered_df.loc[grp_mask, 'id'].dropna().astype(str).unique().tolist()
                    if uuids:
                        ch_discovered_payload.append([int(pg_id), uuids])

                if ch_discovered_payload:
                    ch_client.insert(table='clusters', data=ch_discovered_payload, column_names=['cluster_id', 'activity_ids'])
                    print(f" ClickHouse Logged: Bulk-saved {len(ch_discovered_payload)} newly discovered clusters.")

        # 3. Safe File Serialization - Executed exactly ONCE outside the loops
        if cache_was_updated:
            with open(CENTERS_FILE_PATH, "w") as f:
                serializable_data = {int(k): v.tolist() for k, v in KNOWN_CENTERS_CACHE.items()}
                json.dump(serializable_data, f, indent=4)
            print(" Centers cache file successfully synchronized to disk.")

    # Isolate background noise or tracking gaps to pass back to tracking queue backlog
    noise_df = remaining_pulses_df.iloc[noise_row_indices].copy()
    print(f" HDBSCAN processing completed. {len(noise_df)} rows isolated as backlog noise.")
    return noise_df


def calculate_all_cluster_metrics_vectorized(matched_df: pd.DataFrame) -> list:
    """Calculates summary aggregations with custom IQR PRI filters per cluster."""
    if matched_df.empty:
        return []

    df = matched_df.copy()
    df['center_frequency_hz'] = pd.to_numeric(df['center_frequency_hz'], errors='coerce')
    df['power_dbm'] = pd.to_numeric(df['power_dbm'], errors='coerce')
    df['duration_us'] = pd.to_numeric(df['duration_us'], errors='coerce')
    df['bandwidth_hz'] = pd.to_numeric(df['bandwidth_hz'], errors='coerce')
    df['direction_of_arrival_deg'] = pd.to_numeric(df['direction_of_arrival_deg'], errors='coerce')
    df['time_of_arrival_ns'] = pd.to_datetime(df['time_of_arrival_ns'], errors='coerce').astype('int64')


    counts, edges = np.histogram(df['duration_us'], bins=100)
    mask = counts > 10
    filtered_edges = edges[:-1][mask]

    # 1. Extract static physical features using standard fast group aggregation
    summary_df = df.groupby('target_signal_id').agg(
        pdw_count=('id', 'count'),
        toa_ns=('time_of_arrival_ns', lambda x: int(np.min(x[x > 0])) if len(x[x > 0]) > 0 else 0),

        freq_mean=('center_frequency_hz', 'mean'),
        freq_min=('center_frequency_hz', 'min'),
        freq_max=('center_frequency_hz', 'max'),
        freq_std=('center_frequency_hz', lambda x: np.std(x, ddof=0)),

        power_mean=('power_dbm', 'mean'),
        power_min=('power_dbm', 'min'),
        power_max=('power_dbm', 'max'),
        power_std=('power_dbm', lambda x: np.std(x, ddof=0)),

        pw_mean=('duration_us', 'mean'),
        pw_min=('duration_us', 'min'),
        pw_max=('duration_us', 'max'),
        pw_std=('duration_us', lambda x: np.std(x, ddof=0)),

        bw_mean=('bandwidth_hz', 'mean'),
        bw_min=('bandwidth_hz', 'min'),
        bw_max=('bandwidth_hz', 'max'),
        bw_std=('bandwidth_hz', lambda x: np.std(x, ddof=0)),


        doa_mean=('direction_of_arrival_deg', 'mean'),
        doa_min=('direction_of_arrival_deg', 'min'),
        doa_max=('direction_of_arrival_deg', 'max'),
        doa_std=('direction_of_arrival_deg', lambda x: np.std(x, ddof=0)),


    ).fillna(0)

    # Extract clean outlier-free PRI stats using your custom IQR logic
    # This runs the logic natively per target_signal_id segment
    pri_stats = df.groupby('target_signal_id')['time_of_arrival_ns'].apply(calculate_iqr_pri_stats_per_group)

    # 3. Expand the tuple results into separate database-ready tracking columns
    summary_df['pri_mean'] = [t[0] for t in pri_stats]
    summary_df['pri_min']  = [t[1] for t in pri_stats]
    summary_df['pri_max']  = [t[2] for t in pri_stats]
    summary_df['pri_std']  = [t[3] for t in pri_stats]

    return summary_df.reset_index().to_dict(orient='records')




def calculate_iqr_pri_stats_per_group(toa_group_series: pd.Series) -> tuple:
    # 1. Compute delta values (nanoseconds to microseconds)
    pri_vals = toa_group_series.diff().to_numpy() / 1000.0
    valid_pri = pri_vals[~np.isnan(pri_vals)]

    if len(valid_pri) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 2. IQR outlier removal (removes completely wild values)
    if len(valid_pri) >= 4:
        q25, q75 = np.percentile(valid_pri, [25, 75])
        iqr = q75 - q25

        lower_bound = q25 - (1.5 * iqr)
        upper_bound = q75 + (1.5 * iqr)

        valid_pri = valid_pri[(valid_pri >= lower_bound) & (valid_pri <= upper_bound)]

    if len(valid_pri) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 3. Fixed Bin Histogram Peak Estimation
    try:
        bin_width = 3.0

        # Construct custom bin edges spaced exactly 3.0 microseconds apart
        custom_bins = np.arange(np.min(valid_pri), np.max(valid_pri) + bin_width, bin_width)

        # Fallback if the range is extremely small and yields fewer than 2 edges
        if len(custom_bins) < 2:
            custom_bins = np.array([np.min(valid_pri), np.min(valid_pri) + bin_width])

        counts, bin_edges = np.histogram(valid_pri, bins=custom_bins)

        # Locate the bin index with the highest count (the dominant PRI mode)
        max_bin_idx = np.argmax(counts)

        # Calculate the midpoint of that specific peak bin
        bin_start = bin_edges[max_bin_idx]
        bin_end = bin_edges[max_bin_idx + 1]
        dominant_pri = (bin_start + bin_end) / 2.0

    except (ValueError, TypeError):
        # Fallback if array distributions cause an edge case computation error
        dominant_pri = np.median(valid_pri)

    # 4. Aggregate clean statistics relative to the whole valid array
    return (
        float(dominant_pri),
        float(np.min(valid_pri)),
        float(np.max(valid_pri)),
        float(np.std(valid_pri, ddof=0))
    )



# def handle_known_emitters_vectorized(pulses_df: pd.DataFrame):

#     known_mask = (pulses_df['similarities'] >= SIMILARITY_THRESHOLD) & (pulses_df['matched_emitter_id'] > 1100)

#     # Isolate known matched profiles
#     if np.any(known_mask):
#         known_df = pulses_df[known_mask].copy()
#     else:
#         known_df = pd.DataFrame(columns=pulses_df.columns.tolist() + ['matched_emitter_id'])

#     # Isolate everything dropping through threshold checks as unknown leftovers
#     unknown_df = pulses_df[~known_mask].copy()

#     logger.LOG_INF(f"Matched {len(known_df)} pulses to known signal library; {len(unknown_df)} isolated as unknown signal.")
#     return known_df, unknown_df

# def handle_known_emitters_vectorized(emitter_ids: np.ndarray, similarities: np.ndarray, pulses_df: pd.DataFrame):

#     flat_similarities = np.squeeze(similarities)
#     flat_emitter_ids = np.squeeze(emitter_ids)

#     known_mask = (flat_emitter_ids >= 1117) & (flat_emitter_ids <= 1125)
#     flat_emitter_ids[known_mask] = 1117

#     known_mask = (flat_similarities >= SIMILARITY_THRESHOLD) & (flat_emitter_ids >= 1104)

#     # Isolate known matched profiles
#     if np.any(known_mask):
#         known_df = pulses_df[known_mask].copy()
#         known_df['matched_emitter_id'] = flat_emitter_ids[known_mask].astype(int)
#     else:
#         known_df = pd.DataFrame(columns=pulses_df.columns.tolist() + ['matched_emitter_id'])

#     # Isolate everything dropping through threshold checks as unknown leftovers
#     unknown_df = pulses_df[~known_mask].copy()

#     print(f" Matched {len(known_df)} pulses to library; {len(unknown_df)} isolated as unknown.")
#     return known_df, unknown_df

def handle_known_emitters_vectorized(pulses_df: pd.DataFrame):
    """
    Identifies known signals matching pre-existing library profiles vectorially.
    Returns a unified matched DataFrame and a separate slice DataFrame for unknowns.
    """

    global MEDOIDS
    if not MEDOIDS:
        load_medoids_cache()

    # Force arrays to flat dimensions to avoid orientation mismatches
    flat_similarities = pulses_df['similarities'].to_numpy().copy()
    flat_emitter_ids  = pulses_df['matched_emitter_id'].to_numpy().copy()

    known_mask = (flat_emitter_ids >= 1117) & (flat_emitter_ids <= 1137) & (pulses_df['center_frequency_hz'] > 9e9)
    flat_emitter_ids[known_mask] = 1117

    matched_ids = np.array([1104, 1105, 1106, 1107, 1108, 1116, 1113, 1115, 1109, 1117])

    known_mask = flat_similarities >= KNOWN_DOT_PRODUCT_THRESHOLD
    embedding_cols = [f"dim_{i}" for i in range(1024)]

    medoids_updated = False  # Track to save

    # Isolate known matched profiles
    if np.any(known_mask):
        unique_emitter_ids = np.unique(flat_emitter_ids[known_mask])

        for emitter_id in unique_emitter_ids:
            # Get embeddings for this specific emitter
            emitter_data = pulses_df[(flat_emitter_ids == emitter_id) & known_mask]
            mask = (np.array(flat_emitter_ids) == emitter_id) & known_mask
            emitter_embeddings = emitter_data[embedding_cols].values

            if (emitter_id in MEDOIDS):
                # Calculate similarity with old medoid
                old_medoid = MEDOIDS[emitter_id]
                sim_old = np.dot(emitter_embeddings, old_medoid.T).flatten()
                mean_sim_old = np.mean(sim_old)
            else:
                # Use similarities from data
                sim_old = flat_similarities[mask]
                mean_sim_old = np.mean(sim_old) if len(sim_old) > 0 else 0

            # Find best candidate for new medoid
            max_inner_similarities = np.sum(np.dot(emitter_embeddings, emitter_embeddings.T),axis=1)
            idx = np.argmax(max_inner_similarities)

            # Calculate similarity with selected vector
            centroid_sim_new = emitter_embeddings[idx]
            sim_new = np.dot(emitter_embeddings, centroid_sim_new.T)
            mean_sim_new = np.mean(sim_new)

            # Decide whether to update medoid
            if ((mean_sim_old < mean_sim_new) & (emitter_embeddings.shape[0] > 30)):
                # Add/update medoid
                MEDOIDS[emitter_id] = centroid_sim_new
                flat_similarities[mask] = sim_new
                medoids_updated = True
            else:
                flat_similarities[mask] = sim_old

        is_1117 = (flat_emitter_ids == 1117)
        threshold = np.where(is_1117, SIMILARITY_THRESHOLD_1117, SIMILARITY_THRESHOLD)
        known_mask = (flat_similarities >= threshold) & np.isin(flat_emitter_ids, matched_ids)
        known_df = pulses_df[known_mask].copy()
        known_df['matched_emitter_id'] = flat_emitter_ids[known_mask].astype(int)
    else:
        known_df = pd.DataFrame(columns=pulses_df.columns.tolist() + ['matched_emitter_id'])

    # Isolate everything dropping through threshold checks as unknown leftovers
    unknown_df = pulses_df[~known_mask].copy()

    if medoids_updated:
        save_medoids_cache()

    print(f" Matched {len(known_df)} pulses to library; {len(unknown_df)} isolated as unknown.")
    return known_df, unknown_df

# storage for emitter level clusters ,
known_emitter_cache = {}
def batch_resolve_signal_ids(pg_conn, known_df: pd.DataFrame) -> dict:
    """Resolves signal registrations for active emitters. Registers new rows if missing."""
    unique_emitters = known_df['matched_emitter_id'].unique().tolist()
    emitter_to_signal_map = {}

    for e_id in unique_emitters:
        emitter_key = int(e_id)
        # Check high-speed array cache memory layers
        signal_id = get_signal_id_for_emitter(emitter_key)
        angle,_ = get_cluster_angle_score(known_df)

        # If library signal entry hasn't been registered in database tracking registries yet
        if (signal_id is None):
            # Extract the raw feature snapshot slice to populate the master tracking row template
            emitter_slice = known_df[known_df['matched_emitter_id'] == emitter_key]
            if (len(emitter_slice) > KNOWN_MIN_SIGNAL_SIZE):
                emitter_id = insert_emitter(pg_conn,e_id)
                signal_id = insert_signal(pg_conn, emitter_slice,emitter_id,angle)
                known_emitter_cache[signal_id] = [[], []]
                known_emitter_data = known_emitter_cache[signal_id]

                known_emitter_data[0].extend(known_df['power_dbm'].tolist())
                known_emitter_data[1].extend(known_df['direction_of_arrival_deg'].tolist())

                if signal_id is not None:
                    track_signal_id_for_emitter(emitter_key, signal_id)
        else :
            known_emitter_data = known_emitter_cache[signal_id]

            known_emitter_data[0].extend(known_df['power_dbm'].tolist())
            known_emitter_data[1].extend(known_df['direction_of_arrival_deg'].tolist())

            angle = get_cluster_angle(
                np.array(known_emitter_data[1]),
                np.array(known_emitter_data[0])
            )
            # update duration in signal table
            update_existing_signal(pg_conn, signal_id)
            update_doa_in_signal(pg_conn, signal_id, angle)
        if signal_id is not None:
            emitter_to_signal_map[emitter_key] = int(signal_id)

    return emitter_to_signal_map
# def batch_resolve_signal_ids(pg_conn, known_df: pd.DataFrame) -> dict:
#     """Resolves signal registrations for active emitters. Registers new rows if missing."""
#     unique_emitters = known_df['matched_emitter_id'].unique().tolist()
#     emitter_to_signal_map = {}

#     clusters, _ = cluster_based_on_embedding_similarities(known_df.copy(), 0.8, MEDOIDS, 0)
#     merged_clusters = merge_clusters(clusters.copy())

#     for idx, merged_cluster in enumerate(merged_clusters):
#         cluster_df = merged_cluster[2]  # DataFrame for this merged cluster

#         # Pick a representative emitter id for the registry key
#         # (adjust if your cluster elements have a different layout)
#         e_id = int(cluster_df['matched_emitter_id'].mode().iloc[0])
#         emitter_key = int(e_id)

#         # Check high-speed array cache memory layers
#         signal_id = get_signal_id_for_emitter(emitter_key)
#         angle, _ = get_cluster_angle_score(cluster_df)

#         if signal_id is None:
#             # Use the merged cluster's own rows (already filtered by 0.8 similarity)
#             emitter_slice = cluster_df
#             if len(emitter_slice) > KNOWN_MIN_SIGNAL_SIZE:
#                 emitter_id = insert_emitter(pg_conn, e_id)
#                 signal_id = insert_signal(pg_conn, emitter_slice, emitter_id, angle)
#                 known_emitter_cache[signal_id] = [[], []]
#                 known_emitter_data = known_emitter_cache[signal_id]

#                 known_emitter_data[0].extend(cluster_df['power_dbm'].tolist())
#                 known_emitter_data[1].extend(cluster_df['direction_of_arrival_deg'].tolist())

#                 if signal_id is not None:
#                     track_signal_id_for_emitter(emitter_key, signal_id)
#         else:
#             known_emitter_data = known_emitter_cache[signal_id]

#             known_emitter_data[0].extend(cluster_df['power_dbm'].tolist())
#             known_emitter_data[1].extend(cluster_df['direction_of_arrival_deg'].tolist())

#             angle = get_cluster_angle(
#                 np.array(known_emitter_data[1]),
#                 np.array(known_emitter_data[0]),
#             )
#             print(f"Angle ****************** {angle}")
#             update_existing_signal(pg_conn, signal_id)
#             update_doa_in_signal(pg_conn, signal_id, angle)

#         if signal_id is not None:
#             emitter_to_signal_map[emitter_key] = int(signal_id)

#     return emitter_to_signal_map


# Define the global buffer and threshold at the top of your script
PULSE_BUFFER = []


def handle_unknown_pulses_buffered(incoming_df: pd.DataFrame, pg_conn, ch_client) -> pd.DataFrame:
    global PULSE_BUFFER

    if incoming_df.empty:
        return pd.DataFrame()

    # 1. Store the incoming dataframe directly into the global list
    PULSE_BUFFER.append(incoming_df)

    pulses = pd.concat(PULSE_BUFFER, ignore_index=True)
    pulses['time_of_arrival_ns'] = pd.to_datetime(pulses['time_of_arrival_ns'])
    max_time = pulses['time_of_arrival_ns'].max()
    full_batch_df = pulses[pulses['time_of_arrival_ns'] >= max_time - pd.Timedelta(seconds=UNKNOWN_ACCUMULATION_TIME_SEC)]

    # Calculate current total rows across all pieces in the buffer
    current_total_rows = sum(len(df) for df in PULSE_BUFFER)

    # 2. Run the process only when we have enough pulses
    if current_total_rows >= BUFFER_THRESHOLD:
        logger.LOG_INF(f"Minimum Pulses ({current_total_rows}/{BUFFER_THRESHOLD} pulses) available to process Unknown Emitters . Running process...")

        # Execute your main processing engine
        handle_unknown_pulses(full_batch_df, pg_conn, ch_client,KNOWN_CENTERS_CACHE)


        PULSE_BUFFER = []



    else:
        logger.LOG_INF(f"Buffered {len(incoming_df)} pulses. Current unknown buffer size: {current_total_rows}/{BUFFER_THRESHOLD}")
        return pd.DataFrame()

