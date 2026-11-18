import numpy as np
import pandas as pd
import os
from sklearn.cluster import HDBSCAN
from custom_clustering import simple_clustering
import logger

from cluster_engine import (
    ClusterEngineState,
    calculate_medoid_and_similarity,
    group_similar_clusters
)
from cluster_refinement import get_cluster_score
from database import (
    insert_signal,
    bulk_insert_clusters_to_postgres,
    insert_emitter,
    update_existing_signal
)

UNKNOWN_DOT_PRODUCT_THRESHOLD = 0.80   # Minimum similarity for individual pulse cache hits
CLUSTER_MERGE_THRESHOLD = 0.7         # Minimum similarity for combining separate HDBSCAN outputs
MIN_SCORE_THRESHOLD = 0.2             # Gate threshold for your custom scoring function
MIN_CLUSTER_SIZE = 15                  # Minimum density limit for HDBSCAN groups


# Configure the system to only run global merging once every 10 streaming loops
state_manager = ClusterEngineState(max_exemplars_per_signal=20, window_batch_limit=5)

import numpy as np
import pandas as pd
from typing import Dict
from cluster_engine import EmitterTrack

def _process_known_vector_matches(
    unknown_pulses_df: pd.DataFrame,
    pulse_embeddings: np.ndarray,
    pg_conn,
    ch_client,
    known_centers_cache: Dict[int, EmitterTrack]
) -> np.ndarray:
    """Step 1: Maps incoming pulses to pre-saved single medoid centers using a NamedTuple metadata schema."""
    matched_indices = np.zeros(len(unknown_pulses_df), dtype=bool)
    if not known_centers_cache:
        return matched_indices

    # 1. Gather keys and stack single medoid vectors directly into a matrix using NamedTuple layout
    signal_ids_list = list(known_centers_cache.keys())
    centers_matrix = np.vstack([known_centers_cache[sid].vector for sid in signal_ids_list])

    # 2. Compute similarity (Since inputs are pre-normalized, dot product is exactly cosine similarity)
    # Dimensions: (num_incoming_pulses, num_known_signals)
    similarity_scores = np.dot(pulse_embeddings, centers_matrix.T)

    # 3. Fast extraction of the absolute best-matching signal index per row
    best_center_indices = np.argmax(similarity_scores, axis=1)
    best_similarity_values = np.max(similarity_scores, axis=1)

    # 4. Filter matches based on multi-dimensional tracking gates (Vector, Angle, Modulation)
    ALLOWED_ANGLE_DELTA = 40  # Allow +/- 20 degrees tolerance variance limit

    threshold_mask = np.zeros(len(unknown_pulses_df), dtype=bool)

    for i in range(len(unknown_pulses_df)):
        # Gateway 1: Check baseline math vector similarity boundary
        if best_similarity_values[i] >= UNKNOWN_DOT_PRODUCT_THRESHOLD:
            winning_sid = signal_ids_list[best_center_indices[i]]
            cached_track = known_centers_cache[winning_sid]

            # Gateway 2: Angle of Arrival (AoA) verification
            # Extract raw individual incoming pulse angle metadata safely
            pulse_angle = float(unknown_pulses_df.loc[i, 'direction_of_arrival_deg']) if 'direction_of_arrival_deg' in unknown_pulses_df.columns else None

            if pulse_angle is not None:
                #TODO:: Anurag check for 360 as well , rounding will be needed here
                angle_delta = abs(pulse_angle - cached_track.angle)
                # If the signal drifts past our angle limits, drop the vector match match
                if angle_delta > ALLOWED_ANGLE_DELTA:
                    continue

            # Gateway 3: Modulation Type cross-validation boundary checks
            # Match individual raw incoming modulation parameters against historical trends
            #TODO:: use below onl;y after testing results from field Anurag
            #pulse_mod = str(unknown_pulses_df.loc[i, 'Modulation']) if 'Modulation' in unknown_pulses_df.columns else None

            # If all validation checkpoints pass successfully, authorize the routing index
            threshold_mask[i] = True

    if np.any(threshold_mask):
        # Inline local import to safely break the circular import loop inside Docker
        from cluster_handling import calculate_all_cluster_metrics_vectorized

        matched_indices = threshold_mask
        print(f" Center Matching: Routing {np.sum(threshold_mask)} pulses to existing centers.")

        matched_df = unknown_pulses_df[threshold_mask].copy()
        signal_ids_arr = np.array(signal_ids_list)

        # Route points directly to the winning signal id
        matched_df['target_signal_id'] = signal_ids_arr[best_center_indices[threshold_mask]]
        #print(f"matched signal id {matched_df['target_signal_id'].tolist()}")

        # Handle persistent database sync sequences
        matched_metrics = calculate_all_cluster_metrics_vectorized(matched_df)
        if matched_metrics:
            pg_matched_id_map = bulk_insert_clusters_to_postgres(pg_conn, matched_metrics)

            ch_matched_payload = []
            for sig_id, pg_id in pg_matched_id_map.items():
                # Update parent relational tables in Postgres
                update_existing_signal(pg_conn, sig_id)

                grp_mask = matched_df['target_signal_id'] == sig_id
                uuids = matched_df.loc[grp_mask, 'id'].dropna().astype(str).unique().tolist()
                if uuids:
                    ch_matched_payload.append([int(pg_id), uuids])

            if ch_matched_payload:
                ch_client.insert(table='clusters', data=ch_matched_payload, column_names=['cluster_id', 'activity_ids'])
                print(f" ClickHouse Logged: Saved {len(ch_matched_payload)} matched tracking clusters instantly.")

    return matched_indices



def _process_known_vector_matches_old_working(unknown_pulses_df: pd.DataFrame, pulse_embeddings: np.ndarray, pg_conn, ch_client, known_centers_cache: dict) -> np.ndarray:
    """Step 1: Maps incoming pulses to single pre-saved medoid centers via flat matrix multiplication."""
    matched_indices = np.zeros(len(unknown_pulses_df), dtype=bool)
    if not known_centers_cache:
        return matched_indices

    # 1. Gather keys and stack single medoid vectors directly into a matrix
    signal_ids_list = list(known_centers_cache.keys())
    centers_matrix = np.vstack([known_centers_cache[sid] for sid in signal_ids_list])

    # 2. Compute similarity (Since inputs are pre-normalized, dot product is exactly cosine similarity)
    # Dimensions: (num_incoming_pulses, num_known_signals)
    similarity_scores = np.dot(pulse_embeddings, centers_matrix.T)

    # 3. Fast extraction of the absolute best-matching signal index per row
    best_center_indices = np.argmax(similarity_scores, axis=1)
    best_similarity_values = np.max(similarity_scores, axis=1)

    # 4. Filter matches based on your threshold rule
    threshold_mask = best_similarity_values >= UNKNOWN_DOT_PRODUCT_THRESHOLD
    #TODO:: Anurag : adding of pulses should be based on not only similarity but also angle and modulation etc

    if np.any(threshold_mask):
        # Inline local import to safely break the circular import loop inside Docker
        from cluster_handling import calculate_all_cluster_metrics_vectorized

        matched_indices = threshold_mask
        print(f" Center Matching: Routing {np.sum(threshold_mask)} pulses to existing centers.")

        matched_df = unknown_pulses_df[threshold_mask].copy()
        signal_ids_arr = np.array(signal_ids_list)

        # Route points directly to the winning signal id
        matched_df['target_signal_id'] = signal_ids_arr[best_center_indices[threshold_mask]]
        print(f"matehded signal id {matched_df['target_signal_id']}")
        # Handle persistent database sync sequences
        matched_metrics = calculate_all_cluster_metrics_vectorized(matched_df)
        if matched_metrics:
            pg_matched_id_map = bulk_insert_clusters_to_postgres(pg_conn, matched_metrics)

            ch_matched_payload = []
            for sig_id, pg_id in pg_matched_id_map.items():
                #update signal table
                update_existing_signal(pg_conn, sig_id)
                grp_mask = matched_df['target_signal_id'] == sig_id
                uuids = matched_df.loc[grp_mask, 'id'].dropna().astype(str).unique().tolist()
                if uuids:
                    ch_matched_payload.append([int(pg_id), uuids])

            if ch_matched_payload:
                ch_client.insert(table='clusters', data=ch_matched_payload, column_names=['cluster_id', 'activity_ids'])
                print(f" ClickHouse Logged: Saved {len(ch_matched_payload)} matched tracking clusters instantly.")

    return matched_indices



def handle_unknown_pulses_working_hdbscan(unknown_pulses_df: pd.DataFrame, pg_conn, ch_client,known_centers_cache: Dict[int, EmitterTrack]) -> pd.DataFrame:

    if unknown_pulses_df.empty:
        return unknown_pulses_df

    embedding_cols = [f"dim_{i}" for i in range(1024)]
    if not all(col in unknown_pulses_df.columns for col in embedding_cols):
        print(" Skipping unknown processing: Dataframe missing vector embedding columns.")
        return unknown_pulses_df

    pulse_embeddings = unknown_pulses_df[embedding_cols].to_numpy().astype(np.float32)

    # --- STEP 1: ROUTE TO EXISTING TRACKS AND EMITTERS ---
    matched_indices = _process_known_vector_matches(unknown_pulses_df, pulse_embeddings, pg_conn, ch_client,known_centers_cache)

    # --- STEP 2: CLUSTERING FOR NEW PATTERN RECOGNITION ---
    remaining_pulses_df = unknown_pulses_df.iloc[~matched_indices].copy().reset_index(drop=True)
    #remaining_pulses_df = unknown_pulses_df
    # List to catch un-actioned elements to return at the end of the micro-loop
    current_batch_triage_list = []

    if len(remaining_pulses_df) >= max(2, MIN_CLUSTER_SIZE):
        print(f" Running HDBSCAN Clustering on {len(remaining_pulses_df)} unmapped vectors...")
        leftover_embeddings = remaining_pulses_df[embedding_cols].to_numpy().astype(np.float32)
        cosine_similarity = np.dot(leftover_embeddings, leftover_embeddings.T)

        clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_CLUSTER_SIZE // 2, metric='cosine', copy=True)
        local_labels = clusterer.fit_predict(leftover_embeddings)
        global_labels = state_manager.shift_and_update_labels(local_labels)
        unique_global_labels = np.unique(global_labels)

        # Isolate noise from this batch immediately
        noise_mask = global_labels == -1
        current_batch_triage_list.append(remaining_pulses_df[noise_mask])

        # Stage valid discoveries directly into memory buffers
        for g_label in unique_global_labels:
            if g_label == -1:
                continue
            local_indices = np.where(global_labels == g_label)[0]
            lbl_mask = global_labels == g_label

            mean_sim, avg_medoid_sim, center_vector = calculate_medoid_and_similarity(
                leftover_embeddings, local_indices, cosine_similarity)

            print(f"cluster avg similirarity with mediod {avg_medoid_sim}")


            # --- PUSH DIRECTLY INTO TRANSIENT WINDOW STORAGE ---
            state_manager.buffer_discovered_cluster(
                global_label=g_label,
                df=remaining_pulses_df[lbl_mask].copy(),
                medoid_vector=center_vector,
                local_indices=local_indices
            )
    else:
        # Batch too small to cluster; add everything directly to backlog return
        current_batch_triage_list.append(remaining_pulses_df)

    # --- STEP 3: PERIODIC FLUSH TRIPPED BY WINDOW STATE LIMIT ---
    if not state_manager.should_trigger_merge():
        print(f" Buffer Window State Accumulating ({state_manager.batch_counter}/{state_manager.window_batch_limit}). Bypassing global merge loop.")
        return pd.concat(current_batch_triage_list, ignore_index=True)

    # If code reaches here, the window is full! Time to merge everything in storage.
    print(f" [!] Window limit reached. Running Agglomerative Merge Matrix across {len(state_manager.transient_cluster_buffer)} staged sub-clusters...")

    # Extract medoids from all accumulated clusters in the buffer
    temp_cluster_centers = {c['global_label']: c['medoid'] for c in state_manager.transient_cluster_buffer}
    # TODO:: @Vidya , we can put logic to split each clusters based on AOA and modualtion before moving to split

    # Run global centroid cross-similarity merging
    merged_groups = group_similar_clusters(temp_cluster_centers, CLUSTER_MERGE_THRESHOLD)
    count =0
    # Process each grouped component family
    for root_parent, child_labels in merged_groups.items():
        # Filter buffer components belonging to this family tree
        family_components = [c for c in state_manager.transient_cluster_buffer if c['global_label'] in child_labels]

        combined_df = pd.concat([c['df'] for c in family_components], ignore_index=True)

        # We need a quick way to recompute the unified medoid across the combined embeddings
        family_embeddings = combined_df[embedding_cols].to_numpy().astype(np.float32)
        family_similarity = np.dot(family_embeddings, family_embeddings.T)
        dummy_indices = np.arange(len(family_embeddings))

        mean_sim, avg_medoid_sim, unified_center_vector = calculate_medoid_and_similarity(
            family_embeddings, dummy_indices, family_similarity
        )

        # Run custom scoring gate
        cluster_score,angle,modulation = get_cluster_score(combined_df, avg_medoid_sim)

        if cluster_score >= MIN_SCORE_THRESHOLD:
            new_signal_id, _ = create_instert_new_signal_emitter(combined_df,pg_conn, ch_client,angle)
            if new_signal_id is not None:
                print(f"  [+] Unified Medoid Cluster Certified! Signal {new_signal_id} Saved.")

                known_centers_cache[int(new_signal_id)] = EmitterTrack(
                    vector=unified_center_vector,
                    angle=angle,
                    modulation=modulation)

        else:
            print(f"  [-] Window Flash Triage. Low score cluster quarantined. Score: {cluster_score:.4f} | Mass: {len(combined_df)} pulses.")
            current_batch_triage_list.append(combined_df)
        count = count+1

    #matched_indices = _process_known_vector_matches(unknown_pulses_df, pulse_embeddings, pg_conn, ch_client,known_centers_cache)
    # Wipe transient storage clean for the next operational window

    # Wipe transient storage completely clean for the next window accumulation period
    state_manager.clear_transient_buffer()

    try:
        # Import dynamically if necessary to prevent file loops
        from cluster_handling import save_cluster_centers_cache
        save_cluster_centers_cache()
    except Exception as e:
        print(f"Warning: Async window disk sync skipped ({e}).")

    # Returns unclustered noise + low-confidence clusters as one unified triage dataframe
    return pd.concat(current_batch_triage_list, ignore_index=True)



def create_instert_new_signal_emitter(df:pd.DataFrame,pg_conn, ch_client,angle,is_existing_signal_present,new_signal_id):

    if(is_existing_signal_present == False):
        emitter_id = insert_emitter(pg_conn, "unknown_ai")
        new_signal_id =  insert_signal(pg_conn, df, emitter_id,angle)
        print(f"New emitter id{emitter_id}")
        print(f"New signal id{new_signal_id}")

    df['target_signal_id'] = new_signal_id
    from cluster_handling import calculate_all_cluster_metrics_vectorized

    # Handle persistent database sync sequences
    matched_metrics = calculate_all_cluster_metrics_vectorized(df)

    if matched_metrics:
        pg_matched_id_map = bulk_insert_clusters_to_postgres(pg_conn, matched_metrics)

        ch_matched_payload = []
        for sig_id, pg_id in pg_matched_id_map.items():
            update_existing_signal(pg_conn, sig_id)
            grp_mask = df['target_signal_id'] == sig_id
            uuids = df.loc[grp_mask, 'id'].dropna().astype(str).unique().tolist()
            if uuids:
                ch_matched_payload.append([int(pg_id), uuids])

        if ch_matched_payload:
            ch_client.insert(table='clusters', data=ch_matched_payload, column_names=['cluster_id', 'activity_ids'])
            logger.LOG_VRB(f" ClickHouse Logged: Saved {len(ch_matched_payload)} matched tracking clusters instantly.")

    return new_signal_id, matched_metrics



def handle_unknown_pulses(unknown_pulses_df: pd.DataFrame, pg_conn, ch_client,known_centers_cache: Dict[int, EmitterTrack]) -> pd.DataFrame:

    if unknown_pulses_df.empty:
        return unknown_pulses_df
    # print(f"handle_unknown_pulses : {len(unknown_pulses_df)}")
    embedding_cols = [f"dim_{i}" for i in range(1024)]
    if not all(col in unknown_pulses_df.columns for col in embedding_cols):
        print(" Skipping unknown processing: Dataframe missing vector embedding columns.")
        return unknown_pulses_df

    pulse_embeddings = unknown_pulses_df[embedding_cols].to_numpy().astype(np.float32)

    # --- STEP 1: ROUTE TO EXISTING TRACKS AND EMITTERS ---
    #matched_indices = _process_known_vector_matches(unknown_pulses_df, pulse_embeddings, pg_conn, ch_client,known_centers_cache)

    # --- STEP 2: CLUSTERING FOR NEW PATTERN RECOGNITION ---
    #remaining_pulses_df = unknown_pulses_df.iloc[~matched_indices].copy().reset_index(drop=True)


    remaining_pulses_df = unknown_pulses_df
    # Step 3: run clustering and merge  on remaining pulses
    simple_clustering(remaining_pulses_df, known_centers_cache, pg_conn, ch_client)


