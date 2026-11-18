
import pandas as pd
import numpy as np
from typing import Dict
import logging
log = logging.getLogger(__name__)  
import config
import plot_data
from cluster_refinement import get_cluster_similarity_score, get_cluster_angle_score, get_cluster_score,get_cluster_angle
from collections import namedtuple
from collections import defaultdict
from config import UNKNOWN_MIN_SIGNAL_SIZE
import re
import os
import time

from cluster_engine import EmitterTrack
from database import (
    insert_signal, 
    insert_cluster_to_postgres, 
    add_pdws_to_clusters_table_ch,
    bulk_insert_clusters_to_postgres,
    insert_emitter,
    update_existing_signal,
    update_doa_in_signal
)
UNKNOWN_DOT_PRODUCT_THRESHOLD = 0.50
MIN_CLUSTER_SIZE = 100

def print_truth_emitter_stats(csv_filepath):
    folder_name = os.path.basename(os.path.dirname(csv_filepath))
    txt_path = os.path.join(os.path.dirname(csv_filepath), f"{folder_name}_params.txt")
    if not os.path.exists(txt_path): return

    emitters = []
    current_id = None
    current_params = defaultdict(list)
    param_order = []

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m_id = re.search(r'Emitter\s+(\d+)', line)
            if m_id:
                if current_id is not None:
                    emitters.append((current_id, current_params, param_order[:]))
                current_id = m_id.group(1)
                current_params = defaultdict(list)
                param_order = []
                continue
            m_val = re.match(r'(.+?)\s*=\s*([\d.]+)', line)
            if m_val and current_id:
                k, v = m_val.group(1).strip(), float(m_val.group(2))
                if k not in param_order: param_order.append(k)
                current_params[k].append(v)
    
    if current_id is not None:
        emitters.append((current_id, current_params, param_order))

    if not emitters: return

    # Union of all params in order of first appearance
    all_params = []
    for _, _, p_order in emitters:
        for p in p_order:
            if p not in all_params: all_params.append(p)

    # Header
    COL_W = 18
    header = f"{'Truth eID':<12}"
    for p in all_params:
        header += f"| {p:<{COL_W}}"

    print(" ")
    log.info(f"{'~'*len(header)}")
    log.info(header)
    log.info("~" * len(header))

    # Rows
    for eid, params, _ in emitters:
        row = f"{eid:<12}"
        for p in all_params:
            if p in params and params[p]:
                mn = min(params[p])
                mx = max(params[p])
                # Format: "3.00, 3.17" (No colon at end)
                val_str = f"{mn:.2f}, {mx:.2f}"
            else:
                val_str = "N/A"
            row += f"| {val_str:<{COL_W}}"
        log.info(row)
    log.info("-" * len(header))



def print_cluster_summary_header( w_id=12, w_alist=35, w_list=25, w_rows=6, w_angle=10, w_score=8, w_mod=12, w_result=6) :
    sep_line = f"{'-'*w_id} | {'-'*w_alist} | {'-'*w_list} | {'-'*w_angle} | {'-'*w_mod} | {'-'*w_score} | {'-'*w_result}"
    log.info(sep_line)
    log.info(f"{' Cid':>{w_id}} | {'Truth-eIDs':<{w_alist}} | {'Matched-eIDs':<{w_list}} | {'Angle':>{w_angle}} | {'Modulation':>{w_mod}} | {'Score':>{w_score}} | {'Result':>{w_result}}")
    log.info(sep_line)
    return sep_line
     

def print_cluster_summary_table(clusters):
    """
    Prints the summary information for each of the cluster.
    """
    w_id, w_alist, w_list, w_rows, w_angle, w_mod, w_score, w_result = 12, 50,32, 5, 9, 11, 8, 6
    sep_line = print_cluster_summary_header(w_id, w_alist, w_list, w_rows, w_angle, w_score, w_mod, w_result) 

    all_aeids = set()
    processed_count = 0  
    for item in clusters:
        mc_id = item[0]
        medoid = item[1]
        df_m = item[2]
        
        angle = item[3] if len(item) > 4 else 0
        modulation = item[4] if len(item) > 5 else 0
        score = item[5] if len(item) > 3 else 0
        
        if df_m.empty:
            return

        processed_count += 1

        aeid_list = sorted(df_m["emitter_id"].unique().tolist())
        meid_list = sorted(df_m["matched_emitter_id"].unique().tolist())
        all_aeids.update(aeid_list)
        
        mapping_parts = []
        eid_to_ids = {}  
        for eid in aeid_list:
            sub = df_m[df_m["emitter_id"] == eid]
            count = len(sub)
            mids = sorted(sub["matched_emitter_id"].unique().tolist())
            mapping_parts.append(f"({count}) {eid}->{mids}")
            
            # Store actual IDs for this emitter
            eid_to_ids[eid] = sub["id"].tolist()
        
        actual_mapping_str = ", ".join(mapping_parts)
        mids_str = str(meid_list)
        row_count = len(df_m)
        
        result = "✅" if len(aeid_list) == 1 else "⚠️"
        
        indices_str = ", ".join(map(str, [mc_id]))
        id_count_str = f"({row_count}) {indices_str}"

        # Columns: ( Rows) ID | Actual Mapping | Matched IDs | Angle | Score | Mod | Result
        log.info(f"{id_count_str:>{w_id}} | {actual_mapping_str:<{w_alist}} | {mids_str:<{w_list}} | {angle:>{w_angle}.1f}°| {modulation:>{w_mod}} | {score:>{w_score}.2f} | {result:<{w_result}}")

        # for eid, ids in eid_to_ids.items():
        #     if eid == 2:
        #         logger.LOG_INF(f"\t\t\tEmitter {eid} IDs: {sorted(ids, key=int)}")
    log.info(f"{sep_line}")

    if list(all_aeids) != [0]:
        print("\n")
        log.info(f"{'='*100}")
        log.warning(f"Total unique {len(all_aeids)} Ground Truth Emitters across {processed_count} unknown merged clusters")
        log.info(f"{'='*100}")
        print("\n")

UnmergedCluster = namedtuple('UnmergedCluster', [
    'cid', 'medoid', 'df', 'angle', 'modulation',
    'ssim', 'ascore', 'mscore'
])


# storage for emitter level clusters , 
emitter_cache = {}


def cluster_based_on_matching_emitter_id(df: pd.DataFrame, start_cluster_id: int = 0):
    """
    Cluster rows in `df` by their `matched_emitter_id`.
    Returns a list of (cluster_id, representative_row, cluster_df) tuples.
    """
    if df.empty or "matched_emitter_id" not in df.columns:
        log.warning(" Skipping emitter-id clustering: missing 'matched_emitter_id' column or empty dataframe.")
        return [], start_cluster_id

    clusters = []
    current_id = start_cluster_id

    for emitter_id, group in df.groupby("matched_emitter_id", sort=False):
        log.verbose(f"Cluster {current_id} (emitter_id {emitter_id}): "
                       f"count={len(group)}, indices={group.index.tolist()}")

        # Create cluster dataframe with the new ID
        sub_df = group.copy()
        sub_df["cluster_id"] = current_id
        
        medoid, similarities, score = get_cluster_similarity_score(sub_df)
        
        clusters.append((current_id, medoid, sub_df))
        current_id += 1
        # mask = pd.Series(True, index=df.index) if not df.empty else pd.Series(False, index=df.index)

    return clusters, current_id

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def cluster_based_on_embedding_similarities(
    df: pd.DataFrame,
    threshold: float,
    known_centers_cache=None,
    start_cluster_id: int = 0,
):
    """
    Optimized and robust greedy clustering with explicit 2D matrix slicing logic.
    """
    embedding_cols = [f"dim_{i}" for i in range(1024)]

    if not all(col in df.columns for col in embedding_cols):
        print(" Skipping unknown processing: Dataframe missing vector embedding columns.")
        return df, start_cluster_id

    X = df[embedding_cols].to_numpy().astype(np.float32)
    n_samples = X.shape[0]

    # Precompute similarities once outside the loop
    all_similarities = np.dot(X, X.T)

    clusters = []
    labels = np.full(n_samples, -1, dtype=int)
    current_cluster_id = start_cluster_id

    for i in range(n_samples):
        if labels[i] != -1:
            continue

        sim_row = all_similarities[i]
        matching_mask = (sim_row >= threshold) & (labels == -1)
        
        # FIX 1: Explicitly grab the internal 1D array ([0]) from np.where's tuple output
        matching_indices = np.where(matching_mask)[0]

        if len(matching_indices) >= MIN_CLUSTER_SIZE:
            print(len(matching_indices))

            #logger.LOG_VRB(f"Cluster {current_cluster_id} (Leader index {i}):")
            #logger.LOG_VRB(f"  -> Bundled indices   : {matching_indices.tolist()}")
            #logger.LOG_VRB(f"  -> Median Similarity : {median_sim:.4f}\n")
            
            labels[matching_mask] = current_cluster_id
            medoid_vector = X[matching_indices].mean(axis=0)

            
            sub_df_slice = df.iloc[matching_indices]
            _, angle_score = get_cluster_angle_score(sub_df_slice)
            
            if angle_score > 0.3:
                sub_df = sub_df_slice.copy()
                sub_df["cluster_id"] = current_cluster_id
                clusters.append((current_cluster_id, medoid_vector, sub_df))
                current_cluster_id += 1
    return clusters, current_cluster_id



def simple_clustering(df: pd.DataFrame, known_centers_cache: Dict[int, EmitterTrack], pg_conn, ch_client, threshold=0.8):

    print("\n.... simple_clustering ....")

    # ---------------------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------------------
    if 1: # make it 0 to compare with AI given matched emitter id based clustering
        start = time.perf_counter()
        clusters, _  = cluster_based_on_embedding_similarities(df.copy(), threshold, known_centers_cache, 0)
        print(f"cluster_based_on_embedding_similarities time { (time.perf_counter() - start) * 1000 }") 
        start = time.perf_counter()
        merged_clusters = merge_clusters(clusters.copy())
        print(f"merged_clusters time { (time.perf_counter() - start) * 1000 }") 
    else:
        clusters, next_cluster_id = cluster_based_on_matching_emitter_id(df.copy(), 0)   
        merged_clusters = merge_clusters(clusters.copy()) 

    if not merged_clusters:
        return
    
    if config.DEBUG_UN_INFO:
        print_truth_emitter_stats(config.UNKNOWN_CLUSTER_FILENAME)
        plot_data.save_filtered_clusters(clusters, config.unknown_cluster_csv_desired_order, config.UNKNOWN_CLUSTER_FILENAME, "unmerged_")
        plot_data.save_filtered_clusters(merged_clusters, config.unknown_cluster_csv_desired_order, config.UNKNOWN_CLUSTER_FILENAME, "merged_")

    mcid_map = defaultdict(list)
    aeid_map = defaultdict(set)
    num_mapped_clusters = 0
    for mc_id, medoid_vec, df_m, angle, mod, score in merged_clusters:
        if score > config.UNKNOWN_MERGED_CLUSTER_MIN_SCORE: # make the threshold dynamic
            unique_aeids = sorted(df_m["emitter_id"].unique().tolist())
            log.debug(f"[PASS] Merged Cluster ID: {mc_id} | (Score: {score:.2f}) | Emitter IDs: [{', '.join(map(str, unique_aeids))}]")
            start = time.perf_counter()
            mapped_sid = handle_new_emitter(df_m, known_centers_cache, pg_conn, ch_client, angle, mod, medoid_vec)
            print(f"handle_new_emitter time { (time.perf_counter() - start) * 1000 }") 
            if mapped_sid:
                mcid_map[mapped_sid].append(mc_id)
                aeid_map[mapped_sid].update(unique_aeids)
                num_mapped_clusters += 1
        else:
            log.info(f"\n\n[SKIP] Cluster {mc_id} (Score: {score:.2f}) -> Ignored (Low Confidence)\n")

    if config.DEBUG_UN_INFO:
        if mcid_map:
            total_signals = len(mcid_map)
            total_aeids = len(set(ae for aeids in aeid_map.values() for ae in aeids))
            aeid_present =  any(eids != {0} for eids in aeid_map.values())
            if aeid_present == False and total_aeids == 1:
                aeids_header = ''
            else:
                aeids_header = f"<- ({total_aeids} Unique Ground Truth Emitters)"

            parts = []
            for k, mc_ids in mcid_map.items():
                mc_str = ','.join(map(str, sorted(mc_ids)))
                aeid_str = ''
                if aeid_present:
                    aeid_str = "← (" + ", ".join(map(str, sorted(aeid_map[k]))) + ")"
                # Format: 0,1 → [9638] ← (1,2,4) 
                parts.append(f"                      {mc_str:>6} → [{k:<9}] {aeid_str:<15}\n")

            print("\n")
            log.info(f"{'='*100}")
            log.warning(f"Total {num_mapped_clusters} Merged Clusters mapped to -> [{total_signals:<6} New/Stored Signals] {aeids_header}")
            log.info(f"\nMapping ::: Merge Cluster ID → [Signal ID]\n{''.join(parts)}")
            log.info(f"{'='*100}")

    print("\n")


def get_angle_score(angle_diff):
    max_cutoff = 40.0
    plateau_limit = 20.0  
    angle_score = 0.0 
    if angle_diff <= plateau_limit:
        # Anything inside the main inner beam gets a perfect score
        angle_score = 1.0
    elif angle_diff <= max_cutoff:
        # Smoothly decay only across the outer edge (from 20 to 40 degrees)
        edge_zone = max_cutoff - plateau_limit  # 20 degrees wide
        scaled_rad = ((angle_diff - plateau_limit) / edge_zone) * (np.pi / 2)
        angle_score = np.cos(scaled_rad) ** 2
    else:
        angle_score = 0.0
    
    return angle_score


def merge_clusters(clusters, sim_threshold=0.70, angle_threshold=20.0):
    if not clusters:
        return

    if config.DEBUG_UN_INFO:
        log.info(f"{'~'*100}")
        log.info(f"Processing {len(clusters)} Unknown Initial Clusters)")
        all_eids = set()
        for _, _, df in clusters:
            if not df.empty and "emitter_id" in df.columns:
                all_eids.update(df["emitter_id"].unique())
        eids_str = ", ".join(map(str, sorted(all_eids))) if all_eids else "(none)"
        if list(all_eids) != [0]:
            log.info(f"Unique Truth Emitter IDs across all clusters are {len(all_eids)} :: ({eids_str})")
        log.info(f"{'~'*100}")
        print("")

    embedding_cols = [f"dim_{i}" for i in range(1024)]
    n_clusters = len(clusters)
    
    precomputed_vectors = []
    precomputed_angles = []
    
    for c_id, medoid_vector, df_cluster in clusters:

        if df_cluster.empty or "matched_emitter_id" not in df_cluster.columns:
            log.warning(" Skipping emitter-id clustering: missing 'matched_emitter_id' column or empty dataframe.")

             
        precomputed_vectors.append(medoid_vector)
        angle, _  = get_cluster_angle_score(df_cluster)
        precomputed_angles.append(angle)
        
    precomputed_vectors = np.array(precomputed_vectors)
    
    # Track items merged using a Connected Components Strategy
    # This prevents the "Greedy Chain Skipping" bug
    adj_matrix = {i: set([i]) for i in range(n_clusters)}
    pairwise_sims = np.dot(precomputed_vectors, precomputed_vectors.T)
    #cross_sim_matrix = np.dot(cluster_A, cluster_B.T)

    #median_similarity = np.median(cross_sim_matrix)
    
    for i in range(n_clusters):
        cluster_A_matrix = clusters[i][2][embedding_cols].to_numpy()
        for j in range(i + 1, n_clusters):
            cluster_B_matrix = clusters[j][2][embedding_cols].to_numpy()
            cross_sim_matrix = np.dot(cluster_A_matrix, cluster_B_matrix.T) 
            
            # The Median Way: Find the 50th percentile connection
            median_similarity = np.median(cross_sim_matrix)
            # logger.LOG_DBG(f"Pair ({i}, {j}) - Median Similarity: {median_similarity:.4f}")
            
            sim = pairwise_sims[i, j]
            # print(f" similarity with pair i and j{sim,i,j}")
            #angle_diff = abs(precomputed_angles[i] - precomputed_angles[j])
            raw_diff = (precomputed_angles[i] - precomputed_angles[j] + 180) % 360 - 180
            angle_diff = abs(raw_diff)
            angle_score = get_angle_score(angle_diff)
            # logger.LOG_DBG(f" between cluster angle_score {angle_score}")  
            final_score = (sim*8+angle_score*2)/10
            #final_score = sim
            # logger.LOG_DBG(f" between cluster final_score {final_score}") 
            if final_score > sim_threshold:
                adj_matrix[i].add(j)
                adj_matrix[j].add(i)
                # logger.LOG_DBG(f" $$ Merging clusters {i,j}")
            # else:
            #     logger.LOG_DBG(f" @@ Not Merging clusters {i,j}")
    visited = set()

    merged = []  # 1. Initialize the list at the start
    next_id = 0  # Initialize next_id if not done elsewhere

    for i in range(n_clusters):
        if i in visited:
            continue
            
        # Unroll the cluster chain using BFS
        component = []
        queue = [i]
        visited.add(i)
        
        while queue:
            curr = queue.pop(0)
            component.append(curr)
            # print(f"Adding cluster {curr}")
            for neighbor in adj_matrix[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        # Group and construct the final merged cluster
        dfs_to_combine = [clusters[idx][2] for idx in component]
        #final_df = pd.concat(dfs_to_combine).drop_duplicates()
        final_df = pd.concat(dfs_to_combine)
        # logger.LOG_DBG(f"combining df {len(final_df),curr}")
        # Calculate new medoid for unified cluster
        cluster_vectors = final_df[embedding_cols].to_numpy().astype(np.float32)
        pairwise_sim = np.dot(cluster_vectors, cluster_vectors.T)
        local_medoid_idx = np.argmax(pairwise_sim.sum(axis=1))
        
        medoid_vector = cluster_vectors[local_medoid_idx]
        
        avg_medoid_sim = np.mean(np.dot(medoid_vector, cluster_vectors.T))
        log.debug(f"local_medoid_idx = {local_medoid_idx}, avg similarity {avg_medoid_sim}")
        score, angle, modulation = get_cluster_score(final_df, avg_medoid_sim)

        merged.append((next_id, medoid_vector, final_df, angle, modulation, score))
        next_id += 1

    if config.DEBUG_UN_INFO:
        print_cluster_summary_table(merged)
        # plot_data.plot_pri_vs_time(merged_clusters, cluster_id=None)
        
    return merged



def handle_new_emitter(combined_df, known_centers_cache: Dict[int, EmitterTrack], pg_conn, ch_client, angle, modulation, medoid_vector):
    # Edge case: If cache is empty, immediately treat as a completely new signal
    from pipeline import create_instert_new_signal_emitter
    from cluster_handling import save_cluster_centers_cache
    from cluster_handling import calculate_all_cluster_metrics_vectorized

    if not known_centers_cache:
        if (len(combined_df) > UNKNOWN_MIN_SIGNAL_SIZE):
            new_signal_id, matched_metrics = create_instert_new_signal_emitter(combined_df, pg_conn, ch_client, angle, False, 0)
            if config.DEBUG_UN_PLOTS:
                cid = combined_df["cluster_id"].iloc[0]
                cluster_tuple = (cid, None, combined_df, 0, 0, 0)
                if matched_metrics:
                    plot_data.print_cluster_stats(combined_df, matched_metrics, new_signal_id)
                    plot_data.plot_emitter_doa_vs_power_vs_rotation_polar(combined_df, matched_metrics, new_signal_id)
                    plot_data.print_cluster_stats(combined_df, matched_metrics, new_signal_id)
                    # plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='time_of_arrival_ns', feature_y='power_dbm')
                    # plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='time_of_arrival_ns', feature_y='center_frequency_hz')
                    # plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='direction_of_arrival_deg', feature_y='time_of_arrival_ns')
                    # plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='time_of_arrival_ns', feature_y='duration_us')
        else:
            new_signal_id = 0
        known_centers_cache[0] = EmitterTrack(vector=medoid_vector, angle=angle, modulation=modulation, cardinality=len(combined_df), signal_id=new_signal_id)
    
        save_cluster_centers_cache()
        return new_signal_id

    ALLOWED_ANGLE_DELTA = 40
    new_signal_id = 0
    cluster_ids_list = list(known_centers_cache.keys())
    centers_matrix = np.vstack([known_centers_cache[sid].vector for sid in cluster_ids_list])
    
    # 1. Compute similarity (1D array output)
    similarity_scores = np.dot(medoid_vector, centers_matrix.T)
    
    # 2. Corrected index and max extraction (No axis=1)
    best_idx = np.argmax(similarity_scores).item()
    best_similarity_value = similarity_scores[best_idx]
    
    signal_id = 0
    is_existing_signal_present = False
    matched_cluster_id = cluster_ids_list[best_idx]


    raw_diff = (angle - known_centers_cache[matched_cluster_id].angle + 180) % 360 - 180
    angle_diff = abs(raw_diff)
    if angle_diff <= ALLOWED_ANGLE_DELTA:
        scaled_rad = (angle_diff / ALLOWED_ANGLE_DELTA) * (np.pi / 2)
        angle_score = np.cos(scaled_rad) ** 2
    else:
        angle_score = 0.0

    log.verbose(f" between cluster angle_score {angle_score}")  
    final_score = (best_similarity_value*8+angle_score*2)/10
    log.verbose(f" between cluster final_score {final_score}")
    
    all_eids = set()
    all_eids.update(combined_df["emitter_id"].unique())
    if 0 in all_eids and len(all_eids) == 1:
        aeids_str = ''
    else:
        emitter_list = ', '.join(map(str, sorted(all_eids)))
        aeids_str = f"(Unique Ground Truth Emitters: {emitter_list})"
    
    # 5. Database execution and cache update
    old_cardinality = 0
    cid = combined_df["cluster_id"].iloc[0]
    if final_score > UNKNOWN_DOT_PRODUCT_THRESHOLD:
        old_cardinality = known_centers_cache[matched_cluster_id].cardinality
        if known_centers_cache[matched_cluster_id].signal_id:
            is_existing_signal_present = True
            signal_id = known_centers_cache[matched_cluster_id].signal_id
            new_signal_id, matched_metrics = create_instert_new_signal_emitter(
                combined_df, pg_conn, ch_client, angle, is_existing_signal_present, signal_id
            )
        else:
            if (old_cardinality + len(combined_df) > UNKNOWN_MIN_SIGNAL_SIZE):
                is_existing_signal_present = False
                new_signal_id, matched_metrics = create_instert_new_signal_emitter(
                    combined_df, pg_conn, ch_client, angle, is_existing_signal_present, signal_id
                )
                signal_id = new_signal_id
        log.info(f"Associating Unknown Merged Cluster id {cid} with Existing Signal id [{matched_cluster_id}]. {aeids_str}")
    else:
        matched_cluster_id = len(known_centers_cache)
        if (len(combined_df) > UNKNOWN_MIN_SIGNAL_SIZE):
            is_existing_signal_present = False
            new_signal_id, matched_metrics = create_instert_new_signal_emitter(
                combined_df, pg_conn, ch_client, angle, is_existing_signal_present, signal_id
            )
            signal_id = new_signal_id

    if(angle_score > 0.9 ):
        log.info(f"******** *************** High chance of signal in cluster-{cid} is similar to existing cluster id [{matched_cluster_id}] based on angle direction. {aeids_str}")

    if config.DEBUG_UN_PLOTS:
        cluster_tuple = (cid, None, combined_df, 0, 0, 0)
        plot_data.plot_pri_vs_time([cluster_tuple], cid)
        if matched_metrics:
            plot_data.print_cluster_stats(combined_df, matched_metrics, new_signal_id)
            plot_data.plot_emitter_doa_vs_power_vs_rotation_polar(combined_df, matched_metrics, new_signal_id)
            plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='time_of_arrival_ns', feature_y='power_dbm')
            plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='time_of_arrival_ns', feature_y='center_frequency_hz')
            plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id, feature_x='direction_of_arrival_deg', feature_y='time_of_arrival_ns')
            plot_data.plot_cluster_heatmap_distribution(combined_df, matched_metrics, new_signal_id,  feature_y='duration_us', feature_x='time_of_arrival_ns')
        # plot_data.print_time_angle_power_table(combined_df, new_signal_id, limit=150)
    
    known_centers_cache[matched_cluster_id] = EmitterTrack(
        vector=medoid_vector,
        angle=angle,
        modulation=modulation,
        cardinality=old_cardinality+len(combined_df),
        signal_id=signal_id
    )
    save_cluster_centers_cache()
    if new_signal_id:
        
        clean_id = int(matched_cluster_id)

        if clean_id not in emitter_cache:
            emitter_cache[clean_id] = [[], []] 

        current_emitter_data = emitter_cache[clean_id]
        
        current_emitter_data[0].extend(combined_df['power_dbm'].tolist())
        current_emitter_data[1].extend(combined_df['direction_of_arrival_deg'].tolist())

        angle = get_cluster_angle(
            np.array(current_emitter_data[1]), 
            np.array(current_emitter_data[0])
        )
        # print(f"Angle combined {angle}")
        
        # print(f"New Emitter with Signal id {clean_id} is identified")
        # print(f"Major Angle for this emitter: {angle}")
        # update DOA which is more refined into signal table 
        update_doa_in_signal(pg_conn, new_signal_id, angle)
        
    return new_signal_id
