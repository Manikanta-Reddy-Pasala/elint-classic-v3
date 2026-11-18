
import pandas as pd
import numpy as np
from typing import Dict

from cluster_refinement import get_cluster_angle_score, get_cluster_score,get_cluster_angle

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

# storage for emitter level clusters , 
emitter_cache = {}

def simple_clustering(df: pd.DataFrame, known_centers_cache: Dict[int, EmitterTrack], pg_conn, ch_client, threshold=0.8):
    embedding_cols = [f"dim_{i}" for i in range(1024)]
    
    if not all(col in df.columns for col in embedding_cols):
        print(" Skipping unknown processing: Dataframe missing vector embedding columns.")
        return df
    print("simple_clustering")
    X = df[embedding_cols].to_numpy().astype(np.float32)
    n_samples = X.shape[0]
    clusters = []
    labels = np.full(n_samples, -1, dtype=int)
    current_cluster_id = 0
    
    for i in range(n_samples):
        if labels[i] != -1:
            continue
            
        leader_vector = X[i]
        similarities = np.dot(X, leader_vector)
        matching_mask = (similarities >= threshold) & (labels == -1)
        matching_indices = np.where(matching_mask)[0]
        
        if len(matching_indices) >= MIN_CLUSTER_SIZE:
            cluster_similarities = similarities[matching_mask]
            median_sim = np.median(cluster_similarities)
            
            print(f"Cluster {current_cluster_id} (Leader index {i}):")
            print(f"  -> Bundled indices   : {matching_indices.tolist()}")
            print(f"  -> Median Similarity : {median_sim:.4f}\n")
            labels[matching_mask] = current_cluster_id
            
            # Extract medoid
            cluster_vectors = X[matching_indices]
            pairwise_sim = np.dot(cluster_vectors, cluster_vectors.T)
            local_medoid_idx = np.argmax(pairwise_sim.sum(axis=1))
            
            # Store the specific medoid vector directly to avoid indexing bugs later
            medoid_vector = cluster_vectors[local_medoid_idx]
            angle_score, _ = get_cluster_angle_score(df.iloc[matching_indices])
            if(angle_score > 0.3): #only taking good clusters 
                # Pass the sub-dataframe slice
                clusters.append((current_cluster_id, medoid_vector, df.iloc[matching_indices].copy()))
                current_cluster_id += 1
            
    # Execute the merge step
    merge_clusters(clusters, known_centers_cache, pg_conn, ch_client)


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

def merge_clusters(clusters, known_centers_cache, pg_conn, ch_client, sim_threshold=0.70, angle_threshold=20.0):
    if not clusters:
        return
        
    embedding_cols = [f"dim_{i}" for i in range(1024)]
    n_clusters = len(clusters)
    
    precomputed_vectors = []
    precomputed_angles = []
    
    for c_id, medoid_vector, df_cluster in clusters:
        precomputed_vectors.append(medoid_vector)
        _, angle = get_cluster_angle_score(df_cluster)
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
            print(f"Pair ({i}, {j}) - Median Similarity: {median_similarity:.4f}")
            
            sim = pairwise_sims[i, j]
            print(f"similarity with pair i and j{sim,i,j}")
            #angle_diff = abs(precomputed_angles[i] - precomputed_angles[j])
            raw_diff = (precomputed_angles[i] - precomputed_angles[j] + 180) % 360 - 180
            angle_diff = abs(raw_diff)
            angle_score = get_angle_score(angle_diff)
            print(f" between cluster angle_score {angle_score}")  
            final_score = (sim*8+angle_score*2)/10
            print(f" between cluster final_score {final_score}") 
            if final_score > sim_threshold:
                adj_matrix[i].add(j)
                adj_matrix[j].add(i)
                print(f"Merging clusters {i,j}")
    visited = set()
    
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
            print(f"Adding cluster {curr}")
            for neighbor in adj_matrix[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        # Group and construct the final merged cluster
        dfs_to_combine = [clusters[idx][2] for idx in component]
        #final_df = pd.concat(dfs_to_combine).drop_duplicates()
        final_df = pd.concat(dfs_to_combine)
        print(f"combining df {len(final_df),curr}")
        # Calculate new medoid for unified cluster
        cluster_vectors = final_df[embedding_cols].to_numpy().astype(np.float32)
        pairwise_sim = np.dot(cluster_vectors, cluster_vectors.T)
        local_medoid_idx = np.argmax(pairwise_sim.sum(axis=1))
        
        mediod_vector = cluster_vectors[local_medoid_idx]
        
        avg_medoid_sim = np.mean(np.dot(mediod_vector, cluster_vectors.T))
        print(f"avg similarity {avg_medoid_sim}")
        score, angle, modulation = get_cluster_score(final_df, avg_medoid_sim)
        #print(f"cluster final score {score}")
        #np.set_printoptions(threshold=np.inf)
        print(np.diff   (final_df['TOA'])/1000)
        # TODO: update threshold logic from static 0.2 to dynamic metric
        if score > 0.3: 
            handle_new_emitter(final_df, known_centers_cache, pg_conn, ch_client, angle, modulation, mediod_vector)




def handle_new_emitter(combined_df, known_centers_cache: Dict[int, EmitterTrack], pg_conn, ch_client, angle, modulation, mediod_vector):
    # Edge case: If cache is empty, immediately treat as a completely new signal
    from pipeline import create_instert_new_signal_emitter
    from cluster_handling import save_cluster_centers_cache
    if not known_centers_cache:
        new_signal_id = create_instert_new_signal_emitter(combined_df, pg_conn, ch_client, angle, False, 0)
        if new_signal_id is not None:
            known_centers_cache[int(new_signal_id)] = EmitterTrack(vector=mediod_vector, angle=angle, modulation=modulation)
            save_cluster_centers_cache()
        return

    ALLOWED_ANGLE_DELTA = 40
    signal_ids_list = list(known_centers_cache.keys())
    centers_matrix = np.vstack([known_centers_cache[sid].vector for sid in signal_ids_list])
    
    # 1. Compute similarity (1D array output)
    similarity_scores = np.dot(mediod_vector, centers_matrix.T)
    
    # 2. Corrected index and max extraction (No axis=1)
    best_idx = np.argmax(similarity_scores).item()
    best_similarity_value = similarity_scores[best_idx]
    
    signal_id = 0
    is_existing_signal_present = False
    matched_signal_id = signal_ids_list[best_idx]


    raw_diff = (angle - known_centers_cache[matched_signal_id].angle + 180) % 360 - 180
    angle_diff = abs(raw_diff)
    if angle_diff <= ALLOWED_ANGLE_DELTA:
            scaled_rad = (angle_diff / ALLOWED_ANGLE_DELTA) * (np.pi / 2)
            angle_score = np.cos(scaled_rad) ** 2
    else:
            angle_score = 0.0

    print(f" between cluster angle_score {angle_score} ")  
    final_score = (best_similarity_value*8+angle_score*2)/10
    print(f" between cluster final_score {final_score}")
    
    if final_score > UNKNOWN_DOT_PRODUCT_THRESHOLD:
        is_existing_signal_present = True
        signal_id = matched_signal_id
        print(f"Associating new cluster with existing signal id {matched_signal_id}")
    elif(angle_score > 0.9 ):
        print("******** *************** High chance  they are same emitter ")

    # 5. Database execution and cache update
   
    new_signal_id = create_instert_new_signal_emitter(
        combined_df, pg_conn, ch_client, angle, is_existing_signal_present, signal_id
    )
    
    if new_signal_id is not None:
        clean_id = int(new_signal_id)

        if clean_id not in emitter_cache:
            emitter_cache[clean_id] = [[], []] 
            
        current_emitter_data = emitter_cache[clean_id]
        
        current_emitter_data[0].extend(combined_df['power_dbm'].tolist())
        current_emitter_data[1].extend(combined_df['direction_of_arrival_deg'].tolist())

        angle = get_cluster_angle(
            np.array(current_emitter_data[1]), 
            np.array(current_emitter_data[0])
        )
        print(f"Angle combined {angle}")
        
        print(f"New Emitter with Signal id {clean_id} is identified")
        print(f"Major Angle for this emitter: {angle}")
        # update DOA which is more refined into signal table 
        update_doa_in_signal(pg_conn, new_signal_id,angle)
        known_centers_cache[clean_id] = EmitterTrack(
            vector=mediod_vector,
            angle=angle,
            modulation=modulation
        )
        save_cluster_centers_cache()

        
