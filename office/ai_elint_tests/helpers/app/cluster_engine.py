import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from typing import Dict, List, Tuple


from typing import NamedTuple

class EmitterTrack(NamedTuple):
    vector: np.ndarray
    angle: float
    modulation: float


class ClusterEngineState:
    """Manages globally incrementing labels, multi-exemplar cache, and rolling windows."""
    def __init__(self, max_exemplars_per_signal: int = 20, window_batch_limit: int = 5):
        self.current_label_counter = 0
        self.max_exemplars_per_signal = max_exemplars_per_signal
        
        # --- NEW WINDOWED TRACKING STORAGE ---
        self.window_batch_limit = window_batch_limit
        self.batch_counter = 0
        self.transient_cluster_buffer: List[Dict] = []  # Holds raw cluster data before global merge

    def shift_and_update_labels(self, local_labels: np.ndarray) -> np.ndarray:
        """Offsets local HDBSCAN labels to preserve global index continuity."""
        if not np.any(local_labels != -1):
            return local_labels
            
        global_labels = np.where(local_labels != -1, local_labels + self.current_label_counter, -1)
        num_valid_discovered = int(np.max(local_labels)) + 1
        self.current_label_counter += num_valid_discovered
        return global_labels

    def buffer_discovered_cluster(self, global_label: int, df: pd.DataFrame, medoid_vector: np.ndarray, local_indices: np.ndarray):
        """Stages a newly discovered HDBSCAN cluster into transient window memory."""
        self.transient_cluster_buffer.append({
            'global_label': global_label,
            'df': df,
            'medoid': medoid_vector,
            'local_indices': local_indices
        })

    def should_trigger_merge(self) -> bool:
        """Increments micro-batch count and checks if the merge window is open."""
        self.batch_counter += 1
        if self.batch_counter >= self.window_batch_limit:
            self.batch_counter = 0  # Reset
            return True
        return False

    def clear_transient_buffer(self):
        """Clears out memory staging zones after a global transaction flush completes."""
        self.transient_cluster_buffer.clear()

    def add_exemplars_to_cache(self, cache_dict: dict, signal_id: int, new_vectors: np.ndarray):
        """Appends new vectors to a signal's historical exemplar matrix (FIFO rolling buffer)."""
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.reshape(1, -1)
        new_vectors = new_vectors.astype(np.float32)

        if signal_id in cache_dict:
            existing_matrix = cache_dict[signal_id]
            combined = np.vstack([existing_matrix, new_vectors])
            cache_dict[signal_id] = combined[-self.max_exemplars_per_signal:]
        else:
            cache_dict[signal_id] = new_vectors[-self.max_exemplars_per_signal:]




class ClusterUnionFind:
    """Resolves multi-hop components grouping paths using a Union-Find partition system."""
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def calculate_medoid_and_similarity(
    embeddings: np.ndarray, 
    local_indices: np.ndarray, 
    cosine_similarity: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """Calculates internal pairwise similarity metrics and extracts the medoid vector."""
    num_elements = len(local_indices)
    sub_matrix = cosine_similarity[local_indices[:, None], local_indices]
    
    if num_elements > 1:
        mean_sim = (np.sum(sub_matrix) - num_elements) / (num_elements * (num_elements - 1))
        medoid_sub_idx = np.argmax(np.sum(sub_matrix, axis=1))
        medoid_local_idx = local_indices[medoid_sub_idx]
        avg_medoid_sim = np.mean(sub_matrix[medoid_sub_idx].ravel())
        center_vector = embeddings[medoid_local_idx]
    else:
        mean_sim = 1.0
        avg_medoid_sim = 1.0
        medoid_local_idx = local_indices if num_elements > 0 else 0
        center_vector = embeddings[medoid_local_idx] if num_elements > 0 else np.zeros(embeddings.shape, dtype=np.float32)
        
    return mean_sim, avg_medoid_sim, center_vector


def group_similar_clusters(cluster_centers: Dict[int, np.ndarray], merge_threshold: float) -> Dict[int, List[int]]:
    """Calculates pairwise similarity across cluster medoids and joins groups via Union-Find."""
    labels = list(cluster_centers.keys())
    if not labels:
        return {}
        
    uf = ClusterUnionFind(labels)
    centers_matrix = np.vstack([cluster_centers[lbl] for lbl in labels])
    center_similarity = np.dot(centers_matrix, centers_matrix.T)
    #TODO:: while merging , we should take care of angle as well , merge clusters for which angles are same @anurag
    rows, cols = np.where(center_similarity >= merge_threshold)
    for r, c in zip(rows, cols):
        if r != c:
            uf.union(labels[r], labels[c])
            print(f"merging ROW and COLUMNs {r,c}")
            
    merged_groups = {}
    for lbl in labels:
        root = uf.find(lbl)
        if root not in merged_groups:
            merged_groups[root] = []
        merged_groups[root].append(lbl)
        
    return merged_groups
