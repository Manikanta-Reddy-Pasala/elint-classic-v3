import numpy as np
import pandas as pd
from typing import Tuple
import logger

def get_cluster_similarity_score(df: pd.DataFrame) -> float:
    embedding_cols = [f"dim_{i}" for i in range(1024)]

    if not all(col in df.columns for col in embedding_cols):
        return 0.0


    X = df[embedding_cols].to_numpy().astype(np.float32)
    pairwise_sim = np.dot(X, X.T)
    local_medoid_idx = np.argmax(pairwise_sim.sum(axis=1))
    # Can Medoid vector be based on highes SNR? How far that will be from overall center?
 
        
    medoid_vector = X[local_medoid_idx]

    #Column vector:: similarity of each row with medoid of the cluster
    similarities = np.dot(X, medoid_vector)


    df['similarities'] = similarities

    return medoid_vector, similarities, float(np.median(similarities))

def get_cluster_angle(angles, power_dbm):
    # 1. Convert dBm to linear milliwatts
    power_mw = 10 ** (power_dbm / 10)
    
    # 2. Find the index of the absolute maximum power peak
    peak_idx = np.argmax(power_mw)
    
    # Define window bounds around the peak (1-2 rows before, 1-2 rows after)
    start_idx = max(0, peak_idx - 2)
    end_idx = min(len(power_mw), peak_idx + 3)
    
    # Extract the peak neighborhood values
    peak_angles = angles[start_idx:end_idx]
    peak_power_mw = power_mw[start_idx:end_idx]
    
    # 1. Convert degrees to radians
    peak_angles_rad = np.radians(peak_angles)
    
    # 2. Compute weighted components on cartesian plane (X and Y coordinates)
    x_component = np.average(np.cos(peak_angles_rad), weights=peak_power_mw)
    y_component = np.average(np.sin(peak_angles_rad), weights=peak_power_mw)
    
    # 3. Resolve back to radians and map to positive [0, 360] degrees
    major_angle = np.degrees(np.arctan2(y_component, x_component)) % 360
    return major_angle

def get_cluster_angle_score(df: pd.DataFrame) -> float:
   
    if df.empty:
        return 0.0, 0.0
        
    angles = pd.to_numeric(df['direction_of_arrival_deg'], errors='coerce').values
    power_dbm = df['power_dbm'].values
    
    # 1. Convert dBm to linear milliwatts
    power_mw = 10 ** (power_dbm / 10)
    
    # 2. Find the index of the absolute maximum power peak
    peak_idx = np.argmax(power_mw)
    
        # Define window bounds around the peak (1-2 rows before, 1-2 rows after)
    start_idx = max(0, peak_idx - 2)
    end_idx = min(len(power_mw), peak_idx + 3)
    
    # Extract the peak neighborhood values
    peak_angles = angles[start_idx:end_idx]
    peak_power_mw = power_mw[start_idx:end_idx]
    
    # 1. Convert degrees to radians
    peak_angles_rad = np.radians(peak_angles)
    
    # 2. Compute weighted components on cartesian plane (X and Y coordinates)
    x_component = np.average(np.cos(peak_angles_rad), weights=peak_power_mw)
    y_component = np.average(np.sin(peak_angles_rad), weights=peak_power_mw)
    
    # 3. Resolve back to radians and map to positive [0, 360] degrees
    major_angle = np.degrees(np.arctan2(y_component, x_component)) % 360

    
    # 3. Beam boundaries (+/- 30 degrees)
    half_beamwidth = 40.0 
    
    # 4. Calculate deviations with 180/-180 wrap-around protection
    angular_deviations = np.abs(angles - major_angle)
    angular_deviations = np.where(angular_deviations > 180, 360 - angular_deviations, angular_deviations)
    
    # 5. Count how many total pulses fall inside the beamwidth
    inside_mask = angular_deviations <= half_beamwidth
    num_inside = np.sum(inside_mask)
    total_pulses = len(angles)
    
    # 6. Calculate score as a fraction
    score_val = num_inside / total_pulses
    
    # print(f"major angles {major_angle}, len(df)={len(df)}")
    # print(f"score_val {score_val}")
    '''
    if score_val < 0.3:
        print(angles)
        print(power_dbm)
        print(df['duration_us'])
    #print(f"power_dbm: {power_dbm}")
    #print(f"angles: {angles}")
    '''
    return  major_angle, float(score_val)

def get_modulation_score(df: pd.DataFrame) -> Tuple[float, str]:

    if df.empty:
        return 0.0, "UNKNOWN"
        
    total_pulses = len(df)
    
    # 1. Count the occurrences of each modulation type
    mod_counts = df['Modulation'].value_counts()
    
    # Identify the highest-frequency modulation type string
    major_modulation = str(mod_counts.idxmax())
    
    # If there is only 1 unique modulation type, return perfect score instantly
    if len(mod_counts) <= 1:
        return 1.0, major_modulation
        
    # 2. Calculate probabilities for each modulation type
    probabilities = mod_counts.values / total_pulses
    
    # 3. Calculate Shannon Entropy to measure disorder/mix
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # 4. Find the maximum possible entropy for this number of unique types
    max_entropy = np.log2(len(mod_counts))
    
    # 5. Normalize score: 1.0 is perfectly ordered, drops toward 0.0 as it gets mixed
    modulation_score = 1.0 - (entropy / max_entropy)
    
    return float(modulation_score), major_modulation

def get_cluster_score(df: pd.DataFrame, avg_medoid_sim: float) -> float:
    """Computes and logs the balanced final cluster metrics."""
    major_angle, angle_score =get_cluster_angle_score(df)
    modulation_score = 0 #get_modulation_score(df)
    major_modulation = 0

    final_score = (avg_medoid_sim + angle_score + modulation_score) / 3
    
    logger.LOG_VRB("--- Cluster Scoring Metrics ---")
    logger.LOG_VRB(f"Total Pulses Evaluated : {len(df)}")
    logger.LOG_VRB(f"Average Medoid Sim     : {avg_medoid_sim:.4f}")
    logger.LOG_VRB(f"Beamwidth Angle Score  : {angle_score:.4f}")
    logger.LOG_VRB(f"Modulation Consistency : {modulation_score:.4f}")
    logger.LOG_VRB(f"Calculated Final Score : {final_score:.4f}")
    logger.LOG_VRB("-------------------------------")
    
    return float(final_score),major_angle,major_modulation