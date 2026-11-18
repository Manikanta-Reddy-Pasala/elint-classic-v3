import numpy as np
import pandas as pd


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
        
    angles = df['direction_of_arrival_deg'].values
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
    
    print(f"major angles {major_angle}, len(df)={len(df)}")
    print(f"score_val {score_val}")
    '''
    if score_val < 0.3:
        print(angles)
        print(power_dbm)
        print(df['duration_us'])
    #print(f"power_dbm: {power_dbm}")
    #print(f"angles: {angles}")
    '''
    return float(score_val), major_angle


from typing import Tuple
import pandas as pd
import numpy as np

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
    angle_score,major_angle =get_cluster_angle_score(df)
    modulation_score = 0 #get_modulation_score(df)
    major_modulation = 0
    final_score = (avg_medoid_sim + angle_score + modulation_score) / 3
    
    print("--- Cluster Scoring Metrics ---")
    print(f"Total Pulses Evaluated : {len(df)}")
    print(f"Average Medoid Sim     : {avg_medoid_sim:.4f}")
    print(f"Beamwidth Angle Score  : {angle_score:.4f}")
    print(f"Modulation Consistency : {modulation_score:.4f}")
    print(f"Calculated Final Score : {final_score:.4f}")
    print("-------------------------------")
    
    return float(final_score),major_angle,major_modulation