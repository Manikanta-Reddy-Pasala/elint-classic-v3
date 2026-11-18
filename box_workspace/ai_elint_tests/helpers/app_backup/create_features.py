from collections import Counter
from sklearn.metrics import classification_report
import math
import os
import numpy as np
#from utils.create_features import generate_features_per_window

'''
Generate features per window , ideally this should be called from both test and app
'''
def generate_features_per_window(freq, power, raw_pw, raw_toa, window_size=40):
    # Precompute radius for window indexing
    radius = window_size // 2

    # -------------------------------------------------------------------------
    # 1. SETUP ADAPTIVE LOG-SPACED TOA KERNELS (256 Centers, 10 to 5000)
    # -------------------------------------------------------------------------
    num_toa_centers = 256
    toa_lags_raw = np.logspace(np.log10(10.0), np.log10(5000.0), num=num_toa_centers)

    toa_diff_right = toa_lags_raw[1:] - toa_lags_raw[:-1]
    toa_diff_left  = toa_diff_right
    toa_next_diff = np.concatenate([toa_diff_right, toa_diff_right[-1:]])
    toa_prev_diff = np.concatenate([toa_diff_left[:1], toa_diff_left])

    toa_gaps = np.minimum(toa_next_diff, toa_prev_diff)
    toa_sigma = toa_gaps * (2 / 3)

    # Re-shaped to [1, 1, 256] to broadcast over [512, window_size, 256]
    toa_denom = (2 * (toa_sigma ** 2)).reshape(1, 1, num_toa_centers)
    toa_lags_3d = toa_lags_raw.reshape(1, 1, num_toa_centers)

    # -------------------------------------------------------------------------
    # 2. SETUP ADAPTIVE LOG-SPACED PW KERNELS (100 Centers, 0.1 to 200)
    # -------------------------------------------------------------------------
    num_pw_centers = 100
    pw_lags_raw = np.logspace(np.log10(0.1), np.log10(200.0), num=num_pw_centers)

    pw_diff_right = pw_lags_raw[1:] - pw_lags_raw[:-1]
    pw_diff_left  = pw_diff_right
    pw_next_diff = np.concatenate([pw_diff_right, pw_diff_right[-1:]])
    pw_prev_diff = np.concatenate([pw_diff_left[:1], pw_diff_left])

    pw_gaps = np.minimum(pw_next_diff, pw_prev_diff)
    pw_sigma = pw_gaps * (2 / 3)
    pw_denom = (2 * (pw_sigma ** 2)).reshape(1, num_pw_centers)

    # Pre-create the boundary index grids for TOA calculation
    indices = np.arange(512)[..., np.newaxis]
    offsets = np.arange(-radius, radius + 1)
    clamped_centers = np.clip(indices, radius, 512 - 1 - radius)
    window_indices = clamped_centers + offsets  # Shape: [512, window_size]

    # Calculate and scale basic features natively from individual inputs
    # Ensuring 2D shape [512, 1] for correct concatenation later
    freq_scaled = (freq.reshape(-1, 1) - 50.0) / 10950.0
    power_scaled = (power.reshape(-1, 1) + 100.0) / 100.0

    clamped_pw = np.clip(raw_pw, 0.1, 100.0)

    # -----------------------------------------------------------------
    # 3. VECTORIZED PW RBF CALCULATIONS (Shape: 512, 100)
    # -----------------------------------------------------------------
    pw_diffs = clamped_pw[..., np.newaxis] - pw_lags_raw.reshape(1, num_pw_centers)
    pw_rbf = np.exp(-(pw_diffs ** 2) / pw_denom)

    # -----------------------------------------------------------------
    # 4. HIGH-SPEED LOCAL WINDOW TOA DIFF MAP (Unpadded Boundary Clamp)
    # -----------------------------------------------------------------
    # Gather localized neighborhoods safely without padding. Shape: [512, window_size]
    toa_windows = raw_toa[window_indices]

    # Calculate local absolute differences. Shape: [512, window_size]
    diffs = np.abs(raw_toa[..., np.newaxis] - toa_windows)

    # Unsqueeze into 3D [512, window_size, 1] to cleanly align with logspace kernels
    diffs_3d = diffs[..., np.newaxis]

    # Calculate localized RBF values. Shape: [512, window_size, 256]
    rbf_matrix = np.exp(-((diffs_3d - toa_lags_3d) ** 2) / toa_denom)

    # Sum down the window dimension (axis=1) to compress features back to 256. Shape: [512, 256]
    toa_rbf = rbf_matrix.sum(axis=1)

    # -----------------------------------------------------------------
    # 5. FEATURE CONCATENATION & STORAGE (Output Shape: 512, 358)
    # -----------------------------------------------------------------
    features = np.concatenate([freq_scaled, power_scaled, pw_rbf, toa_rbf], axis=-1)

    return features



