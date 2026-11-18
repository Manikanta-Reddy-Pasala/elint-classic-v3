
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import time
from matplotlib.colors import ListedColormap
import logger
import config
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def _save_and_print_link(fig, filename_prefix, output_dir="/tmp"):
    """Helper to save figure and print clickable link."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        file_url = f"file://{os.path.abspath(filepath)}"
        print(f"👉 Plot saved. Open Graph: {file_url}\n")
        return filepath
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig)
        return None

def save_filtered_clusters(clusters, desired_columns, output_path, prefix=None):
    if config.DEBUG_UN_SAVE_CSV:
        dfs = [item[2] for item in clusters]
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            save_filtered_dataframe(combined_df, desired_columns, output_path, prefix=None)

def save_filtered_dataframe(df, desired_columns, output_path, prefix=None):
    """
    Saves a filtered dataframe to CSV with specific columns in order.

    Args:
        df: Pandas DataFrame or path to CSV file.
        desired_columns: List of column names to keep (e.g., ['col1', 'col2']).
        output_path: Full path for the output file.
        prefix: Optional string prefix to add to the filename (e.g., 'unmerged_').

    Returns:
        The final output path used.
    """
    if config.DEBUG_UN_SAVE_CSV:
        # 1. Load dataframe if a string path is provided
        if isinstance(df, str):
            if not os.path.exists(df):
                raise FileNotFoundError(f"Input file not found: {df}")
            df = pd.read_csv(df)

        # 2. Safety Check: Ensure df is actually a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame or file path string, but got {type(df).__name__}. "
                            f"Did you pass 'desired_columns' as the first argument by mistake?")

        # 2. Filter columns: Keep only those in desired_columns that exist in df
        # Maintains the order from desired_columns
        available_cols = [col for col in desired_columns if col in df.columns]

        if not available_cols:
            raise ValueError("No matching columns found between desired list and dataframe.")

        df_to_save = df[available_cols]

        # 3. Handle filename prefix
        if prefix:
            dirname, filename = os.path.split(output_path)
            filename = f"{prefix}{filename}"
            output_path = os.path.join(dirname, filename)

        # Check if file exists to handle header writing correctly
        file_exists = os.path.isfile(output_path)
        df_to_save.to_csv(output_path, mode='a', header=not file_exists, index=False)
        logger.LOG_INF(f"✅ Successfully saved {len(df_to_save)} pulses to : {output_path}")

        return output_path


def print_pri_summary(results_summary, remove_outliers=False):
    """
    Prints a formatted summary table of PRI stats and outlier removal statistics.

    Args:
        results_summary: List of dicts with keys: 'id', 'peaks', 'total', 'removed', 'min_removed', 'max_removed'
        remove_outliers: Boolean flag to adjust the header text.
    """
    if not results_summary:
        return

    w_cid, w_total, w_r1 = 10, 6, 24
    sep = f"{'-'*w_cid}-+-{'-'*w_total}-+{'-'*w_r1}-+{'-'*w_r1}-+{'-'*w_r1}-+{'-'*(w_r1+7)}"

    logger.LOG_INF(f"{'~' * 130}")
    logger.LOG_INF(f"Top 3 Dominant PRIs (us) - {'after Outliers removed' if remove_outliers else 'without Outliers removed'}")
    logger.LOG_INF(f"{'~' * 130}")
    logger.LOG_INF(f"{'Cluster ID':<{w_cid}} | {'Total':<{w_total}} | {'Rank 1 (us)':<{w_r1}}| {'Rank 2 (us)':<{w_r1}}| {'Rank  3 (us)':<{w_r1}}| {'Removed      [Min , Max]'}")
    logger.LOG_INF(sep)

    for res in results_summary:
        if not res['peaks']:
            logger.LOG_INF(f"{res['id']:<{w_cid}} | {res['total']:<{w_total}} | {'No Data':<{w_r1}}| {'':<{w_r1}}| {'':<{w_r1}}| {'':<{w_r1}}")
            continue

        def fmt_peak(p):
            if not p: return f"N/A".ljust(w_r1)
            return f"{p[0]:>6.1f} ({p[2]:>5.1f}%)".ljust(w_r1)

        p1 = res['peaks'][0] if len(res['peaks']) > 0 else None
        p2 = res['peaks'][1] if len(res['peaks']) > 1 else None
        p3 = res['peaks'][2] if len(res['peaks']) > 2 else None

        rem_pct = (res['removed'] / res['total'] * 100) if res['total'] > 0 else 0

        # Format Min/Max removed
        if res['removed'] > 0 and res['min_removed'] is not None:
            rem_info = f"{res['removed']:>3} ({rem_pct:>5.1f}%) [Min:{res['min_removed']:>6.1f} , Max:{res['max_removed']:>6.1f}]"
        else:
            rem_info = f"{res['removed']:>3} ({rem_pct:>5.1f}%)"

        row = f"{res['id']:<{w_cid}} | {res['total']:<{w_total}} | "
        row += f"{fmt_peak(p1)}| {fmt_peak(p2)}| {fmt_peak(p3)}| "
        row += rem_info.ljust(w_r1)

        logger.LOG_INF(row)

def plot_pri_vs_time(merged_clusters, cluster_id=None, remove_outliers = True, output_dir="plots", max_plots=10):
    """
    Plots Delta Time (us) vs Time of Arrival (ns) in subplots.
    Calculates outlier mask directly using median * 3.0 logic.
    Prints summary of Top 3 PRIs and removed counts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cluster_id is not None:
        clusters_to_plot = [c for c in merged_clusters if c[0] == cluster_id]
        max_plots = 1
    else:
        clusters_to_plot = merged_clusters[:max_plots]

    if not clusters_to_plot:
        return

    n_plots = len(clusters_to_plot)

    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        axes_flat = [ax]
    else:
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
        axes_flat = axes.flatten()

    results_summary = []
    for idx, (mc_id, medoid, df_m, score, angle, mod) in enumerate(clusters_to_plot):
        ax = axes_flat[idx]

        if df_m.empty or 'time_of_arrival_ns' not in df_m.columns:
            ax.set_title(f"Cl {mc_id}: No Data")
            continue

        # 1. Sort and extract times
        df_sorted = df_m.sort_values('time_of_arrival_ns').reset_index(drop=True)
        times_ns = df_sorted['time_of_arrival_ns'].to_numpy()

        if len(times_ns) < 2:
            ax.set_title(f"Cl {mc_id}: < 2 pulses")
            continue

        # 2. Calculate deltas
        ns_vals = [getattr(t, 'value', t) if hasattr(t, 'value') else t for t in times_ns]
        deltas_us = np.diff(ns_vals).astype(float) / 1_000.0
        median_delta = float(np.median(deltas_us))
        x_points = times_ns[1:]

        original_count = len(deltas_us)

        # 3. Calculate Outlier Mask IN PLACE
        threshold = median_delta * 3.0 if median_delta > 0 else 100.0
        if remove_outliers:
            removed_deltas = deltas_us[deltas_us > threshold]
            plot_mask = deltas_us <= threshold
        else:
            removed_deltas = []
            plot_mask = deltas_us == deltas_us

        removed_count = len(removed_deltas)

        # Mask for plotting: Keep non-outliers
        x_clean = x_points[plot_mask]
        y_clean = deltas_us[plot_mask]

        if len(x_clean) == 0:
            ax.set_title(f"Cl {mc_id}: All Outliers")
            results_summary.append({
                "id": mc_id, "peaks": [], "total": original_count,
                "removed": removed_count, "min_removed": None, "max_removed": None
            })
            continue

        # 4. Plot
        ax.scatter(x_clean, y_clean, s=10, alpha=0.6, color='blue', marker='.', linewidths=0)
        ax.set_title(f"Merged Cluster-id {mc_id}", fontsize=10)
        ax.set_xlabel('Time of Arrival (ns)', fontsize=8)
        ax.set_ylabel('Delta Time (us)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        # --- Calculate Top 3 Dominant Deltas for Summary ---
        bin_width = 10.0
        max_val = max(y_clean) if len(y_clean) > 0 else 0
        if max_val < bin_width: max_val = bin_width

        counts, bin_edges = np.histogram(y_clean, bins=np.arange(0, max_val + bin_width, bin_width))
        top_indices = np.argsort(counts)[-3:][::-1]

        peaks = []
        for idx_peak in top_indices:
            if counts[idx_peak] == 0: continue
            peak_val = (bin_edges[idx_peak] + bin_edges[idx_peak+1]) / 2
            pct = (counts[idx_peak] / len(y_clean)) * 100
            peaks.append((peak_val, counts[idx_peak], pct))

        min_rem = float(np.min(removed_deltas)) if len(removed_deltas) > 0 else None
        max_rem = float(np.max(removed_deltas)) if len(removed_deltas) > 0 else None

        results_summary.append({
            "id": mc_id,
            "peaks": peaks,
            "total": original_count,
            "removed": removed_count,
            "min_removed": min_rem,
            "max_removed": max_rem
        })

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('PRI (us) vs Time (ns) - Outliers Removed', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    suffix = f"_{cluster_id}" if cluster_id is not None else "_all"
    filepath = os.path.join(output_dir, f"pri_vs_time{suffix}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    logger.LOG_INF(f"👉 Plot saved: file://{os.path.abspath(filepath)}")

    # print_pri_summary(results_summary, remove_outliers)


def plot_cluster_heatmap_distribution(df, summary, target_cluster_id, feature_x='direction_of_arrival_deg', feature_y='power_dbm', output_dir="/tmp"):
    """
    Plots Density Heatmap.
    X-Axis: Sorted by Time (Time_of_arrival_ns) to show evolution.
    Y-Axis: The feature (e.g., Power).
    """
    # 1. Ensure summary is a DataFrame
    if isinstance(summary, list):
        if not summary: return
        summary = pd.DataFrame(summary)

    # 2. Filter Raw Data
    if 'target_signal_id' not in df.columns: return
    cluster_df = df[df['target_signal_id'] == target_cluster_id].copy()
    if cluster_df.empty: return

    # Check if required columns exist
    if feature_x not in cluster_df.columns or feature_y not in cluster_df.columns:
        print(f"Error: Missing columns. Available: {cluster_df.columns.tolist()}")
        return

    # CRITICAL: Check if we have time data for sorting
    if 'time_of_arrival_ns' not in cluster_df.columns:
        print("Warning: 'time_of_arrival_ns' not found. Data will not be time-sorted.")
        use_time_sort = False
    else:
        use_time_sort = True

    # 3. Define Rounding Logic
    def round_feature(val, col_name):
        if 'power' in col_name.lower():
            return np.round(val / 3) * 3
        elif 'frequency' in col_name.lower() or 'freq' in col_name.lower():
            return np.round(val / 10000) * 10000
        elif 'duration' in col_name.lower() or 'pw' in col_name.lower():
            # Convert ns to us, then round
            val_us = val / 1000.0
            return np.round(val_us / 10) * 10
        elif 'direction' in col_name.lower() or 'doa' in col_name.lower():
            return np.round(val / 5) * 5
        else:
            return val

    # Apply rounding to Y (and X if needed, but usually X is the variable)
    cluster_df['y_rounded'] = cluster_df[feature_y].apply(lambda x: round_feature(x, feature_y))

    # 4. Calculate Density
    density_map = cluster_df.groupby('y_rounded').size().to_dict()
    cluster_df['density'] = cluster_df['y_rounded'].map(density_map)

    # 5. SORT BY TIME (Crucial for X-axis evolution)
    if use_time_sort:
        cluster_df = cluster_df.sort_values('time_of_arrival_ns').reset_index(drop=True)

    # 6. Create Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize colors
    if cluster_df['density'].max() > cluster_df['density'].min():
        norm = Normalize(vmin=cluster_df['density'].min(), vmax=cluster_df['density'].max())
        cmap = plt.cm.coolwarm
    else:
        norm = Normalize(vmin=1, vmax=1)
        cmap = plt.cm.Blues

    # Plot Scatter
    # Since we sorted by time, the points are plotted in chronological order
    # This makes the X-axis (Angle) evolve over time
    scatter = ax.scatter(
        cluster_df[feature_x],
        cluster_df[feature_y],
        c=cluster_df['density'],
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=40,
        edgecolors='w',
        linewidth=0.5,
        label='Pulses (Color = Density)'
    )

    # Add Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pulse Count (in rounded bin)')

    # 7. Add Summary Center
    summary_col_map = {
        'center_frequency_hz': 'freq_mean', 'power_dbm': 'power_mean',
        'duration_us': 'pw_mean', 'bandwidth_hz': 'bw_mean', 'direction_of_arrival_deg': 'doa_mean'
    }
    summary_std_map = {
        'center_frequency_hz': 'freq_std', 'power_dbm': 'power_std',
        'duration_us': 'pw_std', 'bandwidth_hz': 'bw_std', 'direction_of_arrival_deg': 'doa_std'
    }

    if feature_y in summary_col_map:
        summary_key = summary_col_map[feature_y]
        summary_std_key = summary_std_map[feature_y]
        summary_row = summary[summary['target_signal_id'] == target_cluster_id]

        if not summary_row.empty and summary_key in summary_row.columns:
            sum_val = summary_row[summary_key].values[0]
            std_val = summary_row[summary_std_key].values[0] if summary_std_key in summary_row.columns else 0

            ax.axhline(y=sum_val, color='lime', linestyle='--', linewidth=2, label=f'Summary Mean ({sum_val:.1f})')
            if std_val > 0:
                ax.axhspan(sum_val - std_val, sum_val + std_val, color='lime', alpha=0.2)

    # 8. Formatting
    ax.set_title(f'Cluster {target_cluster_id}: {feature_y} vs {feature_x} (Time-Sorted)')
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 9. Save and Link
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"cluster_{target_cluster_id}_{feature_y}_vs_{feature_x}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.LOG_INF(f"👉 Plot {feature_x} vs {feature_y} saved. Open Graph: file://{os.path.abspath(filepath)}")

def print_time_angle_power_table(df, target_cluster_id, limit=20):
    """
    Prints a table of Time, Angle, and Power to verify data distribution.
    """
    # 1. Ensure target_cluster_id is a single value (int/str), not a list
    if isinstance(target_cluster_id, list):
        if len(target_cluster_id) == 0:
            print("Error: target_cluster_id is an empty list.")
            return
        target_cluster_id = target_cluster_id[0] # Take the first element

    if 'target_signal_id' not in df.columns:
        print("Error: 'target_signal_id' not found.")
        return

    # 2. Filter Data
    cluster_df = df[df['target_signal_id'] == target_cluster_id].copy()

    if cluster_df.empty:
        print(f"No data for Cluster {target_cluster_id}")
        return

    # Sort by Time
    cluster_df = cluster_df.sort_values('time_of_arrival_ns').reset_index(drop=True)

    # Select columns
    cols = ['time_of_arrival_ns', 'direction_of_arrival_deg', 'power_dbm']
    available_cols = [c for c in cols if c in cluster_df.columns]

    if not available_cols:
        print("No valid columns found.")
        return

    # Get first and last chunks
    head = cluster_df.head(limit)
    tail = cluster_df.tail(limit)

    print(f"\n{'='*80}")
    print(f"Data Check: Cluster {target_cluster_id} (Total Pulses: {len(cluster_df)})")
    print(f"{'='*80}")

    # Print Head
    print(f"--- FIRST {limit} PULSES (Start of Signal) ---")
    print(head[available_cols].to_string(index=False))

    # Print Tail
    print(f"\n--- LAST {limit} PULSES (End of Signal) ---")
    print(tail[available_cols].to_string(index=False))

    # Print Statistics
    print(f"\n{'='*80}")
    print("Statistics:")
    print(f"Min Power: {cluster_df['power_dbm'].min():.1f} dBm")
    print(f"Max Power: {cluster_df['power_dbm'].max():.1f} dBm")
    print(f"Avg Power: {cluster_df['power_dbm'].mean():.1f} dBm")
    print(f"{'='*80}\n")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from matplotlib.colors import Normalize

def plot_polar_lobes(df, summary, target_cluster_id, output_dir="/tmp"):
    """
    Plots a Polar Lobe View:
    - Angle (Theta): direction_of_arrival_deg
    - Radius (r): Normalized Time (0 to 1) to show the "arc" of the scan
    - Color: Power (dBm)
    """
    # 1. Filter Data
    if 'target_signal_id' not in df.columns: return
    cluster_df = df[df['target_signal_id'] == target_cluster_id].copy()
    if cluster_df.empty: return

    required_cols = ['direction_of_arrival_deg', 'time_of_arrival_ns', 'power_dbm']
    if not all(col in cluster_df.columns for col in required_cols):
        print(f"Error: Missing columns. Need: {required_cols}")
        return

    # 2. Sort by Time
    cluster_df = cluster_df.sort_values('time_of_arrival_ns').reset_index(drop=True)

    # 3. Normalize Time to 0-1 (Radius)
    # This compresses the long spiral into a compact arc
    min_time = cluster_df['time_of_arrival_ns'].min()
    max_time = cluster_df['time_of_arrival_ns'].max()

    if max_time == min_time:
        cluster_df['radius'] = 0.5
    else:
        cluster_df['radius'] = (cluster_df['time_of_arrival_ns'] - min_time) / (max_time - min_time)

    # 4. Normalize Power for Color
    min_p = cluster_df['power_dbm'].min()
    max_p = cluster_df['power_dbm'].max()
    norm = Normalize(vmin=min_p, vmax=max_p)
    cmap = plt.cm.coolwarm

    # 5. Create Polar Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')

    # Convert DOA to Radians
    angles_rad = np.radians(cluster_df['direction_of_arrival_deg'])
    radii = cluster_df['radius'] # 0 to 1

    # Scatter Plot
    scatter = ax.scatter(
        angles_rad,
        radii,
        c=cluster_df['power_dbm'],
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        s=40,
        edgecolors='w',
        linewidth=0.5
    )

    # 6. Add Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Power (dBm)')

    # 7. Add Summary Mean DOA (if available)
    # summary_row = summary[summary['target_signal_id'] == target_cluster_id]
    # if not summary_row.empty:
    #     mean_col = 'doa_mean' if 'doa_mean' in summary.columns else 'direction_of_arrival_deg'
    #     if mean_col in summary_row.columns:
    #         mean_doa = summary_row[mean_col].values[0]
    #         mean_rad = np.radians(mean_doa)
    #         # Draw line at mean angle
    #         ax.plot([mean_rad, mean_rad], [0, 1], 'lime', linestyle='--', linewidth=2, label=f'Mean DOA ({mean_doa:.1f}°)')
    #         ax.legend(loc='upper right')

    # 8. Formatting
    ax.set_theta_zero_location("N") # 0 at North
    ax.set_theta_direction(-1) # Clockwise
    ax.set_title(f'Cluster {target_cluster_id}: Polar Lobe View\n(Radius = Time, Color = Power)', va='bottom')

    # Set radial limits to ensure it looks like a lobe, not a full circle
    ax.set_ylim(0, 1.1)

    # 9. Save and Link
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"cluster_{target_cluster_id}_polar_lobe_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ Plot saved: {filepath}")
    print(f"👉 Open Graph: file://{os.path.abspath(filepath)}\n")




def plot_emitter_doa_vs_power_vs_rotation_polar(df, summary, target_cluster_id, output_dir="/tmp"):
    """
    Plots Polar View with rotation detection based on 'Sudden Drop' logic.
    Detects new rotation when angle drops significantly (e.g., 140 -> 110).
    """
    # 1. Filter & Sort
    if 'target_signal_id' not in df.columns: return
    cluster_df = df[df['target_signal_id'] == target_cluster_id].copy()
    if cluster_df.empty: return

    required = ['direction_of_arrival_deg', 'time_of_arrival_ns', 'power_dbm']
    if not all(c in cluster_df.columns for c in required):
        print(f"Missing columns: {required}")
        return

    # Sort by Time
    cluster_df = cluster_df.sort_values('time_of_arrival_ns').reset_index(drop=True)

    angles = cluster_df['direction_of_arrival_deg'].values
    rotations = np.zeros(len(angles), dtype=int)

    # 2. Robust Rotation Detection: "Sudden Drop" Logic
    current_rotation = 0
    rotations = [0] * len(angles)
    DROP_THRESHOLD = 30.0

    for i in range(1, len(angles)):
        prev = angles[i-1]
        curr = angles[i]
        is_rotation = False

        # 1. Check for Backward Wrap (10 -> 350)
        if prev < 180 and curr > 180:
            # Calculate drop: (10) - (350 - 360) = 10 - (-10) = 20
            drop = prev - (curr - 360)
            if drop > DROP_THRESHOLD:
                is_rotation = True

        # 2. Check for Normal Drop (140 -> 100)
        elif prev - curr > DROP_THRESHOLD:
            is_rotation = True

        # 3. Forward Wrap (350 -> 10) is implicitly ignored here because:
        #    It doesn't match condition 1 (prev > 180)
        #    It doesn't match condition 2 (350 - 10 = 340 > 30? YES! This is the bug!)

        # FIX: We must explicitly EXCLUDE forward wraps from the normal drop check.
        if prev > 180 and curr < 180:
            # Forward wrap, ignore
            is_rotation = False
        elif prev < 180 and curr > 180:
            # Backward wrap
            drop = prev - (curr - 360)
            if drop > DROP_THRESHOLD:
                is_rotation = True
        else:
            # Normal
            if (prev - curr) > DROP_THRESHOLD:
                is_rotation = True

        if is_rotation:
            current_rotation += 1
        rotations[i] = current_rotation

    cluster_df['rotation'] = rotations
    num_rotations = current_rotation + 1

    # 3. Colors
    base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'lime', 'navy', 'teal']
    colors = [base_colors[i % len(base_colors)] for i in range(num_rotations)]
    cmap = ListedColormap(colors)
    norm = plt.Normalize(vmin=0, vmax=num_rotations - 1)

    # 4. Radius = Power (dBm)
    min_p = cluster_df['power_dbm'].min()
    max_p = cluster_df['power_dbm'].max()

    if max_p == min_p:
        radii = np.ones(len(angles)) * 0.5
    else:
        radii = (cluster_df['power_dbm'] - min_p) / (max_p - min_p)
        radii = radii * 0.9 + 0.1

    angles_rad = np.radians(cluster_df['direction_of_arrival_deg'])

    # 5. Plot
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='polar')

    scatter = ax.scatter(
        angles_rad,
        radii,
        c=cluster_df['rotation'],
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        s=60,
        edgecolors='white',
        linewidth=0.5
    )

    # Legend
    handles = []
    for i in range(num_rotations):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colors[i], markersize=12,
                                  label=f'Rotation {i+1}'))
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)

    # Formatting
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f'Cluster {target_cluster_id}: Power vs Angle\n(Radius=Power, Color=Rotation Cycle)', va='bottom', fontsize=14)

    # Y-Ticks for Power
    tick_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    tick_labels = [f"{min_p + (max_p-min_p)*v:.0f} dBm" for v in tick_vals]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(0, 1.1)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"cluster_{target_cluster_id}_power_rot_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.LOG_INF(f"👉 Polar plot saved. Open Graph: file://{os.path.abspath(filepath)}")


def print_cluster_stats(df, summary_df, mapped_emitter_id):

    cid=df['cluster_id'].iloc[0]
    row_count=len(df)
    row = summary_df[0]

    # 1. Extract DOA stats
    doa_stats = [
        row['doa_mean'],
        row['doa_min'],
        row['doa_max'],
        row['doa_std']
    ]

    # 2. Extract PRI stats (treated exactly the same way)
    pri_stats = [
        row['pri_mean'],
        row['pri_min'],
        row['pri_max'],
        row['pri_std']
    ]

    d_mean, d_min, d_max, d_std = doa_stats
    p_mean, p_min, p_max, p_std = pri_stats

    # Create the specific formatted strings
    doa_str = f"(avg:{d_mean:.0f}, min:{d_min:.0f}, max:{d_max:.0f}, std:{d_std:.0f})"
    pri_str = f"(avg:{p_mean:.0f}, min:{p_min:.0f}, max:{p_max:.0f}, std:{p_std:.0f})"
    id_count_str = f"({row_count}) cid:{cid}"

    print("")
    logger.LOG_INF(f"{id_count_str:<12}  | eID::{mapped_emitter_id:<8} | DoA:: {doa_str:<26} | PRI:: {pri_str:<26}")


# -----------------------------------------------------------------------------
# 1. Radar Chart
# -----------------------------------------------------------------------------
def plot_cluster_radar_charts(summary_df, top_n=5, output_dir="/tmp"):
    if summary_df.empty:
        return

    top_clusters = summary_df.nlargest(top_n, 'pdw_count')

    features = [
        ('freq_mean', 'Freq (MHz)'),
        ('power_mean', 'Power (dBm)'),
        ('pw_mean', 'PW (us)'),
        ('bw_mean', 'BW (kHz)'),
        ('doa_mean', 'DOA (deg)'),
        ('pri_mean', 'PRI (us)')
    ]

    # Normalize
    norm_df = summary_df.copy()
    for col, _ in features:
        if col in norm_df.columns:
            min_val = norm_df[col].min()
            max_val = norm_df[col].max()
            if max_val > min_val:
                norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
            else:
                norm_df[col] = 0.5

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(1, top_n, figsize=(5 * top_n, 5), subplot_kw=dict(polar=True))
    if top_n == 1: axs = [axs]

    for i, idx in enumerate(top_clusters.index):
        ax = axs[i]
        row = norm_df.loc[idx]
        values = [row[f] for f, _ in features]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=f"Cl {idx} ({row['pdw_count']} PDWs)")
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f for _, f in features], fontsize=10)
        ax.set_title(f"Cluster {idx}", fontsize=12, pad=20)
        ax.grid(True)

    plt.tight_layout()
    _save_and_print_link(fig, "radar_chart", output_dir)

# -----------------------------------------------------------------------------
# 2. Correlation Heatmap
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(summary_df, output_dir="/tmp"):
    if summary_df.empty:
        return

    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    corr_matrix = summary_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)

    ax.set_title('Correlation Matrix of Cluster Metrics', fontsize=16)
    plt.tight_layout()
    _save_and_print_link(fig, "correlation_heatmap", output_dir)




def plot_feature_scatter_matrix(summary_df, sample_size=50, output_dir="/tmp"):
    """
    Robust scatter matrix plot with automatic column detection.
    """
    if summary_df.empty:
        print("Dataframe is empty.")
        return

    # 1. Select features
    cols_to_plot = ['freq_mean', 'power_mean', 'pw_mean', 'bw_mean', 'doa_mean', 'pri_mean']
    # Filter to only existing columns
    cols_to_plot = [c for c in cols_to_plot if c in summary_df.columns]

    if len(cols_to_plot) < 2:
        print(f"Not enough features to plot. Found: {cols_to_plot}")
        return

    # 2. Identify the ID column
    # Check if 'target_signal_id' exists, otherwise try to use the index or first column
    id_col = 'target_signal_id'
    if id_col not in summary_df.columns:
        # If it's an index, reset it to make it a column
        if id_col in summary_df.index.name:
            summary_df = summary_df.reset_index()
            print(f"Note: '{id_col}' was an index. Reset to column.")

        # If still not found, try to find a generic ID column
        if id_col not in summary_df.columns:
            # Fallback: use the first column that isn't in our numeric list
            potential_ids = [c for c in summary_df.columns if c not in cols_to_plot]
            if potential_ids:
                id_col = potential_ids[0]
                print(f"Note: '{id_col}' not found. Using '{potential_ids[0]}' as ID.")
            else:
                print("Error: No ID column found to use for 'hue'.")
                return

    # 3. Prepare Data
    # Ensure we only select columns that actually exist
    available_cols = cols_to_plot + [id_col]
    df_plot = summary_df[available_cols].copy()

    # Drop rows with any NaN in the selected columns
    df_plot = df_plot.dropna(subset=available_cols)

    if df_plot.empty:
        print("No valid data remaining after dropping NaNs.")
        return

    # Sample if too large
    if len(df_plot) > sample_size:
        df_plot = df_plot.sample(n=sample_size, random_state=42)

    # 4. Force Data Types
    for col in cols_to_plot:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

    # Drop rows again if numeric conversion created NaNs
    df_plot = df_plot.dropna(subset=cols_to_plot)
    if df_plot.empty:
        print("No valid data after numeric conversion.")
        return

    df_plot[id_col] = df_plot[id_col].astype(str)

    # 5. Create Plot
    try:
        g = sns.pairplot(df_plot, vars=cols_to_plot, hue=id_col,
                         palette='tab10', plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'w'})

        g.fig.suptitle(f'Feature Distribution Scatter Matrix (Hue: {id_col})', y=1.02, fontsize=16)

        _save_and_print_link(g.fig, "scatter_matrix", output_dir)

    except Exception as e:
        print(f"Error creating pairplot: {e}")
        plt.close('all')


# -----------------------------------------------------------------------------
# 4. PRI Stability vs Power
# -----------------------------------------------------------------------------
def plot_pri_stability_vs_power(summary_df, output_dir="/tmp"):
    if summary_df.empty or 'pri_std' not in summary_df.columns:
        print("No data for PRI stability plot.")
        return

    df_plot = summary_df.sort_values('pri_std').reset_index(drop=True)
    cluster_ids = df_plot['target_signal_id'].map(str)
    power_vals = df_plot['power_mean'].fillna(df_plot['power_mean'].mean())

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(cluster_ids, df_plot['pri_std'],
                   color=plt.cm.viridis(power_vals.values / power_vals.max()))

    ax.set_xlabel('PRI Standard Deviation (us) - Lower is more stable')
    ax.set_ylabel('Cluster ID')
    ax.set_title('PRI Stability vs. Signal Power (Color)')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=power_vals.min(),
                                                                   vmax=power_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Power (dBm)', rotation=270, labelpad=20)

    plt.tight_layout()
    _save_and_print_link(fig, "pri_stability", output_dir)

def print_scatter_matrix_table(summary_df, sample_size=50):
    """
    Prints the cleaned dataframe used for the scatter matrix plot.
    """
    if summary_df.empty:
        print("Dataframe is empty.")
        return

    # 1. Select features
    cols_to_plot = ['freq_mean', 'power_mean', 'pw_mean', 'bw_mean', 'doa_mean', 'pri_mean']
    cols_to_plot = [c for c in cols_to_plot if c in summary_df.columns]

    if len(cols_to_plot) < 2:
        print(f"Not enough features to plot. Found: {cols_to_plot}")
        return

    # 2. Identify ID column
    id_col = 'target_signal_id'
    if id_col not in summary_df.columns:
        if id_col in summary_df.index.name:
            summary_df = summary_df.reset_index()
        if id_col not in summary_df.columns:
            potential_ids = [c for c in summary_df.columns if c not in cols_to_plot]
            if potential_ids:
                id_col = potential_ids[0]
            else:
                print("No ID column found.")
                return

    # 3. Prepare Data
    available_cols = cols_to_plot + [id_col]
    df_plot = summary_df[available_cols].copy()

    # Drop NaNs
    df_plot = df_plot.dropna(subset=available_cols)

    # Sample if too large
    if len(df_plot) > sample_size:
        df_plot = df_plot.sample(n=sample_size, random_state=42)

    # Force numeric types
    for col in cols_to_plot:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

    # Drop NaNs again
    df_plot = df_plot.dropna(subset=cols_to_plot)

    # Convert ID to string
    df_plot[id_col] = df_plot[id_col].astype(str)

    if df_plot.empty:
        print("No valid data to print.")
        return

    # 4. Print Table
    print(f"\n{'='*80}")
    print(f"Data used for Scatter Matrix (Sample size: {len(df_plot)})")
    print(f"{'='*80}")

    # Reorder columns to put ID first
    display_cols = [id_col] + cols_to_plot
    print(df_plot[display_cols].to_string(index=False))
    print(f"{'='*80}\n")
# -----------------------------------------------------------------------------



def plot_statistics(summary_df):
    plot_cluster_radar_charts(summary_df, top_n=5)

    # 2. Correlation Heatmap
    plot_correlation_heatmap(summary_df)

    # 3. Scatter Matrix
    print_scatter_matrix_table(summary_df)
    plot_feature_scatter_matrix(summary_df)

    # 4. PRI Stability
    # plot_pri_stability_vs_power(summary_df)