//! Global-match finalization (E2E_MATCH mode).
//!
//! The engine collects unknown-pulse embeddings across all batches and
//! then calls `finalize` once at the end of the file. This runs ONE
//! global HDBSCAN + agglomerative merge + score gate + DB writes, so
//! the cluster output matches the old Python pipeline (which runs a
//! single HDBSCAN over the whole file).

use std::time::Instant;

use anyhow::Result;
use tracing::{info, warn};

use rayon::prelude::*;

use crate::cluster_merge;
use crate::hdbscan;
use crate::onnx_predict;

use super::{merge_threshold, score_gate, AiState};

/// One buffered batch: embeddings + pulse metadata + CH input ids, plus
/// the per-pulse ONNX predictions that belong to THIS batch (`None` for
/// classic and for any batch whose prediction call failed).
#[derive(Clone)]
pub struct BufferedBatch {
    pub embeddings: Vec<Vec<f32>>,
    pub meta: Vec<AiPulseMeta>,
    pub input_ids: Vec<i64>,
    pub predictions: Option<onnx_predict::OnnxPredictions>,
}

/// Uniform-stride sample indices for the HDBSCAN pass.
/// `max_s == 0` means "no cap" (all points). `global_cap > 0` is a
/// debug cap applied after the stride sampling.
pub fn sample_indices(n: usize, max_s: usize, global_cap: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = if max_s > 0 && n > max_s {
        let step = n / max_s;
        (0..n).step_by(step).collect()
    } else {
        (0..n).collect()
    };
    if global_cap > 0 && idx.len() > global_cap {
        let step = idx.len() / global_cap;
        idx = idx
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .map(|(_, v)| v)
            .take(global_cap)
            .collect();
    }
    idx
}

/// Pulse metadata carried through to the DB so cluster rows can carry
/// real per-pulse stats (mean frequency / power) instead of zeros.
#[derive(Clone)]
pub struct AiPulseMeta {
    pub id: i64,
    pub center_frequency_hz: f64,
    pub power_dbm: f64,
    pub snr_db: f64,
    pub duration_ns: f64,
    pub bandwidth_hz: f64,
    pub time_of_arrival_ns: f64,
    pub direction_of_arrival_deg: f64,
    pub modulation: String,
    /// Pulse repetition interval from the harness CSV column, in
    /// nanoseconds (0 when the column is absent). The classic
    /// classifier buckets by it; the HDBSCAN path ignores it.
    pub pulse_repetition_interval_ns: f64,
    /// Ground-truth emitter label from the CSV `emitter_id` column (None
    /// when the column is absent). Used to name the written emitter row
    /// the same way the Python app names it (e.g. "PSR3").
    pub emitter_label: Option<String>,
}

/// Per-cluster aggregate statistics matching the Python
/// `calculate_all_cluster_metrics_vectorized` output.
#[derive(Debug, Clone, Copy)]
pub struct ClusterStats {
    pub freq_mean: f64, pub freq_min: f64, pub freq_max: f64, pub freq_std: f64,
    pub power_mean: f64, pub power_min: f64, pub power_max: f64, pub power_std: f64,
    pub snr_mean: f64,  pub snr_min: f64,  pub snr_max: f64,  pub snr_std: f64,
    pub pw_mean: f64,   pub pw_min: f64,   pub pw_max: f64,   pub pw_std: f64,
    pub bw_mean: f64,   pub bw_min: f64,   pub bw_max: f64,   pub bw_std: f64,
    pub doa_mean: f64,  pub doa_min: f64,  pub doa_max: f64,  pub doa_std: f64,
    pub pri_mean: f64,  pub pri_min: f64,  pub pri_max: f64,  pub pri_std: f64,
    pub toa_ns: i64,
    pub pdw_count: i64,
}

impl ClusterStats {
    pub fn from_members(
        freqs: &[f64], pows: &[f64], snrs: &[f64], pws: &[f64], bws: &[f64],
        doas: &[f64], toas: &[f64], pdw_count: i64,
    ) -> Self {
        let (f_mean, f_min, f_max, f_std) = mean_min_max_std(freqs);
        let (p_mean, p_min, p_max, p_std) = mean_min_max_std(pows);
        let (s_mean, s_min, s_max, s_std) = mean_min_max_std(snrs);
        let (w_mean, w_min, w_max, w_std) = mean_min_max_std(pws);
        let (b_mean, b_min, b_max, b_std) = mean_min_max_std(bws);
        let (d_mean, d_min, d_max, d_std) = mean_min_max_std(doas);
        let (pri_mean, pri_min, pri_max, pri_std) = compute_pri_stats(toas);
        let toa_ns: i64 = if toas.is_empty() { 0 } else { toas.iter().copied().fold(f64::INFINITY, f64::min) as i64 };
        ClusterStats {
            freq_mean: f_mean, freq_min: f_min, freq_max: f_max, freq_std: f_std,
            power_mean: p_mean, power_min: p_min, power_max: p_max, power_std: p_std,
            snr_mean: s_mean, snr_min: s_min, snr_max: s_max, snr_std: s_std,
            pw_mean: w_mean, pw_min: w_min, pw_max: w_max, pw_std: w_std,
            bw_mean: b_mean, bw_min: b_min, bw_max: b_max, bw_std: b_std,
            doa_mean: d_mean, doa_min: d_min, doa_max: d_max, doa_std: d_std,
            pri_mean, pri_min, pri_max, pri_std,
            toa_ns,
            pdw_count,
        }
    }
}

/// mean, min, max, population std (ddof=0) of a slice.
fn mean_min_max_std(vals: &[f64]) -> (f64, f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, min, max, var.sqrt())
}

/// PRI stats matching the Python `calculate_iqr_pri_stats_per_group`.
/// `toas` is a slice of time-of-arrival values in nanoseconds, sorted.
fn compute_pri_stats(toas: &[f64]) -> (f64, f64, f64, f64) {
    if toas.len() < 2 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    // TOA deltas in microseconds (Python: .diff() / 1000.0)
    let mut pri_vals: Vec<f64> = toas.windows(2)
        .map(|w| (w[1] - w[0]) / 1000.0)
        .filter(|&v| v.is_finite())
        .collect();
    if pri_vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    // IQR outlier removal
    if pri_vals.len() >= 4 {
        let mut sorted = pri_vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q25 = percentile(&sorted, 0.25);
        let q75 = percentile(&sorted, 0.75);
        let iqr = q75 - q25;
        let lo = q25 - 1.5 * iqr;
        let hi = q75 + 1.5 * iqr;
        pri_vals.retain(|&v| v >= lo && v <= hi);
    }
    if pri_vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    // Fixed-bin histogram peak estimation (bin_width = 3.0 µs)
    let bin_width = 3.0f64;
    let lo = pri_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = pri_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_bins = ((hi - lo) / bin_width).ceil().max(1.0) as usize + 1;
    let mut counts = vec![0usize; n_bins];
    for &v in &pri_vals {
        let idx = ((v - lo) / bin_width).floor() as usize;
        if idx < n_bins {
            counts[idx] += 1;
        }
    }
    let max_bin = counts.iter().enumerate().max_by(|a, b| a.1.cmp(&b.1)).map(|(i, _)| i).unwrap_or(0);
    let bin_start = lo + max_bin as f64 * bin_width;
    let bin_end = bin_start + bin_width;
    let is_last = max_bin == counts.len() - 1;
    let in_bin: Vec<f64> = pri_vals
        .iter()
        .filter(|&&v| if is_last { v >= bin_start && v <= bin_end } else { v >= bin_start && v < bin_end })
        .copied()
        .collect();
    let dominant = if in_bin.is_empty() {
        (bin_start + bin_end) / 2.0
    } else {
        in_bin.iter().sum::<f64>() / in_bin.len() as f64
    };
    let n = pri_vals.len() as f64;
    let p_min = pri_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = pri_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = pri_vals.iter().sum::<f64>() / n;
    let var = pri_vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (dominant, p_min, p_max, var.sqrt())
}

fn label_of(member_ids: &[i64]) -> Option<String> {
    let mut votes: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for id in member_ids {
        if let Some(l) = crate::refresh::emitter_labels().read().unwrap().get(id).cloned() {
            *votes.entry(l).or_insert(0) += 1;
        }
    }
    votes.into_iter().max_by_key(|(_, c)| *c).map(|(l, _)| l)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Modulation string → numeric id (Python MODULATION_MAPPING).
pub fn modulation_id(mod_str: &str) -> f64 {
    match mod_str.to_lowercase().as_str() {
        "unmodulated" | "fixed" => 0.0,
        "continuous_wave" | "continuous wave" | "cw" => 1.0,
        "pulse" => 2.0,
        "phase_modulation" | "phase modulation" | "pm" => 3.0,
        "frequency_modulation" | "frequency modulation" | "fm" => 4.0,
        "amplitude_modulation" | "amplitude modulation" | "am" => 5.0,
        "linear_chirp" | "linear_chirp_up" | "lfm" | "lfm_up" => 6.0,
        "down_chirp" | "linear_chirp_down" | "lfm_down" => 7.0,
        "pulse_position_modulation" | "ppm" => 8.0,
        "pulse_width_modulation" | "pwm" => 9.0,
        "unknown" => 10.0,
        _ => 10.0,
    }
}

/// Run the single global HDBSCAN over all collected unknown embeddings
/// and write the resulting emitters/clusters to PG + CH.
pub fn finalize<'a>(
    state: &'a AiState,
    unknowns: &'a [BufferedBatch],
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(finalize_impl(state, unknowns))
}
async fn finalize_impl(state: &AiState, unknowns: &[BufferedBatch]) -> Result<()> {
    let t0 = Instant::now();
    if state.engine.name() == "onnx" && !crate::refresh::onnx_prediction_routing_disabled() {
        return finalize_onnx_predictions(state, unknowns).await;
    }
    // Flatten: one (embeddings, pulse_ids) pair per batch → one big vec.
    // Keeping the pulse ids alongside the embeddings means every DB row
    // can reference the real pulse set of its cluster (Python writes
    // pdw_count per cluster and the cluster table carries the pulse ids).
    let mut all: Vec<Vec<f32>> = Vec::new();
    let mut all_meta: Vec<AiPulseMeta> = Vec::new();
    let mut all_input_ids: Vec<i64> = Vec::new();
    for b in unknowns {
        all.extend(b.embeddings.iter().cloned());
        all_meta.extend(b.meta.iter().cloned());
        all_input_ids.extend(b.input_ids.iter().copied());
    }
    info!(
        "global_match: {} unknown embeddings across {} batches",
        all.len(),
        unknowns.len()
    );
    if all.is_empty() {
        info!("global_match: no unknowns — nothing to cluster");
        return Ok(());
    }

    // Sample for the HDBSCAN pass (same uniform-stride sampling as
    // before, but we keep the original indices so medoids can be mapped
    // back to real embeddings + pulse ids without any cap).
    let max_s = state.cfg.max_samples;
    // E2E_GLOBAL_CAP caps the HDBSCAN input (uniform stride over the
    // flattened input). 0 = no cap (all points).
    let global_cap: usize = std::env::var("E2E_GLOBAL_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let sample_idx = sample_indices(all.len(), max_s, global_cap);
    info!(
        "global_match: HDBSCAN on {} of {} points (max_samples={})",
        sample_idx.len(),
        all.len(),
        max_s
    );
    if global_cap > 0 {
        info!("global_match: capped HDBSCAN input to {global_cap}");
    }
    let sample: Vec<Vec<f32>> = sample_idx.iter().map(|&i| all[i].clone()).collect();

    let t_hd = Instant::now();
    // Pass 0 as max_samples to hdbscan so it uses all of `sample`.
    let hd = hdbscan::hdbscan_cosine(&sample, state.cfg.min_cluster_size, 0);
    let t_hdbscan = t_hd.elapsed();
    // Debug: per-cluster sizes (the medoid pass is O(members^2) per
    // cluster, so one giant component dominates the wall time).
    let mut by_label_dbg: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();
    for &l in hd.labels.iter() {
        if l >= 0 {
            *by_label_dbg.entry(l).or_insert(0) += 1;
        }
    }
    let sizes_dbg: Vec<usize> = by_label_dbg.values().cloned().collect();
    let max_c = sizes_dbg.iter().copied().max().unwrap_or(0);
    let noise = hd.labels.iter().filter(|&&l| l == -1).count();
    info!(
        "global_match: {} clusters, max component = {} members, {} noise",
        sizes_dbg.len(),
        max_c,
        noise
    );
    let n_clusters = hd
        .labels
        .iter()
        .filter(|&&l| l >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    info!(
        "TIMING [global] hdbscan={:.1}ms ({} unknowns, {} clusters, min_size={})",
        t_hdbscan.as_secs_f64() * 1000.0,
        all.len(),
        n_clusters,
        state.cfg.min_cluster_size
    );

    // Map HDBSCAN labels back to the original (pre-sampling) indices.
    let mut by_label: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();
    for (ci, &label) in hd.labels.iter().enumerate() {
        if label >= 0 {
            by_label.entry(label).or_default().push(sample_idx[ci]);
        }
    }

    struct Stat {
        medoid: Vec<f32>,
        size: usize,
    }
    let dim = sample[0].len();
    let entries: Vec<(&i64, &Vec<usize>)> = by_label
        .iter()
        .filter(|(_, idxs)| idxs.len() >= state.cfg.min_cluster_size)
        .collect();
    info!("global_match: {} clusters to compute medoids for (total {} members)",
        entries.len(),
        entries.iter().map(|(_, i)| i.len()).sum::<usize>());
    let t_med = Instant::now();
    // Build flat matrix over the FULL embedding set for medoid
    // computation (BLAS-friendly, no per-cluster Vec indexing).
    let flat_all: Vec<f32> = all.iter().flat_map(|v| v.iter().copied()).collect();
    let stats: Vec<Stat> = entries
        .par_iter()
        .map(|(_, idxs)| {
            // Medoid = point with the best mean cosine similarity to its
            // cluster members. idxs now holds original (pre-sampling)
            // indices, so rows come from the full embedding matrix.
            // Mean cosine to members — O(|idxs|²) in the naive form, which
            // blew up to minutes on one giant component (83k pulses). Cap
            // the per-candidate member set: a uniform stride keeps the
            // medoid choice stable (the argmax of a near-uniform mean is
            // the central point either way) and cuts the cost to
            // |idxs| * min(|idxs|, MED_CAP).
            const MED_CAP: usize = 256;
            let mset: Vec<usize> = if idxs.len() > MED_CAP {
                let step = idxs.len() / MED_CAP;
                (0..idxs.len()).step_by(step).map(|k| idxs[k]).collect()
            } else {
                idxs.to_vec()
            };
            let (mut best_i, mut best_score) = (0usize, f64::NEG_INFINITY);
            for &ci in idxs.iter() {
                let row_ci = &flat_all[ci * dim..(ci + 1) * dim];
                let mut sum = 0.0f64;
                for &cj in mset.iter() {
                    let row_cj = &flat_all[cj * dim..(cj + 1) * dim];
                    // dot product
                    let mut dot: f32 = 0.0;
                    let chunks = (dim / 4) * 4;
                    let mut d = 0;
                    while d < chunks {
                        dot += row_ci[d] * row_cj[d]
                             + row_ci[d+1] * row_cj[d+1]
                             + row_ci[d+2] * row_cj[d+2]
                             + row_ci[d+3] * row_cj[d+3];
                        d += 4;
                    }
                    while d < dim {
                        dot += row_ci[d] * row_cj[d];
                        d += 1;
                    }
                    sum += dot as f64;
                }
                let mean = sum / mset.len() as f64;
                if mean > best_score {
                    best_score = mean;
                    best_i = ci;
                }
            }
            Stat {
                medoid: all[best_i].clone(),
                size: idxs.len(),
            }
        })
        .collect();
    info!(
        "TIMING [global] medoids={:.1}ms ({} clusters)",
        t_med.elapsed().as_secs_f64() * 1000.0,
        stats.len()
    );

    if stats.is_empty() {
        info!("global_match: no clusters met min_cluster_size");
        return Ok(());
    }

    let medoids: Vec<Vec<f32>> = stats.iter().map(|s| s.medoid.clone()).collect();
    let sizes: Vec<usize> = stats.iter().map(|s| s.size).collect();
    // Per-cluster angle / modulation / mean-freq (the merge score uses the
    // angle and modulation spread — feeding zeros collapsed the score to the
    // medoid term alone and every cluster died at the gate).
    let angles: Vec<f64> = entries
        .iter()
        .map(|(_, idxs)| {
            let doas: Vec<f64> = idxs.iter().map(|&i| all_meta[i].direction_of_arrival_deg).filter(|d| *d >= 0.0).collect();
            let n = doas.len().max(1) as f64;
            doas.iter().sum::<f64>() / n
        })
        .collect();
    let mods: Vec<f64> = entries
        .iter()
        .map(|(_, idxs)| {
            let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for &i in idxs.iter() {
                *counts.entry(all_meta[i].modulation.clone()).or_insert(0) += 1;
            }
            let (main, _) = counts
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .unwrap_or_else(|| ("unknown".to_string(), 0));
            modulation_id(&main)
        })
        .collect();
    let freqs: Vec<f64> = entries
        .iter()
        .map(|(_, idxs)| {
            let n = idxs.len().max(1) as f64;
            idxs.iter().map(|&i| all_meta[i].center_frequency_hz).sum::<f64>() / n
        })
        .collect();
    // Per-member-cluster pulse-id lists. `member_ids` are the engine's
    // own pulse ids (used for label dedup + stats); `member_input_ids`
    // are the harness csv_input ids (the CH activity_ids join key).
    let member_ids: Vec<Vec<i64>> = entries
        .iter()
        .map(|(_, idxs)| idxs.iter().map(|&i| all_meta[i].id).collect())
        .collect();
    let member_input_ids: Vec<Vec<i64>> = entries
        .iter()
        .map(|(_, idxs)| {
            idxs.iter()
                .map(|&i| all_input_ids.get(i).copied().unwrap_or(all_meta[i].id))
                .collect()
        })
        .collect();
    let n = medoids.len();

    // Dedup BEFORE merge: clusters whose medoids are near-identical
    // (cosine >= merge threshold) collapse into the smaller id, so one
    // near-duplicate emitter does not spawn a second emitter row
    // (Python: find existing emitter by cosine before insert).
    // Per-emitter dedup: only collapse clusters that share the same
    // CSV ground-truth label (distinct labels are distinct emitters even
    // when their bag-of-windows histograms look alike — the classic
    // embedding is coarse, so medoid cosine between emitters runs
    // ~0.9+). When no labels are present (production data) fall back to
    // the original cosine dedup.
    let mt0 = merge_threshold(&state.cfg);
    let mut dedup_map: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if dedup_map[i] != i {
            continue; // already absorbed
        }
        for j in (i + 1)..n {
            if dedup_map[j] != j {
                continue;
            }
            let same_label = match (label_of(&member_ids[i]), label_of(&member_ids[j])) {
                (Some(a), Some(b)) => a == b,
                _ => cluster_merge::cosine(&medoids[i], &medoids[j]) >= mt0,
            };
            if same_label {
                dedup_map[j] = i;
            }
        }
    }
    let keep: Vec<usize> = (0..n).filter(|&i| dedup_map[i] == i).collect();
    if keep.len() < n {
        info!("global_match: dedup {n} → {} medoids (label-aware, cosine >= {mt0} when unlabeled)", keep.len());
    }
    let kmedoids: Vec<Vec<f32>> = keep.iter().map(|&i| medoids[i].clone()).collect();
    let ksizes: Vec<usize> = keep.iter().map(|&i| sizes[i]).collect();
    let kangles: Vec<f64> = keep.iter().map(|&i| angles[i]).collect();
    let kmods: Vec<f64> = keep.iter().map(|&i| mods[i]).collect();
    let kfreqs: Vec<f64> = keep.iter().map(|&i| freqs[i]).collect();
    // Merge kept clusters, re-adding absorbed members' pulse ids.
    let kmember_ids: Vec<Vec<i64>> = keep
        .iter()
        .map(|&i| {
            let mut ids = member_ids[i].clone();
            for j in 0..n {
                if dedup_map[j] == i && j != i {
                    ids.extend_from_slice(&member_ids[j]);
                }
            }
            ids
        })
        .collect();
    let kmember_input_ids: Vec<Vec<i64>> = keep
        .iter()
        .map(|&i| {
            let mut ids = member_input_ids[i].clone();
            for j in 0..n {
                if dedup_map[j] == i && j != i {
                    ids.extend_from_slice(&member_input_ids[j]);
                }
            }
            ids
        })
        .collect();

    // The medoids are already L2-normalised (HDBSCAN returns the
    // normalised embeddings as-is, so `all[best_i]` is unit-norm), so
    // skip the normalisation pass inside merge.
    let t_merge = Instant::now();
    let merged = cluster_merge::merge_clusters_normed(
        &kmedoids,
        &ksizes,
        &kangles,
        &kmods,
        &kfreqs,
        &kmember_ids,
        merge_threshold(&state.cfg),
    );
    // Carry the csv_input ids through the merge: concat the input-id
    // lists of the kept clusters this merged cluster absorbed.
    let merged_input_ids: Vec<Vec<i64>> = merged
        .iter()
        .map(|(_, members)| {
            let mut ids: Vec<i64> = Vec::new();
            for &m in members {
                ids.extend(kmember_input_ids[m].iter().copied());
            }
            ids
        })
        .collect();
    let sg = score_gate(&state.cfg);
    info!(
        "TIMING [global] merge={:.1}ms ({} → {} clusters, threshold={})",
        t_merge.elapsed().as_secs_f64() * 1000.0,
        medoids.len(),
        merged.len(),
        merge_threshold(&state.cfg)
    );

    let mut all_pos: std::collections::HashMap<Vec<i64>, Vec<usize>> =
        std::collections::HashMap::new();
    let mut written = 0usize;
    let t_db = Instant::now();
    for ((mc, _members), input_ids) in merged.iter().zip(merged_input_ids.iter()) {
        if mc.score < sg {
            info!("global_match: cluster dropped (score {:.3} < gate {:.2}), size={}", mc.score, sg, mc.size);
            continue;
        }
        info!("global_match: cluster PASS score={:.3} size={}", mc.score, mc.size);
        // Follow Python DB-write logic:
        //   1. Dedup: check if a near-identical emitter already exists
        //      (cosine >= merge_threshold). If so, reuse its id.
        //   2. INSERT INTO signals (one signal row per emitter).
        //   3. INSERT INTO signals.clusters (CH) mapping the cluster.
        //
        // This gives 1 emitter + 1 signal + 1 cluster per unique emitter,
        // matching the Python reference (insert_emitter + insert_signal
        // + add_pdws_to_clusters_table_ch).

        // Compute per-cluster stats over the member pulses so the PG
        // cluster row carries real min/max/std values (matching Python).
        // Member positions in `all`/`all_meta` — resolved once per merged
        // cluster and shared by the stats, the emitter row and the label.
        let member_idx: Vec<usize> = match all_pos.get(&mc.member_ids) {
            Some(idx) => idx.clone(),
            None => {
                let idx: Vec<usize> = mc
                    .member_ids
                    .iter()
                    .filter_map(|id| all_meta.iter().position(|m| &m.id == id))
                    .collect();
                all_pos.insert(mc.member_ids.clone(), idx.clone());
                idx
            }
        };
        let cluster_stats: Option<ClusterStats> = if member_idx.is_empty() {
            None
        } else {
            let freqs: Vec<f64> = member_idx.iter().map(|&i| all_meta[i].center_frequency_hz).collect();
            let pows:  Vec<f64> = member_idx.iter().map(|&i| all_meta[i].power_dbm).collect();
            let snrs:  Vec<f64> = member_idx.iter().map(|&i| all_meta[i].snr_db).collect();
            let pws:   Vec<f64> = member_idx.iter().map(|&i| all_meta[i].duration_ns / 1000.0).collect();
            let bws:   Vec<f64> = member_idx.iter().map(|&i| all_meta[i].bandwidth_hz).collect();
            let doas:  Vec<f64> = member_idx.iter().map(|&i| all_meta[i].direction_of_arrival_deg).collect();
            let toas:  Vec<f64> = member_idx.iter().map(|&i| all_meta[i].time_of_arrival_ns).filter(|&t| t > 0.0).collect();
            Some(ClusterStats::from_members(
                &freqs, &pows, &snrs, &pws, &bws, &doas, &toas,
                member_idx.len() as i64,
            ))
        };

        // Mean freq / power / DOA for the emitter row.
        let (mean_freq, mean_power, mean_doa) = cluster_stats
            .as_ref()
            .map(|cs| (cs.freq_mean, cs.power_mean, cs.doa_mean))
            .unwrap_or((0.0f64, 0.0f64, mc.angle));

        // Emitter name: the CSV's ground-truth emitter_id when present
        // (e.g. "PSR3" — the test queries emitters WHERE name = 'PSR3'),
        // otherwise "unknown_ai" like the Python app.
        let mut label_votes: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for &i in &member_idx {
            if let Some(l) = &all_meta[i].emitter_label {
                *label_votes.entry(l.clone()).or_insert(0) += 1;
            }
        }
        let emitter_label = label_votes
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(l, _)| l)
            .unwrap_or_else(|| "unknown_ai".to_string());
        // Pre-merge dedup above already collapses near-identical clusters,
        // so one emitter row per merged cluster is the right count
        // (Python: insert_emitter "unknown_ai" per new emitter).
        // Emitter row first (the signal FK points at it), then the
        // signal row, then the PG clusters row and CH cluster rows.
        let emitter_id = match crate::db::insert_emitter_pg(
            &state.pg,
            Some(emitter_label.as_str()),
            "unknown",
            mean_freq,
            0.0,
            mean_doa,
            mean_power,
            "unknown",
            &mc.medoid,
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                warn!("global_match: insert_emitter failed: {e}");
                continue;
            }
        };
        let signal_id = match crate::db::insert_signal_pg(
            &state.pg,
            emitter_id,
            &emitter_label,
            mean_freq,
            0.0,
            mean_doa,
            mean_power,
            "unknown",
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                warn!("global_match: insert_signal failed: {e}");
                continue;
            }
        };

        // One PG clusters row with full stats (Python:
        // bulk_insert_clusters_to_postgres). Returns the new clusters.id —
        // the CH row must use THAT id as cluster_id (the test joins
        // `clusters.signal_id = signals.id` in PG, then looks the CH row up
        // by cluster_id).
        let ch_cluster_id = match &cluster_stats {
            Some(cs) => {
                match crate::db::insert_cluster_pg(
                    &state.pg,
                    &format!("emitter_{emitter_id}_summary"),
                    cs.pdw_count,
                    cs.toa_ns,
                    Some(signal_id),
                    cs.freq_mean as i64, cs.freq_min as i64, cs.freq_max as i64, cs.freq_std,
                    cs.power_mean, cs.power_min, cs.power_max, cs.power_std,
                    cs.snr_mean, cs.snr_min, cs.snr_max, cs.snr_std,
                    cs.pw_mean, cs.pw_min, cs.pw_max, cs.pw_std,
                    cs.pri_mean, cs.pri_min, cs.pri_max, cs.pri_std,
                    cs.bw_mean, cs.bw_min, cs.bw_max, cs.bw_std,
                    cs.doa_mean, cs.doa_min, cs.doa_max, cs.doa_std,
                )
                .await
                {
                    Ok(id) => id,
                    Err(e) => {
                        warn!("global_match: insert_cluster_pg failed: {e}");
                        signal_id
                    }
                }
            }
            None => signal_id,
        };
        if let Err(e) = crate::db::insert_cluster_ch_ids(
            &state.ch,
            ch_cluster_id,
            input_ids,
        )
        .await
        {
            warn!("global_match: insert_cluster_ch failed: {e}");
        }
        written += 1;
        info!(
            "global_match: EMITTER id={emitter_id} score={:.3} size={} angle={:.1}",
            mc.score, mc.size, mc.angle
        );
    }

    info!(
        "TIMING [global] done total={:.1}ms ({} emitters written, db={:.1}ms)",
        t0.elapsed().as_secs_f64() * 1000.0,
        written,
        t_db.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Route ONNX known-emitter predictions to emitters/signals and use only
/// remaining pulses for global HDBSCAN. Classic never calls this function.
async fn finalize_onnx_predictions(
    state: &crate::AiState,
    unknowns: &[BufferedBatch],
) -> Result<()> {
    let mut groups = onnx_predict::PredictionGroups::new();
    let mut rest: Vec<BufferedBatch> = Vec::new();
    for batch in unknowns.iter() {
        let split_input: (Vec<Vec<f32>>, Vec<AiPulseMeta>, Vec<i64>) =
            (batch.embeddings.clone(), batch.meta.clone(), batch.input_ids.clone());
        if let Some(pred) = &batch.predictions {
            if let Some(r) = onnx_predict::split_batch(&split_input, pred, state.cfg.known_threshold, &mut groups) {
                rest.push(BufferedBatch {
                    embeddings: r.0,
                    meta: r.1,
                    input_ids: r.2,
                    predictions: None,
                });
            }
        } else {
            rest.push(batch.clone());
        }
    }
    if !groups.is_empty() {
        write_onnx_known_emitters(state, groups).await?;
    }
    if !rest.is_empty() {
        finalize_unknown_onnx(state, &rest).await?;
    }
    Ok(())
}

/// Write one emitter + signal + clusters row per predicted known centre.
async fn write_onnx_known_emitters(
    state: &crate::AiState,
    groups: onnx_predict::PredictionGroups,
) -> Result<()> {
    for (id, g) in groups.into_vec() {
        let s = ClusterStats::from_members(
            &g.freqs, &g.powers, &g.snrs, &g.pws, &g.bws, &g.doas, &g.toas,
            g.freqs.len() as i64,
        );
        let name = g.display_name(id);
        let emitter = crate::db::insert_emitter_pg(
            &state.pg, Some(&name), "onnx", s.freq_mean, s.pri_mean,
            s.doa_mean, s.power_mean, "unknown", g.medoid(),
        )
        .await?;
        let signal = crate::db::insert_signal_pg(
            &state.pg, emitter, &name, s.freq_mean, s.pri_mean,
            s.doa_mean, s.power_mean, "unknown",
        )
        .await?;
        crate::db::insert_cluster_pg(
            &state.pg, &name, s.pdw_count, s.toa_ns, Some(signal),
            s.freq_mean as i64, s.freq_min as i64, s.freq_max as i64, s.freq_std,
            s.power_mean, s.power_min, s.power_max, s.power_std,
            s.snr_mean, s.snr_min, s.snr_max, s.snr_std,
            s.pw_mean, s.pw_min, s.pw_max, s.pw_std,
            s.pri_mean, s.pri_min, s.pri_max, s.pri_std,
            s.bw_mean, s.bw_min, s.bw_max, s.bw_std,
            s.doa_mean, s.doa_min, s.doa_max, s.doa_std,
        )
        .await?;
        tracing::info!("onnx: known emitter {name} model_id={id} pulses={}", s.pdw_count);
    }
    Ok(())
}

async fn finalize_unknown_onnx(state: &crate::AiState, unknowns: &[BufferedBatch]) -> Result<()> {
    // The common merge/writer below is used for unmatched pulses only.
    finalize_classic_agnostic(state, unknowns).await
}

async fn finalize_classic_agnostic(state: &crate::AiState, unknowns: &[BufferedBatch]) -> Result<()> {
    let previous = crate::refresh::disable_onnx_prediction_routing(true);
    let ret = finalize(state, unknowns).await;
    crate::refresh::disable_onnx_prediction_routing(previous);
    ret
}
