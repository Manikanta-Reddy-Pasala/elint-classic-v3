/// Known-emitter matching — exact POC `_process_known_vector_matches` port.
///
/// For each embedded pulse:
///   1. Stack all known medoids → (K x 1024) matrix
///   2. dot(pulse_embedding, centers.T) → similarity matrix (pre-normalised = cosine)
///   3. argmax per pulse → best emitter + score
///   4. Gate 1: score >= 0.80 (UNKNOWN_DOT_PRODUCT_THRESHOLD)
///   5. Gate 2: |pulse_doa - emitter_angle| <= 40 deg (AoA gate)
///   6. PASS → known (update medoid via EMA), FAIL → unknown
use std::sync::Arc;

use crate::cache::{Cache, EmitterTrack};

/// Result of matching one pulse against the known table.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub matched: bool,
    pub signal_id: i64,
    pub score: f32,
    pub emitter_angle: f64,
}

/// Match a batch of embedded pulses against known tracks.
///
/// `embeddings`:  n x 1024  (L2-normalised f32)
/// `doas`:        n
/// `tracks`:      K known emitter tracks from Dragonfly
/// `threshold`:   0.80
/// `angle_gate`:  40.0
/// `alpha`:       EMA 0.1
pub fn match_known(
    embeddings: &[Vec<f32>],
    doas: &[f64],
    tracks: &[EmitterTrack],
    threshold: f32,
    angle_gate: f32,
) -> Vec<MatchResult> {
    let n = embeddings.len();
    let k = tracks.len();
    if n == 0 {
        return Vec::new();
    }
    // Embedding dimension is read from the data, not hardcoded.
    let dim = embeddings[0].len();

    if k == 0 {
        return (0..n)
            .map(|_| MatchResult {
                matched: false,
                signal_id: -1,
                score: 0.0,
                emitter_angle: 0.0,
            })
            .collect();
    }

    // Build (K x dim) centers matrix — row-major for cache-friendly dot products.
    let mut centers = vec![0f32; k * dim];
    for (i, t) in tracks.iter().enumerate() {
        for (j, &v) in t.vector.iter().enumerate().take(dim) {
            centers[i * dim + j] = v;
        }
    }

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let emb = &embeddings[i];
        let mut best_score: f32 = 0.0;
        let mut best_idx: usize = 0;

        // Vectorised dot product: emb · centers[j] for all j.
        for j in 0..k {
            let mut dot: f32 = 0.0;
            for d in 0..dim {
                dot += emb[d] * centers[j * dim + d];
            }
            if dot > best_score {
                best_score = dot;
                best_idx = j;
            }
        }

        let best_track = &tracks[best_idx];
        let angle_diff = angle_diff_deg(doas[i], best_track.angle);
        let passed = best_score >= threshold && angle_diff <= angle_gate as f64;

        results.push(MatchResult {
            matched: passed,
            signal_id: if passed { best_track.signal_id } else { -1 },
            score: best_score,
            emitter_angle: best_track.angle,
        });
    }
    results
}

/// EMA-update the medoid for a matched signal (POC feedback loop).
pub fn apply_ema(cache: &Arc<Cache>, signal_id: i64, embedding: &[f32], alpha: f32) {
    cache.ema_update_track(signal_id, embedding, alpha);
}

/// Circular angle difference in degrees (0..180).
pub fn angle_diff_deg(a: f64, b: f64) -> f64 {
    let mut d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        d = 360.0 - d;
    }
    d
}
