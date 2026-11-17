/// Classic (non-neural) embedding for pulse features.
///
/// Replaces the ONNX transformer when `EMBED=classic` (or when no ONNX
/// model/lib is configured). The embedding is a deterministic
/// bag-of-windows over the 458-dim feature vector:
///
///   1. split the 458 dims into 58 buckets of 8 (last 6) and
///      L2-normalise each bucket;
///   2. count the 40 surrounding windows (same windowing as the feature
///      extractor) — a pulse "votes" into one 8-dim bucket per window
///      member;
///   3. normalise the accumulated 360-dim histogram (45 buckets x 8) to
///      unit length.
///
/// Why this shape: pulses from the same emitter share frequency, power,
/// PRI and TOA statistics, so their histograms are near-identical
/// (high cosine); different emitters land in different buckets
/// (low cosine). HDBSCAN over these embeddings therefore separates
/// emitters, and the cluster's voted ground-truth label names the
/// emitter — matching what the Python test suite expects.

/// Classic output dim for a feature width of `d`: `d.div_ceil(8) * 8`.
/// (458-dim features -> 58 buckets -> 464.)
pub const fn out_dim(d: usize) -> usize {
    d.div_ceil(8) * 8
}
/// Classic output dim for the 458-dim model input (see `features::FEATURE_DIM`).
pub const OUT_DIM: usize = out_dim(crate::features::FEATURE_DIM);

/// Build the classic embedding for `n` pulses from their 458-dim
/// feature rows. `pulses` carries the raw pulse fields used for the
/// 40-pulse windowing (same window as the feature extractor).
pub fn embed(features: &[Vec<f64>], window: usize) -> Vec<Vec<f32>> {
    use rayon::prelude::*;

    let radius = window / 2;
    let n = features.len();
    let dim = if n > 0 { features[0].len() } else { 0 };
    if dim == 0 {
        return vec![Vec::new(); n];
    }
    // Bucket layout: ceil(dim/8) buckets of 8 dims.
    let n_buckets = dim.div_ceil(8);
    let out_dim = n_buckets * 8;

    let rows = features.par_iter().enumerate().map(|(i, row)| {
        let mut hist: Vec<f32> = vec![0.0f32; out_dim];
        for w in (i.saturating_sub(radius)).min(n - 1)
            ..=
            ((i + radius).min(n - 1))
        {
            let r = &features[w];
            for b in 0..n_buckets {
                let start = b * 8;
                let end = (start + 8).min(dim);
                let mut norm = 0.0f64;
                for v in &r[start..end] {
                    norm += v * v;
                }
                norm = norm.sqrt().max(1e-12);
                for (k, v) in r[start..end].iter().enumerate() {
                    hist[start + k] += (*v / norm) as f32;
                }
                // zero the pad slots so buckets keep equal weight
                if end < start + 8 {
                    for k in end - start..8 {
                        hist[start + k] = 0.0;
                    }
                }
            }
        }
        // L2 normalise the histogram.
        let mut norm = 0.0f64;
        for v in &hist {
            norm += *v as f64 * *v as f64;
        }
        norm = norm.sqrt().max(1e-12);
        for v in hist.iter_mut() {
            *v = (*v / norm as f32) as f32;
        }
        hist
    });
    rows.collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_emitter_high_cosine() {
        // Two synthetic "emitters": one around a fixed feature row, one
        // very different. Same-emitter histograms must cluster tight.
        let base_a: Vec<f64> = [0.21, -0.3, 0.9, 0.1, 0.44, 0.02, -0.1, 0.7].iter().cycle().take(458).copied().collect();
        let base_b: Vec<f64> = [-0.8, 0.5, 0.01, 0.33, -0.2, 0.66, 0.4, -0.1].iter().cycle().take(458).copied().collect();
        let row_a = base_a.clone();
        let row_b = base_b.clone();
        let feats: Vec<Vec<f64>> = std::iter::repeat(row_a).take(20).collect();
        let em_a = embed(&feats, 40);
        let feats_b: Vec<Vec<f64>> = std::iter::repeat(row_b).take(20).collect();
        let em_b = embed(&feats_b, 40);

        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };
        let same = cosine(&em_a[0], &em_a[19]);
        let diff = cosine(&em_a[0], &em_b[0]);
        assert!(same > 0.99, "same emitter cosine {same}");
        assert!(diff < 0.9, "different emitter cosine {diff}");
        assert_eq!(em_a[0].len(), OUT_DIM);
    }

    #[test]
    fn empty_input() {
        let out = embed(&[], 40);
        assert_eq!(out.len(), 0);
    }
}
