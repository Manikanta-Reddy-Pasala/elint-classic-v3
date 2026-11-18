/// Feature extraction — exact Rust port of POC `create_features.py`.
///
/// Input columns per pulse (the caller applies the Python unit
/// conversions — see `lib.rs`): freq in GHz (Hz/1e7), power in dBm,
/// raw_pw in µs (ns/1000), raw_toa in µs (ns/1000).
/// Output: 458-dim feature vector per pulse.
///   [0]       freq_scaled  = (freq - 50) / 10950
///   [1]       power_scaled = (power + 100) / 100
///   [2..201]  pw_rbf       = RBF against 200 log-spaced centers [0.1, 300]
///   [202..357] toa_rbf     = sum over 40-pulse window of RBF against
///                             256 log-spaced centers [10, 5000]
///
/// Parallelisation: ONE rayon task per pulse. The earlier version also
/// split the inner 40-row x 256-centre TOA loop across rayon, which made
/// ~10k extra tasks PER PULSE (~200M for a 20k batch) plus two ~160 KB
/// per-pulse allocations — the task setup and malloc churn cost more than
/// the exp() calls they parallelised. The outer par alone saturates the
/// cores, and the inner loop reads the 40 window rows straight from the
/// pulse slice (no precompute, no per-pulse allocation).

pub const WINDOW_SIZE: usize = 40;
pub const NUM_TOA_CENTERS: usize = 256;
/// Must match the Python reference (`create_features.py`):
/// `num_pw_centers = 200`, `pw_lags_raw = logspace(0.1, 300)`.
/// 2 + 200 + 256 = 458-dim feature vector (the model's real input width).
pub const NUM_PW_CENTERS: usize = 200;
/// Total feature width: freq(1) + power(1) + pw_rbf(200) + toa_rbf(256).
pub const FEATURE_DIM: usize = 2 + NUM_PW_CENTERS + NUM_TOA_CENTERS; // 458

pub fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let llo = lo.log10();
    let lhi = hi.log10();
    let step = (lhi - llo) / (n - 1) as f64;
    (0..n).map(|i| 10f64.powf(llo + i as f64 * step)).collect()
}

pub fn adaptive_sigma(centers: &[f64]) -> Vec<f64> {
    let n = centers.len();
    let mut sigma = Vec::with_capacity(n);
    for i in 0..n {
        let prev = if i > 0 { centers[i] - centers[i - 1] } else { centers[1] - centers[0] };
        let next = if i + 1 < n { centers[i + 1] - centers[i] } else { prev };
        sigma.push(prev.min(next) * 2.0 / 3.0);
    }
    sigma
}

#[derive(Debug, Clone)]
pub struct Pulse {
    pub freq: f64,
    pub power: f64,
    pub raw_pw: f64,
    pub raw_toa: f64,
}

/// Build the full 458-dim feature matrix for a batch of pulses.
///
/// One rayon task per pulse; the per-pulse 40-row TOA window is read
/// directly from the pulse slice. Addition order per TOA centre matches
/// the serial reference (rows in -radius..=radius order, then centres),
/// so the output is bit-identical.
pub fn generate_features(pulses: &[Pulse]) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let n = pulses.len();
    if n == 0 {
        return Vec::new();
    }

    let pw_centers = logspace(0.1, 300.0, NUM_PW_CENTERS);
    let pw_sigma = adaptive_sigma(&pw_centers);
    let pw_denom: Vec<f64> = pw_sigma.iter().map(|s| 2.0 * s * s).collect();

    let toa_centers = logspace(10.0, 5000.0, NUM_TOA_CENTERS);
    let toa_sigma = adaptive_sigma(&toa_centers);
    let toa_denom: Vec<f64> = toa_sigma.iter().map(|s| 2.0 * s * s).collect();

    let radius = WINDOW_SIZE / 2;
    let n_sub1 = (n - 1) as i64;
    // Upper bound >= radius even when n < WINDOW_SIZE/2 + 1 (small batches).
    let center_hi = n_sub1.max(radius as i64);

    pulses
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let mut feat = Vec::with_capacity(FEATURE_DIM);
            feat.push((p.freq - 50.0) / 10950.0);
            feat.push((p.power + 100.0) / 100.0);

            // PW block (100 centres, serial — small).
            let clamped_pw = p.raw_pw.clamp(0.1, 300.0);
            for c in 0..NUM_PW_CENTERS {
                let d = clamped_pw - pw_centers[c];
                feat.push((-d * d / pw_denom[c]).exp());
            }

            // TOA block: 40 window rows x 256 centres. Read the window
            // straight from the pulse slice (no per-pulse allocation).
            let toa_i = p.raw_toa;
            let center = (i as i64).clamp(radius as i64, center_hi);
            let mut toa_sum = [0.0f64; NUM_TOA_CENTERS];
            for offset in (-(radius as i64))..=(radius as i64) {
                let j = (center + offset).clamp(0, n_sub1) as usize;
                let toa_j = pulses[j].raw_toa;
                for c in 0..NUM_TOA_CENTERS {
                    let d = (toa_i - toa_j) - toa_centers[c];
                    toa_sum[c] += (-d * d / toa_denom[c]).exp();
                }
            }
            feat.extend_from_slice(&toa_sum);
            feat
        })
        .collect()
}

/// L2-normalise a feature vector in place.
pub fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_dims_match_python() {
        // 10 pulses at distinct TOAs.
        let pulses: Vec<Pulse> = (0..10)
            .map(|i| Pulse {
                freq: 2749.0,
                power: -60.0,
                raw_pw: 1.3,
                raw_toa: i as f64 * 100.0,
            })
            .collect();
        let feats = generate_features(&pulses);
        assert_eq!(feats.len(), 10);
        assert_eq!(feats[0].len(), 458);
        // freq_scaled = (2749 - 50)/10950, power_scaled = (-60+100)/100.
        assert!((feats[0][0] - (2749.0 - 50.0) / 10950.0).abs() < 1e-12);
        assert!((feats[0][1] - 0.4).abs() < 1e-12);
    }
}
