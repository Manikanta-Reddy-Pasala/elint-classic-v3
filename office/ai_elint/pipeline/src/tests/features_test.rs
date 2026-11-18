/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::features::*;

    /// Serial reference implementation (single-threaded, original logic).
    fn generate_features_serial(pulses: &[Pulse]) -> Vec<Vec<f64>> {
        let n = pulses.len();
        if n == 0 {
            return Vec::new();
        }
        // NOTE: must stay in lockstep with features.rs — the production
        // ranges are logspace(0.1, 300.0, PW) and logspace(10.0, 5000.0, TOA).
        let pw_centers = logspace(0.1, 300.0, NUM_PW_CENTERS);
        let pw_sigma = adaptive_sigma(&pw_centers);
        let pw_denom: Vec<f64> = pw_sigma.iter().map(|s| 2.0 * s * s).collect();
        let toa_centers = logspace(10.0, 5000.0, NUM_TOA_CENTERS);
        let toa_sigma = adaptive_sigma(&toa_centers);
        let toa_denom: Vec<f64> = toa_sigma.iter().map(|s| 2.0 * s * s).collect();
        let radius = WINDOW_SIZE / 2;

        pulses.iter().enumerate().map(|(i, p)| {
            let mut feat = Vec::with_capacity(FEATURE_DIM);
            feat.push((p.freq - 50.0) / 10950.0);
            feat.push((p.power + 100.0) / 100.0);
            let clamped_pw = p.raw_pw.clamp(0.1, 300.0);
            for c in 0..NUM_PW_CENTERS {
                let d = clamped_pw - pw_centers[c];
                feat.push((-d * d / pw_denom[c]).exp());
            }
            let center = i.clamp(radius, n.saturating_sub(1).max(radius));
            let center_i = center as i64;
            let mut toa_sum = [0f64; NUM_TOA_CENTERS];
            let toa_i = p.raw_toa;
            for offset in (-(radius as i64))..=(radius as i64) {
                let j = (center_i + offset).clamp(0, (n - 1) as i64) as usize;
                let toa_j = pulses[j].raw_toa;
                for c in 0..NUM_TOA_CENTERS {
                    let d = (toa_i - toa_j) - toa_centers[c];
                    toa_sum[c] += (-d * d / toa_denom[c]).exp();
                }
            }
            feat.extend_from_slice(&toa_sum);
            feat
        }).collect()
    }

    #[test]
    fn parallel_matches_serial() {
        let pulses: Vec<Pulse> = (0..1024)
            .map(|i| Pulse {
                freq: 2.0 + (i as f64 % 100.0) / 10.0,
                power: -50.0 + (i as f64 % 80.0),
                raw_pw: (i as f64 % 50.0) + 1.0,
                raw_toa: 50.0 + (i as f64 % 4000.0),
            })
            .collect();

        let serial = generate_features_serial(&pulses);
        let parallel = generate_features(&pulses);

        assert_eq!(serial.len(), parallel.len());
        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert_eq!(s.len(), p.len());
            for (a, b) in s.iter().zip(p.iter()) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "mismatch: serial={a} parallel={b}"
                );
            }
        }
    }

    #[test]
    fn small_batch_matches() {
        let pulses: Vec<Pulse> = (0..7)
            .map(|i| Pulse {
                freq: 5.0,
                power: -20.0,
                raw_pw: 10.0,
                raw_toa: 100.0 + i as f64,
            })
            .collect();
        let serial = generate_features_serial(&pulses);
        let parallel = generate_features(&pulses);
        assert_eq!(serial.len(), parallel.len());
        for (s, p) in serial.iter().zip(parallel.iter()) {
            for (a, b) in s.iter().zip(p.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }

}
