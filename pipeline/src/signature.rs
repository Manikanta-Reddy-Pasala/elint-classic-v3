//! Measured signature of one emitter group (report output for unknown and
//! known radars alike). Built from measured PDW fields only — the CSV's
//! ground-truth `emitter_id` / generator PRI are never read.
//!
//! Units: MHz, us, deg.
//! - `pri_fund_us`: shortest well-populated pulse gap (see [`pri_fund_us`]),
//!   so a staggered or jittered train reports its base interval, not a harmonic.
//! - `pri_mode_us`: most frequent 1-us gap.
//! - `doa_spread_deg`: narrowest arc holding 80 % of the angles (circular;
//!   near 288+ = omni / no direction finding).
use std::collections::{BTreeMap, HashMap};

use crate::classic_v2::freq_mode;
use crate::onnx_flow::FlowPulse;

#[derive(serde::Serialize, Debug, Clone)]
pub struct Signature {
    pub n: usize,
    pub fmode: &'static str,
    pub f_q5_mhz: f64,
    pub f_q95_mhz: f64,
    pub pw_med_us: f64,
    pub bw_med_mhz: f64,
    pub modulation: String,
    pub mod_share: f64,
    pub pri_mode_us: f64,
    pub pri_fund_us: f64,
    pub doa_spread_deg: f64,
}

/// 1-us gap histogram of a TOA-sorted train (ns); gaps below `min_gap_us`
/// (the same pulse seen twice) are skipped.
fn gap_hist(toas: &[f64], min_gap_us: f64) -> BTreeMap<i64, usize> {
    let mut h: BTreeMap<i64, usize> = BTreeMap::new();
    for w in toas.windows(2) {
        let d = (w[1] - w[0]) / 1e3;
        if d >= min_gap_us.max(0.5) {
            *h.entry(d.round() as i64).or_default() += 1;
        }
    }
    h
}

/// PRI fundamental (us) of a TOA-sorted train (ns): the shortest gap whose
/// +-1 us neighbourhood holds >= 25 % of the busiest one. NaN: no gaps.
pub fn pri_fund_us(toas: &[f64], min_gap_us: f64) -> f64 {
    let h = gap_hist(toas, min_gap_us);
    let near = |k: i64| (k - 1..=k + 1).map(|j| *h.get(&j).unwrap_or(&0)).sum::<usize>();
    let top = h.keys().map(|k| near(*k)).max().unwrap_or(0);
    h.keys()
        .find(|k| top > 0 && near(**k) as f64 >= 0.25 * top as f64)
        .map(|k| *k as f64)
        .unwrap_or(f64::NAN)
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

pub fn signature(ps: &[&FlowPulse], min_gap_us: f64) -> Signature {
    let n = ps.len();
    let mut c: HashMap<String, usize> = HashMap::new();
    for p in ps {
        *c.entry(p.modulation.trim().to_ascii_lowercase()).or_default() += 1;
    }
    let (modulation, mc) = c.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or_default();
    let mut t: Vec<f64> = ps.iter().map(|p| p.toa_ns).filter(|x| x.is_finite()).collect();
    t.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let h = gap_hist(&t, min_gap_us);
    let pri_mode_us = h.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map(|x| *x.0 as f64).unwrap_or(f64::NAN);
    let mut d: Vec<f64> = ps.iter().map(|p| p.doa_deg).filter(|x| x.is_finite()).map(|x| x.rem_euclid(360.0)).collect();
    d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let doa_spread_deg = if d.len() < 5 {
        f64::NAN
    } else {
        let m = d.len();
        let k = ((m as f64) * 0.8).ceil() as usize;
        (0..m)
            .map(|i| {
                let j = i + k - 1;
                if j < m { d[j] - d[i] } else { d[j - m] + 360.0 - d[i] }
            })
            .fold(360.0, f64::min)
    };
    let f: Vec<f64> = ps.iter().map(|p| p.freq_hz / 1e6).collect();
    Signature {
        n,
        fmode: if n == 0 { "unknown" } else { freq_mode(ps) },
        f_q5_mhz: quantile(f.clone(), 0.05),
        f_q95_mhz: quantile(f, 0.95),
        pw_med_us: quantile(ps.iter().map(|p| p.duration_ns / 1e3).collect(), 0.5),
        bw_med_mhz: quantile(ps.iter().map(|p| p.bandwidth_hz / 1e6).collect(), 0.5),
        modulation,
        mod_share: if n == 0 { 0.0 } else { mc as f64 / n as f64 },
        pri_mode_us,
        pri_fund_us: pri_fund_us(&t, min_gap_us),
        doa_spread_deg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train(pri_us: &[f64], n: usize, doa: impl Fn(usize) -> f64) -> Vec<FlowPulse> {
        let mut t = 0.0;
        (0..n)
            .map(|i| {
                t += pri_us[i % pri_us.len()];
                FlowPulse {
                    input_id: 0,
                    freq_hz: 3100e6,
                    power_dbm: -70.0,
                    bandwidth_hz: 1e6,
                    duration_ns: 10_000.0,
                    doa_deg: doa(i),
                    snr_db: 20.0,
                    toa_ns: t * 1e3,
                    modulation: "lfm_up".into(),
                }
            })
            .collect()
    }

    fn ts(v: &[FlowPulse]) -> Vec<f64> {
        v.iter().map(|p| p.toa_ns).collect()
    }

    #[test]
    fn pri_fundamental_of_fixed_and_staggered_trains() {
        assert_eq!(pri_fund_us(&ts(&train(&[294.0], 500, |_| 0.0)), 1.0), 294.0);
        // stagger 300/350/260: the base (shortest) interval, not the 910 us frame
        assert_eq!(pri_fund_us(&ts(&train(&[300.0, 350.0, 260.0], 600, |_| 0.0)), 1.0), 260.0);
        assert!(pri_fund_us(&[1.0], 1.0).is_nan());
    }

    #[test]
    fn signature_reports_measured_values() {
        let v = train(&[500.0], 400, |_| 40.0);
        let s = signature(&v.iter().collect::<Vec<_>>(), 1.0);
        assert_eq!((s.n, s.fmode, s.pri_fund_us, s.pri_mode_us), (400, "fixed", 500.0, 500.0));
        assert_eq!(s.modulation, "lfm_up");
        assert!((s.pw_med_us - 10.0).abs() < 1e-9 && (s.bw_med_mhz - 1.0).abs() < 1e-9);
        assert!(s.doa_spread_deg < 1.0);
        // angles around north (350..10 deg): circular spread ~20, not 340
        let w = train(&[500.0], 105, |i| (350.0 + (i % 21) as f64) % 360.0);
        let s = signature(&w.iter().collect::<Vec<_>>(), 1.0);
        assert!(s.doa_spread_deg <= 20.0, "{}", s.doa_spread_deg);
        assert_eq!(signature(&[], 1.0).n, 0);
    }
}
