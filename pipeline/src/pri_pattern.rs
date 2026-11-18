//! PRI pattern of one emitter's pulse train, from TOA only (report label;
//! it never changes grouping).
//!
//! Steps:
//!  1. Sort TOAs, drop duplicate pulses (< `DUP_US` after the previous kept
//!     pulse: the same pulse seen by two receivers).
//!  2. Gaps between kept pulses. Sorted gaps are split into clusters where
//!     two neighbours differ by more than `tol` (relative, adaptive to the
//!     number of gaps). Clusters with < 1 % of gaps are ignored.
//!  3. Primary levels vs harmonics: clusters are taken from the smallest up.
//!     A cluster is a harmonic (missing pulse) when its value is a sum of
//!     2-4 primary levels AND it rarely follows itself AND it holds fewer
//!     gaps than the primaries. Otherwise it is a primary level.
//!  4. Type:
//!     - smallest primary is wide (spread > `WIDE`): continuous PRI.
//!       Consecutive gaps close to each other (small step vs spread) =
//!       `sliding`, otherwise `jitter` (jitter % = half-width of the spread).
//!     - one narrow primary: `constant`.
//!     - several narrow primaries: a gap usually followed by the same level
//!       = `dwell_switch`; else the shortest repeat k of the level sequence =
//!       `stagger` (elements in order, starting at the smallest; period = sum).
//!     - anything else: `complex`.
//!  5. Missing pulses: gaps outside the primary levels, counted as whole
//!     multiples of the mean PRI.
use serde::Serialize;

/// Two pulses closer than this (µs) are one pulse seen twice.
pub const DUP_US: f64 = 1.0;
/// Fewer gaps than this: no pattern (`too_few`).
pub const MIN_GAPS: usize = 16;
/// Cluster half-width (relative) above which a level is a continuous spread.
pub const WIDE: f64 = 0.005;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Stagger {
    pub period_us: f64,
    /// one period, in pulse order, starting at the smallest element
    pub elements_us: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Dwell {
    /// dwell PRIs, smallest first
    pub pris_us: Vec<f64>,
    /// median number of gaps (pulses) per dwell
    pub dwell_pulses: f64,
    /// median dwell length in time
    pub dwell_us: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Sliding {
    pub min_us: f64,
    pub max_us: f64,
    /// median PRI change from one pulse to the next
    pub step_us: f64,
    /// pulses to cross the whole range
    pub ramp_pulses: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PriPattern {
    /// constant | jitter | stagger | dwell_switch | sliding | complex | too_few
    pub kind: &'static str,
    /// fundamental (constant), mean (jitter / sliding / dwell), period / k (stagger)
    pub pri_us: f64,
    /// half-width of the PRI spread around the mean, % (jitter; ~0 for sharp levels)
    pub jitter_pct: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stagger: Option<Stagger>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dwell: Option<Dwell>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding: Option<Sliding>,
    /// primary gap levels (µs), smallest first
    pub levels_us: Vec<f64>,
    /// gap levels explained as sums of primaries (missing pulses), µs
    pub harmonics_us: Vec<f64>,
    /// estimated share of pulses not received, %
    pub missing_pct: f64,
    pub dup_skipped: usize,
    pub n_gaps: usize,
}

impl PriPattern {
    fn empty(kind: &'static str, dup: usize, n_gaps: usize) -> Self {
        PriPattern {
            kind,
            pri_us: f64::NAN,
            jitter_pct: f64::NAN,
            stagger: None,
            dwell: None,
            sliding: None,
            levels_us: vec![],
            harmonics_us: vec![],
            missing_pct: 0.0,
            dup_skipped: dup,
            n_gaps,
        }
    }
}

#[derive(Debug, Clone)]
struct Cluster {
    lo: f64,
    hi: f64,
    med: f64,
    n: usize,
    /// (q95 - q05) / (2 * median)
    w: f64,
}

fn q_sorted(s: &[f64], f: f64) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let x = (s.len() - 1) as f64 * f;
    let (i, fr) = (x.floor() as usize, x - x.floor());
    if i + 1 < s.len() {
        s[i] + (s[i + 1] - s[i]) * fr
    } else {
        s[i]
    }
}

fn sorted(v: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut s: Vec<f64> = v.collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

fn clusters(g: &[f64], tol: f64) -> Vec<Cluster> {
    let s = sorted(g.iter().copied());
    let mut out = Vec::new();
    let mut a = 0;
    for i in 1..=s.len() {
        if i == s.len() || s[i] > s[i - 1] * (1.0 + tol) {
            let seg = &s[a..i];
            let med = q_sorted(seg, 0.5);
            out.push(Cluster {
                lo: seg[0],
                hi: seg[seg.len() - 1],
                med,
                n: seg.len(),
                w: (q_sorted(seg, 0.95) - q_sorted(seg, 0.05)) / (2.0 * med),
            });
            a = i;
        }
    }
    out
}

/// Is `x` a sum of 2..=4 values from `prim` (repetition allowed) within `tol`?
fn is_sum(x: f64, prim: &[f64], tol: f64) -> bool {
    fn rec(x: f64, prim: &[f64], start: usize, k: usize, acc: f64, tol: f64) -> bool {
        if k >= 2 && (acc - x).abs() <= tol * x {
            return true;
        }
        if k == 4 || acc > x * (1.0 + tol) {
            return false;
        }
        (start..prim.len()).any(|j| rec(x, prim, j, k + 1, acc + prim[j], tol))
    }
    rec(x, prim, 0, 0, 0.0, tol)
}

/// Pattern of one pulse train. `toas_ns` need not be sorted.
pub fn analyze(toas_ns: &[f64]) -> PriPattern {
    let t = sorted(toas_ns.iter().filter(|x| x.is_finite()).map(|x| x / 1e3));
    let mut kept: Vec<f64> = Vec::with_capacity(t.len());
    let mut dup = 0usize;
    for x in t {
        if let Some(&l) = kept.last() {
            if x - l < DUP_US {
                dup += 1;
                continue;
            }
        }
        kept.push(x);
    }
    let g: Vec<f64> = kept.windows(2).map(|w| w[1] - w[0]).collect();
    let n = g.len();
    if n < MIN_GAPS {
        return PriPattern::empty("too_few", dup, n);
    }
    // neighbour split tolerance: 0.3 %, wider for short trains so a jittered
    // train with few gaps is not cut into many pieces
    let tol = (3.0 / n as f64).clamp(0.003, 0.02);
    let all = clusters(&g, tol);
    let min_n = ((n as f64) * 0.01).ceil().max(3.0) as usize;
    let sig: Vec<Cluster> = all.into_iter().filter(|c| c.n >= min_n).collect();
    if sig.is_empty() {
        let mut p = PriPattern::empty("complex", dup, n);
        p.pri_us = q_sorted(&sorted(g.iter().copied()), 0.5);
        return p;
    }
    // label of each gap = index into `sig`
    let lab: Vec<Option<usize>> = g
        .iter()
        .map(|&x| sig.iter().position(|c| x >= c.lo && x <= c.hi))
        .collect();
    let self_follow = |c: usize| {
        let (mut a, mut b) = (0usize, 0usize);
        for i in 0..n - 1 {
            if lab[i] == Some(c) {
                b += 1;
                if lab[i + 1] == Some(c) {
                    a += 1;
                }
            }
        }
        if b == 0 { 0.0 } else { a as f64 / b as f64 }
    };
    // primary levels vs harmonics, smallest first
    let mut prim: Vec<usize> = Vec::new();
    let mut harm: Vec<usize> = Vec::new();
    for (ci, c) in sig.iter().enumerate() {
        if prim.is_empty() {
            prim.push(ci);
            continue;
        }
        let pv: Vec<f64> = prim.iter().map(|&p| sig[p].med).collect();
        let max_n = prim.iter().map(|&p| sig[p].n).max().unwrap_or(0);
        let c0 = &sig[prim[0]];
        // a sum of 2-4 gaps: sharp levels -> exact sums; a wide (continuous)
        // smallest level -> anywhere in its 2x..4x range
        let summed = if c0.w > WIDE {
            c.med >= 2.0 * c0.lo * 0.98 && c.med <= 4.0 * c0.hi * 1.02
        } else {
            prim.len() <= 16 && is_sum(c.med, &pv, 0.015)
        };
        if summed && self_follow(ci) < 0.5 && (c.n as f64) < 0.6 * max_n as f64 {
            harm.push(ci);
        } else {
            prim.push(ci);
        }
    }
    let is_prim: Vec<bool> = (0..sig.len()).map(|i| prim.contains(&i)).collect();
    let plab: Vec<Option<usize>> = lab.iter().map(|l| l.filter(|&c| is_prim[c])).collect();
    let mut p = PriPattern::empty("complex", dup, n);
    p.levels_us = prim.iter().map(|&c| sig[c].med).collect();
    p.harmonics_us = harm.iter().map(|&c| sig[c].med).collect();
    let c0 = &sig[prim[0]];
    let gaps_in = |c: usize| sorted(g.iter().zip(&lab).filter(|(_, l)| **l == Some(c)).map(|(x, _)| *x));
    if c0.w > WIDE {
        let s0 = gaps_in(prim[0]);
        let mean = s0.iter().sum::<f64>() / s0.len() as f64;
        p.pri_us = mean;
        if prim.len() == 1 {
            let mut steps = Vec::new();
            for i in 0..n - 1 {
                if lab[i] == Some(prim[0]) && lab[i + 1] == Some(prim[0]) {
                    steps.push((g[i + 1] - g[i]).abs());
                }
            }
            let steps = sorted(steps.into_iter());
            let iqr = q_sorted(&s0, 0.75) - q_sorted(&s0, 0.25);
            let step = q_sorted(&steps, 0.5);
            let (lo, hi) = (q_sorted(&s0, 0.02), q_sorted(&s0, 0.98));
            if iqr > 0.0 && step / iqr < 0.25 {
                p.kind = "sliding";
                p.jitter_pct = 0.0;
                p.sliding = Some(Sliding { min_us: lo, max_us: hi, step_us: step, ramp_pulses: if step > 0.0 { (hi - lo) / step } else { f64::INFINITY } });
            } else {
                p.kind = "jitter";
                p.jitter_pct = (q_sorted(&s0, 0.975) - q_sorted(&s0, 0.025)) / 2.0 / mean * 100.0 / 0.95;
            }
        }
    } else if prim.iter().any(|&c| sig[c].w > WIDE) {
        p.pri_us = c0.med;
        if let Some((m, sl)) = smooth_sliding(&g, &plab, prim.len()) {
            p.kind = "sliding";
            p.pri_us = m;
            p.sliding = Some(sl);
        }
    } else if prim.len() == 1 {
        p.kind = "constant";
        p.pri_us = c0.med;
        p.jitter_pct = c0.w * 100.0;
    } else {
        let k_lv = prim.len();
        p.jitter_pct = prim.iter().map(|&c| sig[c].w).fold(0.0, f64::max) * 100.0;
        // same level as the previous gap (both primary)
        let (mut same, mut pairs) = (0usize, 0usize);
        for i in 0..n - 1 {
            if let (Some(a), Some(b)) = (plab[i], plab[i + 1]) {
                pairs += 1;
                same += (a == b) as usize;
            }
        }
        let s1 = if pairs == 0 { 0.0 } else { same as f64 / pairs as f64 };
        if s1 >= 0.6 {
            p.kind = "dwell_switch";
            let (runs, times) = dwell_runs(&g, &plab, &sig);
            let (rs, ts) = (sorted(runs.into_iter()), sorted(times.into_iter()));
            let pris: Vec<f64> = p.levels_us.clone();
            p.pri_us = pris.iter().sum::<f64>() / k_lv as f64;
            p.dwell = Some(Dwell { pris_us: pris, dwell_pulses: q_sorted(&rs, 0.5), dwell_us: q_sorted(&ts, 0.5) });
        } else if let Some((k, el)) = stagger_period(&plab, &sig) {
            p.kind = "stagger";
            let period: f64 = el.iter().sum();
            p.pri_us = period / k as f64;
            p.stagger = Some(Stagger { period_us: period, elements_us: el });
        } else {
            p.pri_us = c0.med;
            if let Some((m, sl)) = smooth_sliding(&g, &plab, k_lv) {
                p.kind = "sliding";
                p.pri_us = m;
                p.sliding = Some(sl);
            }
        }
    }
    // missing pulses: non-primary gaps as whole multiples of the mean PRI
    if p.pri_us.is_finite() && p.pri_us > 0.0 {
        let mut miss = 0.0;
        for (x, l) in g.iter().zip(&plab) {
            if l.is_none() {
                let k = (x / p.pri_us).round();
                if k >= 2.0 {
                    miss += k - 1.0;
                }
            }
        }
        p.missing_pct = miss / (n as f64 + miss) * 100.0;
    }
    p
}

/// Many levels (>= 6) visited in a smooth order: a sliding PRI whose steps
/// repeat every ramp. Step between consecutive primary gaps small vs their
/// spread (median |step| < 0.25 IQR). Returns (mean PRI, sliding).
fn smooth_sliding(g: &[f64], plab: &[Option<usize>], k_lv: usize) -> Option<(f64, Sliding)> {
    if k_lv < 6 {
        return None;
    }
    let n = g.len();
    let s0 = sorted(g.iter().zip(plab).filter(|(_, l)| l.is_some()).map(|(x, _)| *x));
    let mut steps = Vec::new();
    for i in 0..n - 1 {
        if plab[i].is_some() && plab[i + 1].is_some() {
            steps.push((g[i + 1] - g[i]).abs());
        }
    }
    let steps = sorted(steps.into_iter());
    let iqr = q_sorted(&s0, 0.75) - q_sorted(&s0, 0.25);
    let step = q_sorted(&steps, 0.5);
    if !(iqr > 0.0 && step / iqr < 0.25) {
        return None;
    }
    let (lo, hi) = (q_sorted(&s0, 0.02), q_sorted(&s0, 0.98));
    let mean = s0.iter().sum::<f64>() / s0.len() as f64;
    Some((mean, Sliding { min_us: lo, max_us: hi, step_us: step, ramp_pulses: if step > 0.0 { (hi - lo) / step } else { f64::INFINITY } }))
}

/// Dwell runs: (gaps per complete run, run time µs). A non-primary gap that
/// is a whole multiple of the current level continues the run (missing
/// pulses); anything else breaks it. First and last runs are partial: dropped.
fn dwell_runs(g: &[f64], plab: &[Option<usize>], sig: &[Cluster]) -> (Vec<f64>, Vec<f64>) {
    let mut runs = Vec::new();
    let mut times = Vec::new();
    let mut cur: Option<usize> = None;
    let (mut cnt, mut tm, mut clean, mut first) = (0.0, 0.0, true, true);
    let mut close = |cnt: f64, tm: f64, clean: bool, first: bool, runs: &mut Vec<f64>, times: &mut Vec<f64>| {
        if !first && clean && cnt > 0.0 {
            runs.push(cnt);
            times.push(tm);
        }
    };
    for (i, &x) in g.iter().enumerate() {
        match (plab[i], cur) {
            (Some(c), Some(cc)) if c == cc => {
                cnt += 1.0;
                tm += x;
            }
            (Some(c), _) => {
                if cur.is_some() {
                    close(cnt, tm, clean, first, &mut runs, &mut times);
                    first = false;
                }
                cur = Some(c);
                cnt = 1.0;
                tm = x;
                clean = true;
            }
            (None, Some(cc)) => {
                let m = sig[cc].med;
                let k = (x / m).round();
                if k >= 1.0 && (x - k * m).abs() <= 0.02 * x {
                    cnt += k;
                    tm += x;
                } else {
                    clean = false;
                    cnt += 1.0;
                    tm += x;
                }
            }
            (None, None) => {}
        }
    }
    // last run is partial: not closed
    (runs, times)
}

/// Shortest k (2..=16) with level[i] == level[i+k] on >= 85 % of contiguous
/// primary stretches; returns (k, elements in order starting at the smallest).
fn stagger_period(plab: &[Option<usize>], sig: &[Cluster]) -> Option<(usize, Vec<f64>)> {
    let n = plab.len();
    // run_len[i] = number of consecutive primary labels starting at i
    let mut run_len = vec![0usize; n + 1];
    for i in (0..n).rev() {
        run_len[i] = if plab[i].is_some() { run_len[i + 1] + 1 } else { 0 };
    }
    for k in 2..=16usize {
        let (mut ok, mut tot) = (0usize, 0usize);
        for i in 0..n {
            if run_len[i] > k {
                tot += 1;
                ok += (plab[i] == plab[i + k]) as usize;
            }
        }
        if tot >= 10 && ok as f64 >= 0.85 * tot as f64 {
            // most common window of k labels, rotated to a canonical phase
            let mut counts: std::collections::HashMap<Vec<usize>, usize> = Default::default();
            for i in 0..n {
                if run_len[i] >= k {
                    let w: Vec<usize> = (0..k).map(|j| plab[i + j].unwrap()).collect();
                    *counts.entry(canon(&w, sig)).or_default() += 1;
                }
            }
            let best = counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))?.0;
            return Some((k, best.iter().map(|&c| sig[c].med).collect()));
        }
    }
    None
}

/// Rotation of a window starting at the smallest value; ties -> smallest
/// value sequence (lexicographic).
fn canon(w: &[usize], sig: &[Cluster]) -> Vec<usize> {
    let k = w.len();
    (0..k)
        .map(|r| (0..k).map(|j| w[(r + j) % k]).collect::<Vec<usize>>())
        .min_by(|a, b| {
            let va: Vec<f64> = a.iter().map(|&c| sig[c].med).collect();
            let vb: Vec<f64> = b.iter().map(|&c| sig[c].med).collect();
            va.partial_cmp(&vb).unwrap()
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn train(gaps_us: impl Iterator<Item = f64>, miss: f64, seed: u64) -> Vec<f64> {
        let mut r = rand::rngs::StdRng::seed_from_u64(seed);
        let mut t = 0.0;
        let mut out = Vec::new();
        for g in gaps_us {
            t += g * (1.0 + r.gen_range(-0.0005..0.0005));
            if r.gen::<f64>() >= miss {
                out.push(t * 1e3);
            }
        }
        out
    }

    #[test]
    fn constant_with_missing_pulses() {
        let p = analyze(&train(std::iter::repeat(1000.0).take(3000), 0.15, 1));
        assert_eq!(p.kind, "constant");
        assert!((p.pri_us - 1000.0).abs() < 2.0, "{p:?}");
        assert!(p.harmonics_us.iter().any(|h| (h - 2000.0).abs() < 20.0));
        assert!((p.missing_pct - 15.0).abs() < 3.0, "{}", p.missing_pct);
    }

    #[test]
    fn jitter_ten_percent() {
        let mut r = rand::rngs::StdRng::seed_from_u64(2);
        let g: Vec<f64> = (0..3000).map(|_| 800.0 * (1.0 + r.gen_range(-0.10..0.10))).collect();
        let p = analyze(&train(g.into_iter(), 0.05, 3));
        assert_eq!(p.kind, "jitter", "{p:?}");
        assert!((p.jitter_pct - 10.0).abs() < 1.5, "{}", p.jitter_pct);
        assert!((p.pri_us - 800.0).abs() < 10.0);
    }

    #[test]
    fn stagger_with_repeat_and_missing() {
        let el = [700.0, 700.0, 910.0, 820.0];
        let p = analyze(&train(el.iter().cycle().take(4000).copied(), 0.1, 4));
        assert_eq!(p.kind, "stagger", "{p:?}");
        let s = p.stagger.unwrap();
        assert!((s.period_us - 3130.0).abs() < 5.0);
        let want = [700.0, 700.0, 910.0, 820.0];
        assert_eq!(s.elements_us.len(), 4);
        for (a, b) in s.elements_us.iter().zip(want) {
            assert!((a - b).abs() < 3.0, "{:?}", s.elements_us);
        }
    }

    #[test]
    fn dwell_switch_three_levels() {
        let pris = [500.0, 1300.0, 900.0];
        let g = (0..60).flat_map(|d| std::iter::repeat(pris[d % 3]).take(40));
        let p = analyze(&train(g, 0.05, 5));
        assert_eq!(p.kind, "dwell_switch", "{p:?}");
        let d = p.dwell.unwrap();
        assert_eq!(d.pris_us.len(), 3);
        assert!((d.dwell_pulses - 40.0).abs() <= 2.0, "{}", d.dwell_pulses);
    }

    #[test]
    fn sliding_sawtooth() {
        let g = (0..3000).map(|i| 1000.0 + 400.0 * ((i % 200) as f64 / 200.0));
        let p = analyze(&train(g, 0.05, 6));
        assert_eq!(p.kind, "sliding", "{p:?}");
        let s = p.sliding.unwrap();
        assert!(s.min_us < 1020.0 && s.max_us > 1370.0, "{s:?}");
    }

    #[test]
    fn sliding_fast_ramp_repeating_steps() {
        // 62-pulse ramp: every step value repeats each ramp (sharp levels)
        let g = (0..6000).map(|i| 830.0 + 424.0 * ((i % 62) as f64 / 62.0));
        let p = analyze(&train(g, 0.2, 8));
        assert_eq!(p.kind, "sliding", "{:?}", p.kind);
    }

    #[test]
    fn duplicates_skipped_and_too_few() {
        let mut t = train(std::iter::repeat(1000.0).take(500), 0.0, 7);
        let d: Vec<f64> = t.iter().map(|x| x + 300.0).collect(); // +0.3 µs copies
        t.extend(d);
        let p = analyze(&t);
        assert_eq!(p.kind, "constant");
        assert!(p.dup_skipped >= 490);
        assert_eq!(analyze(&[0.0, 1e6, 2e6]).kind, "too_few");
    }
}
