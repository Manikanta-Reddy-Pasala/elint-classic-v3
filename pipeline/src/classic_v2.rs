//! Classic engine v2 — model-free emitter detection from measured PDW
//! fields only (TOA, carrier, PW, BW, power, DOA, intra-pulse modulation).
//! It never reads the CSV's ground-truth `emitter_id` or generator PRI.
//!
//!  1. Cells: (modulation, PW on a 1 % log grid, BW 1 MHz, carrier 1 MHz);
//!     cells with < `min_cell` pulses are noise. `unk*` modulation pulses
//!     (failed intra-pulse measurement) are skipped.
//!  2. Components: neighbouring cells (±1 PW, ±1 BW, ±`f_adj` MHz) join.
//!  3. Same-emitter split repair: components with overlapping carrier and
//!     BW ranges and PW within 2 % join.
//!  4. Pulse-train coherence: two groups whose pulses keep a fixed TOA
//!     relation (top-4 1-µs bins of "offset to the preceding pulse of the
//!     other group" hold ≥ `coh_thr` of pulses) are one emitter — this
//!     joins the modes of multi-mode radars and the hops of agile ones.
//!     Both directions must agree: max ≥ `coh_thr` and min ≥ `coh_min`.
//!     Calibrated on the generator corpus: same emitter max ≥ 0.19 /
//!     min ≥ 0.06; different emitters max ≤ 0.14 with min ≤ 0.09, and the
//!     one-way PRI-beat case (max 0.14, min 0.01) is rejected.
//!  5. Naming: library decision tree on the group (carrier mode, BW mode,
//!     PRI from TOA, 99th-pct PW, lfm_up share). One emitter per library
//!     id; every other group is its own `unknown_ai` emitter.
//!  6. Writes: one emitter + signal per emitter, cluster rows per
//!     `cluster_sec` time slice (members in TOA order so PRI is measured
//!     on the real pulse train), CH `signals.clusters` activity ids.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use tracing::{debug, info};

use crate::onnx_flow::{angle_score, cluster_angle, signature_id, Db, FlowPulse, Member};
use crate::AiState;

#[derive(Clone, Debug)]
pub struct ClassicCfg {
    pub min_cell: usize,
    pub pw_rel: f64,
    pub f_adj: i64,
    pub merge_f_hz: f64,
    pub merge_bw_hz: f64,
    pub merge_pw_rel: f64,
    pub coh_thr: f64,
    pub coh_min: f64,
    pub coh_top: usize,
    pub coh_samples: usize,
    /// Minimum A pulses that actually have a preceding B pulse; fewer -> score 0.
    pub coh_min_pairs: usize,
    /// Split repair: do not join two look-alike groups whose pulse trains
    /// clearly keep different clocks (both directions below `coh_min`).
    pub repair_clock_veto: bool,
    pub min_group: usize,
    pub cluster_sec: f64,
    /// Intervals below this (µs) are the same pulse seen twice — skipped for PRI.
    pub pri_min_gap_us: f64,
    /// Time slices with fewer pulses are folded into their neighbour.
    pub min_slice: usize,
    /// v3 (frequency-agile support): rescue sparse cells + join pieces
    /// that form one clean pulse train + span-checked naming.
    /// Default ON; CLASSIC_V3=0 = exact v2 behaviour.
    pub v3: bool,
    /// v3: gap histogram bin width (relative, log scale) and top-k bins.
    pub reg_bin_rel: f64,
    pub reg_top: usize,
    /// v3: a joined train needs this top-k share ...
    pub reg_min: f64,
    /// ... and every piece's pulses must fit it at >= reg_fit x that share.
    pub reg_fit: f64,
    /// v3: share of a group's pulses inside a rule's carrier window to name it.
    pub name_share: f64,
    /// v3: a rescued sparse group must spread at least this wide (MHz);
    /// narrower leftovers are stray pulses of fixed radars.
    pub rescue_span_mhz: f64,
    /// v3: one transmitter never sends two pulses closer than its own
    /// shortest gap. Piece pulses closer than coll_rel x that gap to another
    /// piece's pulse are collisions; more than coll_max of them = other radar.
    pub coll_rel: f64,
    pub coll_max: f64,
    /// v3: rescued sparse groups need at least this many pulses.
    pub rescue_min: usize,
    /// v3: time-disjoint pieces of one radar share a main gap value
    /// (within this relative tolerance).
    pub gap_ratio_tol: f64,
    /// v3: the beat step never joins groups whose pulses overlap in time
    /// (a pulse starting while the other group's pulse is on air) beyond
    /// this share — one transmitter cannot do that.
    pub coh_max_overlap: f64,
    /// v3: optional cap on the carrier gap (MHz) between two groups the beat
    /// step may join (widest agile band expected). 0 = off.
    pub coh_max_fgap_mhz: f64,
    /// v3 DOA gate (direction-finder data only, detected from the data):
    /// never join pieces whose bearings differ at the same moment. 0 = off.
    pub doa_gate: bool,
    /// A receiver is a direction finder when time-adjacent pulses of one
    /// group differ by at most this (median, deg). Omni / random DOA: ~90.
    pub df_noise_max: f64,
    /// Track-following tolerance (deg) = max(doa_track_tol_min, doa_tol_k x noise).
    pub doa_track_tol_min: f64,
    pub doa_tol_k: f64,
    /// Centre tolerance (deg) = max(doa_tol_min, doa_center_k x centre spread).
    pub doa_tol_min: f64,
    pub doa_center_k: f64,
    /// Bearings are usable only if one radar's track centres spread <= this (deg).
    pub df_center_max: f64,
    /// Pulses closer than this (ms) count as "at the same moment".
    pub doa_win_ms: f64,
    /// A bearing track ends after this silence (ms).
    pub doa_track_gap_ms: f64,
    /// A direction split keeps a part only if it holds this share of the group.
    pub doa_split_share: f64,
    /// 1 = parts below that size fold into the largest part; 0 = kept apart.
    pub doa_fold_small: bool,
    /// DOA noise: closest bearing among this many following pulses.
    pub doa_next: usize,
    /// Same-moment comparisons needed before bearings count as evidence.
    pub doa_min_evidence: usize,
    /// Bearing drift (deg/s) up to which the receiver counts as a fixed
    /// direction finder (a spinning antenna sweeps ~100s of deg/s).
    pub doa_fixed_rate: f64,
    /// DF: split every group by direction before the v3 join (1) or only the
    /// rescued groups (0); and split final groups by direction tracks (1) or
    /// by the v2-era DOA histogram rule (0).
    pub doa_presplit: bool,
    pub doa_final_split: bool,
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl ClassicCfg {
    pub fn from_env() -> Self {
        Self {
            min_cell: env_or("CLASSIC_MIN_CELL", 20),
            pw_rel: env_or("CLASSIC_PW_REL", 0.01),
            f_adj: env_or("CLASSIC_F_ADJ", 2),
            merge_f_hz: env_or("CLASSIC_MERGE_F_HZ", 3e6),
            merge_bw_hz: env_or("CLASSIC_MERGE_BW_HZ", 1e6),
            merge_pw_rel: env_or("CLASSIC_MERGE_PW_REL", 0.02),
            coh_thr: env_or("CLASSIC_COH_THR", 0.13),
            coh_min: env_or("CLASSIC_COH_MIN", 0.04),
            coh_top: env_or("CLASSIC_COH_TOP", 4),
            coh_samples: env_or("CLASSIC_COH_SAMPLES", 3000),
            coh_min_pairs: env_or("CLASSIC_COH_MIN_PAIRS", 20),
            repair_clock_veto: env_or("CLASSIC_REPAIR_CLOCK_VETO", 1) == 1,
            min_group: env_or("CLASSIC_MIN_GROUP", 50),
            cluster_sec: env_or("CLASSIC_CLUSTER_SEC", 1.0),
            pri_min_gap_us: env_or("CLASSIC_PRI_MIN_GAP_US", 1.0),
            min_slice: env_or("CLASSIC_MIN_SLICE", 50),
            v3: env_or("CLASSIC_V3", 1) == 1,
            reg_bin_rel: env_or("CLASSIC_REG_BIN_REL", 0.01),
            reg_top: env_or("CLASSIC_REG_TOP", 8),
            reg_min: env_or("CLASSIC_REG_MIN", 0.25),
            reg_fit: env_or("CLASSIC_REG_FIT", 0.5),
            name_share: env_or("CLASSIC_NAME_SHARE", 0.8),
            rescue_span_mhz: env_or("CLASSIC_RESCUE_SPAN_MHZ", 20.0),
            coll_rel: env_or("CLASSIC_COLL_REL", 0.4),
            coll_max: env_or("CLASSIC_COLL_MAX", 0.02),
            rescue_min: env_or("CLASSIC_RESCUE_MIN", 200),
            gap_ratio_tol: env_or("CLASSIC_GAP_RATIO_TOL", 0.05),
            coh_max_overlap: env_or("CLASSIC_COH_MAX_OVERLAP", 0.002),
            coh_max_fgap_mhz: env_or("CLASSIC_COH_MAX_FGAP_MHZ", 0.0),
            doa_gate: env_or("CLASSIC_DOA_GATE", 1) == 1,
            df_noise_max: env_or("CLASSIC_DF_NOISE_MAX_DEG", 10.0),
            doa_track_tol_min: env_or("CLASSIC_DOA_TRACK_TOL_MIN_DEG", 5.0),
            doa_tol_k: env_or("CLASSIC_DOA_TOL_K", 4.0),
            doa_tol_min: env_or("CLASSIC_DOA_TOL_MIN_DEG", 6.0),
            doa_center_k: env_or("CLASSIC_DOA_CENTER_K", 3.0),
            df_center_max: env_or("CLASSIC_DF_CENTER_MAX_DEG", 20.0),
            doa_win_ms: env_or("CLASSIC_DOA_WIN_MS", 20.0),
            doa_track_gap_ms: env_or("CLASSIC_DOA_TRACK_GAP_MS", 50.0),
            doa_split_share: env_or("CLASSIC_DOA_SPLIT_SHARE", 0.1),
            doa_fold_small: env_or("CLASSIC_DOA_FOLD_SMALL", 0) == 1,
            doa_next: env_or("CLASSIC_DOA_NEXT", 6),
            doa_min_evidence: env_or("CLASSIC_DOA_MIN_EVIDENCE", 20),
            doa_fixed_rate: env_or("CLASSIC_DOA_FIXED_RATE", 15.0),
            doa_presplit: env_or("CLASSIC_DOA_PRESPLIT", 1) == 1,
            doa_final_split: env_or("CLASSIC_DOA_FINAL_SPLIT", 1) == 1,
        }
    }
}

struct Uf(Vec<usize>);
impl Uf {
    fn new(n: usize) -> Self {
        Uf((0..n).collect())
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra] = rb;
        }
    }
}

fn norm_mod(m: &str) -> String {
    m.trim().to_ascii_lowercase()
}

/// Linear-interpolated quantile by selection (O(n), no full sort).
fn quantile(v: &mut [f64], q: f64) -> f64 {
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
    let vlo = *v.select_nth_unstable_by(lo, cmp).1;
    let vhi = if hi == lo { vlo } else { *v.select_nth_unstable_by(hi, cmp).1 };
    vlo + (vhi - vlo) * (pos - lo as f64)
}

/// pandas `Series.mode().iloc[0]` on integer bins: most frequent, ties → smallest.
/// Dense vector histogram when the bin range is small (the usual case),
/// HashMap otherwise.
fn mode_bin(it: impl Iterator<Item = i64>) -> Option<i64> {
    let v: Vec<i64> = it.collect();
    let (&lo, &hi) = (v.iter().min()?, v.iter().max()?);
    if hi - lo < 1 << 20 {
        let mut h = vec![0u32; (hi - lo + 1) as usize];
        for b in &v {
            h[(b - lo) as usize] += 1;
        }
        let mut best = 0usize;
        for i in 1..h.len() {
            if h[i] > h[best] {
                best = i;
            }
        }
        return Some(lo + best as i64);
    }
    let mut h: HashMap<i64, usize> = HashMap::new();
    for b in v {
        *h.entry(b).or_insert(0) += 1;
    }
    h.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map(|(b, _)| b)
}

/// Dominant pulse interval (µs) of a TOA-sorted train.
fn pri_mode_us(toas: &[f64]) -> f64 {
    mode_bin(toas.windows(2).map(|w| (w[1] - w[0]) / 1e3).filter(|d| *d > 0.5).map(|d| d.round() as i64))
        .map(|b| b as f64)
        .unwrap_or(0.0)
}

/// Top-k share of 1-µs offsets from A pulses to the preceding B pulse.
fn coherence(ta: &[f64], tb: &[f64], top: usize, samples: usize, min_pairs: usize) -> f64 {
    let (score, tot) = coherence_n(ta, tb, top, samples);
    // Too few A pulses with a B pulse before them (e.g. B on air only at the
    // very end): a top-k share over 2 offsets is 1.0 by chance, not a clock.
    if tot < min_pairs.max(1) {
        return 0.0;
    }
    score
}

/// Classic step-4 rule: two pulse trains keep the same clock (both ways).
pub(crate) fn same_clock(ta: &[f64], tb: &[f64], cfg: &ClassicCfg) -> bool {
    let c1 = coherence(ta, tb, cfg.coh_top, cfg.coh_samples, cfg.coh_min_pairs);
    let c2 = coherence(tb, ta, cfg.coh_top, cfg.coh_samples, cfg.coh_min_pairs);
    c1.max(c2) >= cfg.coh_thr && c1.min(c2) >= cfg.coh_min
}

/// Coherence score plus the number of A pulses actually compared.
fn coherence_n(ta: &[f64], tb: &[f64], top: usize, samples: usize) -> (f64, usize) {
    if ta.len() < 20 || tb.len() < 20 {
        return (0.0, 0);
    }
    let n = samples.min(ta.len());
    let mut h: HashMap<i64, usize> = HashMap::new();
    let mut tot = 0usize;
    for k in 0..n {
        let i = if n == 1 { 0 } else { ((k as f64) * (ta.len() - 1) as f64 / (n - 1) as f64) as usize };
        let t = ta[i];
        let j = tb.partition_point(|x| *x < t);
        if j == 0 {
            continue;
        }
        *h.entry(((t - tb[j - 1]) / 1e3).round() as i64).or_insert(0) += 1;
        tot += 1;
    }
    if tot == 0 {
        return (0.0, 0);
    }
    let mut c: Vec<usize> = h.into_values().collect();
    c.sort_unstable_by(|a, b| b.cmp(a));
    (c.iter().take(top).sum::<usize>() as f64 / tot as f64, tot)
}

#[derive(serde::Deserialize, Debug, Clone)]
struct LibRule {
    id: i64,
    f: [f64; 2],
    pw_max_ge: Option<f64>,
    lfm_ge: Option<f64>,
    pri_lt: Option<f64>,
    /// "mode" (default): carrier mode inside `f`. "span": share of the
    /// group's pulses inside `f` >= name_share (frequency-agile radars).
    #[serde(default)]
    r#match: Option<String>,
    /// Optional median pulse width range (us) and dominant modulation.
    #[serde(default)]
    pw: Option<[f64; 2]>,
    #[serde(default)]
    modulation: Option<String>,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct LibTable {
    max_bw_mhz: f64,
    rules: Vec<LibRule>,
}

/// Built-in naming table (same file as `config/classic_library.json`).
const DEFAULT_LIBRARY: &str = include_str!("../../config/classic_library.json");

/// Naming table: `CLASSIC_LIBRARY=/path.json` if set and valid, else built-in.
fn library_table() -> &'static LibTable {
    static T: std::sync::OnceLock<LibTable> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        if let Ok(path) = std::env::var("CLASSIC_LIBRARY") {
            let parsed = std::fs::read_to_string(&path)
                .map_err(anyhow::Error::from)
                .and_then(|s| serde_json::from_str::<LibTable>(&s).map_err(anyhow::Error::from));
            match parsed {
                Ok(t) => {
                    info!("classic_v2: naming table {path} ({} rules)", t.rules.len());
                    return t;
                }
                Err(e) => tracing::warn!("classic_v2: CLASSIC_LIBRARY {path} unusable ({e}); using built-in table"),
            }
        }
        serde_json::from_str(DEFAULT_LIBRARY).expect("built-in classic_library.json is valid")
    })
}

/// Library id from measured group features (no labels, no generator PRI).
/// Rules come from the naming table, checked top to bottom; first match wins.
fn library_id(ps: &[&FlowPulse], toas_sorted: &[f64], cfg: &ClassicCfg) -> Option<i64> {
    let f_mhz = mode_bin(ps.iter().map(|p| (p.freq_hz / 1e5).round() as i64))? as f64 / 10.0;
    let bw_mhz = mode_bin(ps.iter().map(|p| (p.bandwidth_hz / 1e5).round() as i64)).unwrap_or(0) as f64 / 10.0;
    let lib = library_table();
    if bw_mhz >= lib.max_bw_mhz {
        return None;
    }
    // Group statistics are computed once, only when a rule needs them.
    let mut pw_max: Option<f64> = None;
    let mut lfm: Option<f64> = None;
    let mut pri: Option<f64> = None;
    let share_in = |r: &LibRule| {
        ps.iter().filter(|p| (r.f[0]..=r.f[1]).contains(&(p.freq_hz / 1e6))).count() as f64 / ps.len() as f64
    };
    let mut pw_med: Option<f64> = None;
    let mut mod_top: Option<String> = None;
    for r in &lib.rules {
        if let Some([lo, hi]) = r.pw {
            let x = *pw_med.get_or_insert_with(|| {
                let mut w: Vec<f64> = ps.iter().map(|p| p.duration_ns / 1e3).collect();
                quantile(&mut w, 0.5)
            });
            if x < lo || x > hi {
                continue;
            }
        }
        if let Some(m) = &r.modulation {
            let top = mod_top.get_or_insert_with(|| {
                let mut c: HashMap<String, usize> = HashMap::new();
                for p in ps {
                    *c.entry(norm_mod(&p.modulation).replace('_', "")).or_default() += 1;
                }
                c.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map(|x| x.0).unwrap_or_default()
            });
            if *top != norm_mod(m).replace('_', "") {
                continue;
            }
        }
        let span_rule = r.r#match.as_deref() == Some("span");
        if span_rule {
            if !cfg.v3 || share_in(r) < cfg.name_share {
                continue;
            }
        } else {
            if !(r.f[0]..=r.f[1]).contains(&f_mhz) {
                continue;
            }
            // v3: a wide (agile) group is not named by its busiest carrier alone
            if cfg.v3 && share_in(r) < cfg.name_share {
                continue;
            }
        }
        if let Some(v) = r.pw_max_ge {
            let x = *pw_max.get_or_insert_with(|| {
                let mut pw: Vec<f64> = ps.iter().map(|p| p.duration_ns / 1e3).collect();
                quantile(&mut pw, 0.99)
            });
            if x < v {
                continue;
            }
        }
        if let Some(v) = r.lfm_ge {
            let x = *lfm.get_or_insert_with(|| {
                ps.iter().filter(|p| norm_mod(&p.modulation).replace('_', "") == "lfmup").count() as f64
                    / ps.len() as f64
            });
            if x < v {
                continue;
            }
        }
        if let Some(v) = r.pri_lt {
            let x = *pri.get_or_insert_with(|| pri_mode_us(toas_sorted));
            if x >= v {
                continue;
            }
        }
        return Some(r.id);
    }
    None
}

/// Frequency mode of one group, from carrier vs time only (report label):
/// fixed | sliding | burst | n_modal | agile_discrete | agile_continuous.
pub fn freq_mode(ps: &[&FlowPulse]) -> &'static str {
    if ps.len() < 20 {
        return "unknown";
    }
    let mut v: Vec<(f64, f64)> = ps.iter().map(|p| (p.toa_ns, p.freq_hz / 1e6)).collect();
    v.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut f: Vec<f64> = v.iter().map(|x| x.1).collect();
    let span = quantile(&mut f.clone(), 0.98) - quantile(&mut f, 0.02);
    if span <= 3.0 {
        return "fixed";
    }
    // carrier strongly tied to time -> sliding
    let n = v.len() as f64;
    let (mt, mf) = (v.iter().map(|x| x.0).sum::<f64>() / n, v.iter().map(|x| x.1).sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (t, fr) in &v {
        sxy += (t - mt) * (fr - mf);
        sxx += (t - mt) * (t - mt);
        syy += (fr - mf) * (fr - mf);
    }
    let r = if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 };
    // 1 MHz histogram -> peaks (local maxima holding >= 3 % of pulses)
    let mut h: BTreeMap<i64, usize> = BTreeMap::new();
    for x in &v {
        *h.entry(x.1.round() as i64).or_default() += 1;
    }
    let get = |k: i64| *h.get(&k).unwrap_or(&0);
    let peaks: Vec<i64> = h
        .iter()
        .filter(|(k, c)| **c as f64 >= 0.03 * n && **c >= get(**k - 1) && **c > get(**k + 1))
        .map(|(k, _)| *k)
        .collect();
    let in_peak = |fr: f64| peaks.iter().position(|p| (fr.round() as i64 - p).abs() <= 1);
    let covered = v.iter().filter(|x| in_peak(x.1).is_some()).count() as f64 / n;
    if r.abs() >= 0.9 && peaks.len() <= 1 {
        return "sliding";
    }
    if covered < 0.85 || peaks.is_empty() {
        return if r.abs() >= 0.9 { "sliding" } else { "agile_continuous" };
    }
    // run length of the same peak in TOA order
    let lab: Vec<Option<usize>> = v.iter().map(|x| in_peak(x.1)).collect();
    let (mut runs, mut cur) = (1usize, lab[0]);
    for l in &lab[1..] {
        if *l != cur {
            runs += 1;
            cur = *l;
        }
    }
    let mean_run = n / runs as f64;
    if mean_run >= 5.0 {
        "burst"
    } else if peaks.len() <= 3 {
        "n_modal"
    } else {
        "agile_discrete"
    }
}

/// Group pulses into emitters. Returns (emitter key, pulse indices) where
/// the key is `Some(library id)` or `None` for an unknown emitter.
pub fn detect(pulses: &[FlowPulse], cfg: &ClassicCfg) -> Vec<(Option<i64>, Vec<usize>)> {
    use crate::layer_timing::emit_dur;
    use rayon::prelude::*;
    let mut t = Instant::now();
    // 1. cells — modulation interned to a small id, then a parallel fold of
    //    per-thread cell maps merged at the end (no String per pulse).
    let mut mod_ids: HashMap<String, i64> = HashMap::new();
    // v3: a receiver without intra-pulse analysis reports no modulation at
    // all; then every pulse is one "none" class instead of being skipped.
    let no_mod = cfg.v3 && {
        let known = pulses.iter().filter(|p| {
            let m = norm_mod(&p.modulation);
            !(m.is_empty() || m.starts_with("unk"))
        });
        (known.count() as f64) < 0.1 * pulses.len() as f64
    };
    let pulse_mod: Vec<i64> = pulses
        .iter()
        .map(|p| {
            if no_mod {
                return 0;
            }
            let m = norm_mod(&p.modulation);
            if m.is_empty() || m.starts_with("unk") {
                return -1;
            }
            let n = mod_ids.len() as i64;
            *mod_ids.entry(m).or_insert(n)
        })
        .collect();
    type Cell = (i64, i64, i64, i64);
    let lw = (1.0 + cfg.pw_rel).ln();
    let mut cells: HashMap<Cell, Vec<usize>> = pulses
        .par_iter()
        .enumerate()
        .fold(HashMap::<Cell, Vec<usize>>::new, |mut acc, (i, p)| {
            let m = pulse_mod[i];
            if m >= 0 {
                let wc = ((p.duration_ns / 1e3).max(0.05).ln() / lw).round() as i64;
                let bc = (p.bandwidth_hz / 1e6).round() as i64;
                let fc = (p.freq_hz / 1e6).round() as i64;
                acc.entry((m, wc, bc, fc)).or_default().push(i);
            }
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, mut v) in b {
                a.entry(k).or_default().append(&mut v);
            }
            a
        });
    for v in cells.values_mut() {
        v.sort_unstable();
    }
    if let Some(p) = pulses.get(pulses.len() / 2) {
        debug!("classic_v2: sample pulse {:?}", p);
    }
    let n_cells_all = cells.len();
    let sparse: Vec<usize> = if cfg.v3 {
        cells.values().filter(|v| v.len() < cfg.min_cell).flatten().copied().collect()
    } else {
        Vec::new()
    };
    cells.retain(|_, v| v.len() >= cfg.min_cell);
    debug!("classic_v2: {} cells, {} with >= {} pulses", n_cells_all, cells.len(), cfg.min_cell);
    emit_dur("classic.cells", t.elapsed());
    t = Instant::now();
    let mut keys: Vec<Cell> = cells.keys().cloned().collect();
    keys.sort();
    let idx: HashMap<&Cell, usize> = keys.iter().enumerate().map(|(i, k)| (k, i)).collect();
    // 2. components over neighbouring cells
    let mut uf = Uf::new(keys.len());
    for (i, (m, w, b, f)) in keys.iter().enumerate() {
        for dw in -1..=1 {
            for db in -1..=1 {
                for df in -cfg.f_adj..=cfg.f_adj {
                    if let Some(&j) = idx.get(&(*m, w + dw, b + db, f + df)) {
                        uf.union(i, j);
                    }
                }
            }
        }
    }
    let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        let r = uf.find(i);
        comps.entry(r).or_default().extend_from_slice(&cells[k]);
    }
    let mut groups: Vec<Vec<usize>> = comps.into_values().collect();
    emit_dur("classic.components", t.elapsed());
    t = Instant::now();
    // v3: direction-finder check (from the data) for the DOA gate
    let dctx = if cfg.v3 && cfg.doa_gate { doa_ctx(pulses, &groups, cfg) } else { DoaCtx::default() };
    if std::env::var("CLASSIC_DOA_DEBUG").is_ok() {
        eprintln!("DOA ctx {:?}", dctx);
    }

    // 3. same-emitter split repair (carrier + BW overlap, PW within 2 %)
    let summ = |g: &Vec<usize>| {
        let mut f: Vec<f64> = g.iter().map(|&i| pulses[i].freq_hz).collect();
        let mut b: Vec<f64> = g.iter().map(|&i| pulses[i].bandwidth_hz).collect();
        let mut w: Vec<f64> = g.iter().map(|&i| pulses[i].duration_ns).collect();
        (quantile(&mut f, 0.02), quantile(&mut f, 0.98), quantile(&mut b, 0.02), quantile(&mut b, 0.98), quantile(&mut w, 0.5))
    };
    let s: Vec<_> = groups.par_iter().map(summ).collect();
    let rtoas: Vec<Vec<f64>> =
        if cfg.repair_clock_veto { groups.par_iter().map(|g| sorted_toas(pulses, g)).collect() } else { Vec::new() };
    // Two radars of the same type can look identical (carrier a few MHz
    // apart, same PW); their trains then keep unrelated clocks. Pieces of one
    // radar split by jitter keep the same clock (scores 0.6-1.0).
    let different_clock = |a: usize, b: usize| -> bool {
        if !cfg.repair_clock_veto {
            return false;
        }
        // Veto only on real evidence: both directions compared enough pulses.
        let (c1, n1) = coherence_n(&rtoas[a], &rtoas[b], cfg.coh_top, cfg.coh_samples);
        let (c2, n2) = coherence_n(&rtoas[b], &rtoas[a], cfg.coh_top, cfg.coh_samples);
        n1 >= cfg.coh_min_pairs && n2 >= cfg.coh_min_pairs && c1.max(c2) < cfg.coh_min
    };
    let rtr: Vec<Vec<(f64, f64)>> =
        if dctx.df { groups.par_iter().map(|g| doa_train(pulses, g)).collect() } else { Vec::new() };
    let mut uf = Uf::new(groups.len());
    for a in 0..groups.len() {
        for b in a + 1..groups.len() {
            let (fa0, fa1, ba0, ba1, wa) = s[a];
            let (fb0, fb1, bb0, bb1, wb) = s[b];
            if fa0 - cfg.merge_f_hz <= fb1
                && fb0 - cfg.merge_f_hz <= fa1
                && ba0 - cfg.merge_bw_hz <= bb1
                && bb0 - cfg.merge_bw_hz <= ba1
                && (wa - wb).abs() <= cfg.merge_pw_rel * wa.min(wb)
                && !different_clock(a, b)
                && !(dctx.df && doa_conflict(&rtr[a], &rtr[b], &dctx, cfg))
            {
                uf.union(a, b);
            }
        }
    }
    groups = regroup(groups, &mut uf);
    groups.retain(|g| g.len() >= cfg.min_cell);
    debug!("classic_v2: {} groups after split repair", groups.len());
    emit_dur("classic.split_repair", t.elapsed());
    t = Instant::now();

    // 4. pulse-train coherence
    let toas: Vec<Vec<f64>> = groups.par_iter().map(|g| sorted_toas(pulses, g)).collect();
    let ctr: Vec<Vec<(f64, f64)>> =
        if dctx.df { groups.par_iter().map(|g| doa_train(pulses, g)).collect() } else { Vec::new() };
    let mut uf = Uf::new(groups.len());
    {
        let pairs: Vec<(usize, usize)> =
            (0..groups.len()).flat_map(|a| (a + 1..groups.len()).map(move |b| (a, b))).collect();
        let hits: Vec<(usize, usize)> = pairs
            .par_iter()
            .filter(|(a, b)| {
                // Same emitter: coherent both ways. A one-way peak is a PRI
                // beat between two unrelated trains (e.g. 2116 µs vs 220 µs).
                let c1 = coherence(&toas[*a], &toas[*b], cfg.coh_top, cfg.coh_samples, cfg.coh_min_pairs);
                let c2 = coherence(&toas[*b], &toas[*a], cfg.coh_top, cfg.coh_samples, cfg.coh_min_pairs);
                if !(c1.max(c2) >= cfg.coh_thr && c1.min(c2) >= cfg.coh_min) {
                    return false;
                }
                if cfg.v3 {
                    if pulse_overlap(pulses, &groups[*a], &groups[*b]) > cfg.coh_max_overlap {
                        return false;
                    }
                    if dctx.df && doa_conflict(&ctr[*a], &ctr[*b], &dctx, cfg) {
                        return false;
                    }
                    if cfg.coh_max_fgap_mhz > 0.0 {
                        let fq = |g: &Vec<usize>, q: f64| {
                            let mut v: Vec<f64> = g.iter().map(|&i| pulses[i].freq_hz / 1e6).collect();
                            quantile(&mut v, q)
                        };
                        let (a0, a1, b0, b1) = (fq(&groups[*a], 0.05), fq(&groups[*a], 0.95), fq(&groups[*b], 0.05), fq(&groups[*b], 0.95));
                        if (a0 - b1).max(b0 - a1) > cfg.coh_max_fgap_mhz {
                            return false;
                        }
                    }
                }
                true
            })
            .copied()
            .collect();
        if std::env::var("CLASSIC_COH_DEBUG").is_ok() {
            let st = |g: &Vec<usize>| {
                let mut w: Vec<f64> = g.iter().map(|&i| pulses[i].duration_ns / 1e3).collect();
                let mut bw: Vec<f64> = g.iter().map(|&i| pulses[i].bandwidth_hz / 1e6).collect();
                let mut mc: HashMap<String, usize> = HashMap::new();
                for &i in g {
                    *mc.entry(norm_mod(&pulses[i].modulation)).or_default() += 1;
                }
                let m = mc.into_iter().max_by_key(|x| x.1).map(|x| x.0).unwrap_or_default();
                (m, quantile(&mut w, 0.5), quantile(&mut bw, 0.5), g.len())
            };
            for (a, b) in &hits {
                let (ma, wa, ba, na) = st(&groups[*a]);
                let (mb, wb, bb, nb) = st(&groups[*b]);
                let (r, fit, coll) = train_fit(pulses, &groups, &[*a, *b], cfg);
                // overlap: B pulse starting while an A pulse is on air (or vice versa)
                let ov = pulse_overlap(pulses, &groups[*a], &groups[*b]);
                let fq = |g: &Vec<usize>, q: f64| { let mut v: Vec<f64> = g.iter().map(|&i| pulses[i].freq_hz / 1e6).collect(); quantile(&mut v, q) };
                let pw_ = |g: &Vec<usize>| { let mut v: Vec<f64> = g.iter().map(|&i| pulses[i].power_dbm).collect(); quantile(&mut v, 0.5) };
                let fgap = (fq(&groups[*a], 0.05) - fq(&groups[*b], 0.95)).max(fq(&groups[*b], 0.05) - fq(&groups[*a], 0.95)).max(0.0);
                let dp = (pw_(&groups[*a]) - pw_(&groups[*b])).abs();
                eprintln!("COH|{ma}|{mb}|{wa:.2}|{wb:.2}|{ba:.2}|{bb:.2}|{na}|{nb}|{r:.3}|{:.3}|{:.3}|{:.4}|{:.4}|{ov:.4}|{fgap:.1}|{dp:.2}", fit[0], fit[1], coll[0], coll[1]);
            }
        }
        for (a, b) in hits {
            uf.union(a, b);
        }
    }
    groups = regroup(groups, &mut uf);
    groups.retain(|g| g.len() >= cfg.min_group);
    debug!("classic_v2: {} groups after coherence merge", groups.len());
    emit_dur("classic.coherence", t.elapsed());
    t = Instant::now();

    if cfg.v3 {
        groups = v3_rescue_and_join(pulses, &pulse_mod, groups, sparse, cfg, &dctx);
        groups = if dctx.df && cfg.doa_final_split {
            groups.into_par_iter().flat_map_iter(|g| doa_track_split(pulses, g, &dctx, cfg)).collect()
        } else {
            groups.into_iter().flat_map(|g| doa_split(pulses, g)).collect()
        };
        groups.retain(|g| g.len() >= cfg.min_group);
        debug!("classic_v3: {} groups after agile join", groups.len());
        emit_dur("classic.v3_join", t.elapsed());
        t = Instant::now();
    }

    // 5. naming; one emitter per library id
    let mut known: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut out: Vec<(Option<i64>, Vec<usize>)> = Vec::new();
    let names: Vec<Option<i64>> = groups
        .par_iter()
        .map(|g| {
            let ps: Vec<&FlowPulse> = g.iter().map(|&i| &pulses[i]).collect();
            library_id(&ps, &sorted_toas(pulses, g), cfg)
        })
        .collect();
    for (g, name) in groups.into_iter().zip(names) {
        match name {
            Some(id) => known.entry(id).or_default().extend(g),
            None => out.push((None, g)),
        }
    }
    let mut res: Vec<(Option<i64>, Vec<usize>)> = known.into_iter().map(|(k, v)| (Some(k), v)).collect();
    res.extend(out);
    emit_dur("classic.naming", t.elapsed());
    res
}

/// v3 step A + B.
/// A. Pulses from sparse cells (an agile radar spreads thin over 1 MHz cells)
///    are regrouped on (modulation, PW, BW) only, ignoring the carrier.
/// B. Pieces with the same modulation, PW (within merge_pw_rel) and
///    overlapping BW are candidates for one radar whatever their carrier.
///    The candidate set is merged into one pulse train; pieces whose pulses
///    do not fit that train's gap pattern are pushed out and retried as
///    their own set. One radar's hops/bursts rebuild its PRI pattern; a
///    different radar's pulses fall at random offsets and fit badly.
fn v3_rescue_and_join(
    pulses: &[FlowPulse],
    pulse_mod: &[i64],
    mut groups: Vec<Vec<usize>>,
    sparse: Vec<usize>,
    cfg: &ClassicCfg,
    dctx: &DoaCtx,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;
    let lw = (1.0 + cfg.pw_rel).ln();
    let key = |i: usize| {
        let p = &pulses[i];
        (pulse_mod[i], ((p.duration_ns / 1e3).max(0.05).ln() / lw).round() as i64, (p.bandwidth_hz / 1e6).round() as i64)
    };
    // A. rescue
    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for i in sparse {
        cells.entry(key(i)).or_default().push(i);
    }
    let keys: Vec<(i64, i64, i64)> = cells.keys().cloned().collect();
    let idx: HashMap<(i64, i64, i64), usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let mut uf = Uf::new(keys.len());
    for (i, (m, w, b)) in keys.iter().enumerate() {
        for dw in -1..=1 {
            for db in -1..=1 {
                if let Some(&j) = idx.get(&(*m, w + dw, b + db)) {
                    uf.union(i, j);
                }
            }
        }
    }
    let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        comps.entry(uf.find(i)).or_default().extend_from_slice(&cells[k]);
    }
    let rescued: Vec<Vec<usize>> = comps
        .into_values()
        .filter(|g| {
            if g.len() < cfg.rescue_min.max(cfg.min_group) {
                return false;
            }
            let mut f: Vec<f64> = g.iter().map(|&i| pulses[i].freq_hz / 1e6).collect();
            quantile(&mut f.clone(), 0.95) - quantile(&mut f, 0.05) >= cfg.rescue_span_mhz
        })
        .collect();
    if dctx.df {
        // pieces must be direction-pure before joining: a piece that still
        // holds two directions has a meaningless centre. Rescued groups
        // (carrier ignored) mix directions most; the join rebuilds each radar.
        if cfg.doa_presplit {
            groups = groups.into_par_iter().flat_map_iter(|g| doa_track_split(pulses, g, dctx, cfg)).collect();
        }
        groups.extend(
            rescued
                .into_iter()
                .flat_map(|g| doa_track_split(pulses, g, dctx, cfg))
                .filter(|g| g.len() >= cfg.min_group),
        );
    } else {
        groups.extend(rescued);
    }

    // B. candidate sets: same modulation, PW close, BW overlap
    let stat = |g: &Vec<usize>| {
        let mut b: Vec<f64> = g.iter().map(|&i| pulses[i].bandwidth_hz).collect();
        let mut w: Vec<f64> = g.iter().map(|&i| pulses[i].duration_ns).collect();
        let mut mc: HashMap<i64, usize> = HashMap::new();
        for &i in g {
            *mc.entry(pulse_mod[i]).or_default() += 1;
        }
        let m = mc.into_iter().max_by_key(|x| (x.1, -x.0)).map(|x| x.0).unwrap_or(-1);
        (m, quantile(&mut w, 0.5), quantile(&mut b, 0.05), quantile(&mut b, 0.95))
    };
    let st: Vec<_> = groups.par_iter().map(stat).collect();
    let jtr: Vec<Vec<(f64, f64)>> =
        if dctx.df { groups.par_iter().map(|g| doa_train(pulses, g)).collect() } else { Vec::new() };
    let conflict = |a: usize, b: usize| dctx.df && doa_conflict(&jtr[a], &jtr[b], dctx, cfg);
    let mut uf = Uf::new(groups.len());
    for a in 0..groups.len() {
        for b in a + 1..groups.len() {
            let (ma, wa, ba0, ba1) = st[a];
            let (mb, wb, bb0, bb1) = st[b];
            if ma == mb
                && (wa - wb).abs() <= cfg.merge_pw_rel * wa.min(wb)
                && ba0 - cfg.merge_bw_hz <= bb1
                && bb0 - cfg.merge_bw_hz <= ba1
                && !conflict(a, b)
            {
                uf.union(a, b);
            }
        }
    }
    let mut sets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for g in 0..groups.len() {
        sets.entry(uf.find(g)).or_default().push(g);
    }
    // each candidate set is independent: decide in parallel, groups read-only;
    // every decision is a list of group indices, assembled in set order below
    let groups_ref = &groups;
    let decide = |set: Vec<usize>| -> Vec<Vec<usize>> {
        let groups = groups_ref;
        let mut out: Vec<Vec<usize>> = Vec::new();
        let mut todo = vec![set];
        while let Some(mut cand) = todo.pop() {
            if cand.len() == 1 {
                out.push(vec![cand[0]]);
                continue;
            }
            // DF: pieces seen in another direction than the largest piece leave
            if dctx.df {
                let big = (0..cand.len()).max_by_key(|k| groups[cand[*k]].len()).unwrap();
                let gb = cand[big];
                let off: Vec<usize> = cand.iter().copied().filter(|g| *g != gb && conflict(*g, gb)).collect();
                if !off.is_empty() {
                    cand.retain(|g| !off.contains(g));
                    todo.push(off);
                    if cand.len() == 1 {
                        out.push(vec![cand[0]]);
                        continue;
                    }
                }
            }
            // base-gap check against the largest piece
            let base: Vec<Option<Vec<f64>>> = cand.iter().map(|g| own_gap_set(pulses, &groups[*g], cfg)).collect();
            let big = (0..cand.len()).max_by_key(|k| groups[cand[*k]].len()).unwrap();
            if let Some(b0) = base[big].clone() {
                let ref_t = sorted_toas(pulses, &groups[cand[big]]);
                let ref_med = own_gap_q(&ref_t, cfg, 0.5).unwrap_or(0.0);
                let ok = |k: usize| match &base[k] {
                    None => true,
                    // on air together with the largest piece: the collision
                    // test decides; the base gap only for time-disjoint pieces
                    Some(_) if k == big => true,
                    Some(_) if {
                        let tk = sorted_toas(pulses, &groups[cand[k]]);
                        let w = 2.0 * ref_med.max(own_gap_q(&tk, cfg, 0.5).unwrap_or(0.0));
                        let near = tk
                            .iter()
                            .filter(|x| {
                                let j = ref_t.partition_point(|y| y < x);
                                (j > 0 && *x - ref_t[j - 1] <= w) || (j < ref_t.len() && ref_t[j] - *x <= w)
                            })
                            .count();
                        near as f64 >= 0.5 * tk.len() as f64
                    } => true,
                    // time-disjoint pieces of one radar share a main gap value
                    // (a sparse sample of the same radar shows whole multiples)
                    Some(b) => b.iter().any(|x| {
                        b0.iter().any(|y| {
                            let r = x / y;
                            let m = r.round();
                            (1.0..=8.0).contains(&m) && (r - m).abs() <= cfg.gap_ratio_tol * m.sqrt()
                        })
                    }),
                };
                let off: Vec<usize> = (0..cand.len()).filter(|k| !ok(*k)).map(|k| cand[k]).collect();
                if !off.is_empty() && std::env::var("CLASSIC_V3_DEBUG").is_ok() {
                    eprintln!("v3 base-gap off: {:?} base={:?}", off.iter().map(|g| groups[*g].len()).collect::<Vec<_>>(), base);
                }
                if !off.is_empty() {
                    cand.retain(|g| !off.contains(g));
                    todo.push(off);
                    if cand.len() == 1 {
                        out.push(vec![cand[0]]);
                        continue;
                    }
                }
            }
            loop {
                let (r, fit, coll) = train_fit(pulses, groups, &cand, cfg);
                if std::env::var("CLASSIC_V3_DEBUG").is_ok() {
                    let sz: Vec<usize> = cand.iter().map(|g| groups[*g].len()).collect();
                    eprintln!("v3 cand sizes={sz:?} r={r:.3} fit={fit:.2?} coll={coll:.3?}");
                }
                let keep: Vec<usize> = cand
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| r >= cfg.reg_min && fit[*k] >= cfg.reg_fit * r && coll[*k] <= cfg.coll_max)
                    .map(|(_, g)| *g)
                    .collect();
                if keep.len() == cand.len() {
                    // one clean train: join every piece
                    out.push(cand.clone());
                    break;
                }
                if keep.len() <= 1 {
                    // no clean train: push out the worst-fitting piece, retry the rest
                    // worst = most collisions, then worst fit
                    let worst = (0..cand.len())
                        .max_by(|a, b| (coll[*a], -fit[*a]).partial_cmp(&(coll[*b], -fit[*b])).unwrap())
                        .unwrap();
                    let w = cand.remove(worst);
                    out.push(vec![w]);
                    if cand.len() == 1 {
                        out.push(vec![cand[0]]);
                        break;
                    }
                    continue;
                }
                let rest: Vec<usize> = cand.iter().filter(|g| !keep.contains(g)).copied().collect();
                todo.push(rest);
                cand = keep;
            }
        }
        out
    };
    let sets: Vec<Vec<usize>> = sets.into_values().collect();
    let decided: Vec<Vec<Vec<usize>>> = sets.into_par_iter().map(decide).collect();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for lists in decided {
        for l in lists {
            let mut j = Vec::new();
            for g in l {
                j.append(&mut groups[g]);
            }
            out.push(j);
        }
    }
    out.retain(|g| !g.is_empty());
    out
}

/// v3 DOA context, measured from the data: is the receiver a direction
/// finder and how noisy are its bearings. Not usable -> every DOA rule is off.
///
/// Bearings are judged per *track*: pulses of one group walked in time order,
/// each joining the open track whose last bearings are within `track_tol`
/// (a track ends after a silence). A rotating antenna that only hears a
/// radar while pointing near it gives one track per look ("dwell"), with
/// the bearing sweeping across the beam; its centre (median) is stable from
/// look to look. A fixed direction finder gives one long track per radar.
#[derive(Clone, Copy, Debug, Default)]
pub struct DoaCtx {
    pub df: bool,
    /// median |DOA difference| of time-adjacent pulses in one group (deg)
    pub noise: f64,
    /// bearings hardly move while a radar is heard (fixed direction finder,
    /// not a spinning antenna): bearings at different times may be compared
    pub fixed: bool,
    /// median bearing drift while one radar is heard (deg/s)
    pub rate: f64,
    /// track-following tolerance (deg)
    pub track_tol: f64,
    /// two pulse trains whose bearings at the same moment differ by more
    /// than this (median) are different radars (deg)
    pub tol: f64,
    /// the same pulse seen twice (< pri_min_gap apart): median |DOA
    /// difference| of the two reports and their share of time-adjacent pairs.
    /// Two receivers that disagree on the same pulse = DOA not usable.
    pub dup_diff: f64,
    pub dup_share: f64,
}

/// Signed circular difference a - b in (-180, 180].
fn circ_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        d - 360.0
    } else {
        d
    }
}

/// Weighted circular median of bearings (deg): median of offsets around the
/// weighted circular mean.
fn circ_median(v: &[(f64, f64)]) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for (x, w) in v {
        s += w * x.to_radians().sin();
        c += w * x.to_radians().cos();
    }
    let m = s.atan2(c).to_degrees();
    let off: Vec<(f64, f64)> = v.iter().map(|(x, w)| (circ_diff(*x, m), *w)).collect();
    (m + weighted_median(off)).rem_euclid(360.0)
}

fn weighted_median(mut v: Vec<(f64, f64)>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let half = v.iter().map(|x| x.1).sum::<f64>() / 2.0;
    let mut acc = 0.0;
    for (x, w) in &v {
        acc += w;
        if acc >= half {
            return *x;
        }
    }
    v.last().unwrap().0
}

/// Pulses of a group with a finite DOA, in time order.
fn doa_order(pulses: &[FlowPulse], g: &[usize]) -> Vec<usize> {
    let mut o: Vec<usize> = g.iter().copied().filter(|&i| pulses[i].doa_deg.is_finite()).collect();
    o.sort_unstable_by(|a, b| pulses[*a].toa_ns.partial_cmp(&pulses[*b].toa_ns).unwrap());
    o
}

/// Bearing tracks of time-ordered pulses (see `DoaCtx`). A track ends after
/// a silence of max(doa_track_gap_ms, 10 x the group's median gap), at most 2 s.
fn doa_tracks(pulses: &[FlowPulse], ord: &[usize], tol: f64, cfg: &ClassicCfg) -> Vec<Vec<usize>> {
    if ord.is_empty() {
        return Vec::new();
    }
    let mut gaps: Vec<f64> = ord.windows(2).map(|w| pulses[w[1]].toa_ns - pulses[w[0]].toa_ns).collect();
    let med = if gaps.is_empty() { 0.0 } else { quantile(&mut gaps, 0.5) };
    let gap = (cfg.doa_track_gap_ms * 1e6).max(10.0 * med).min(2e9);
    // (last toa, last 3 bearings, members)
    let mut tracks: Vec<(f64, Vec<f64>, Vec<usize>)> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for &i in ord {
        let (t, d) = (pulses[i].toa_ns, pulses[i].doa_deg);
        active.retain(|k| t - tracks[*k].0 <= gap);
        let mut best: Option<(usize, f64)> = None;
        for &k in &active {
            let rc = &tracks[k].1;
            let last = *rc.last().unwrap();
            let mut ds: Vec<f64> = rc.iter().map(|x| circ_diff(*x, last)).collect();
            let e = circ_diff(d, last + quantile(&mut ds, 0.5)).abs();
            if best.map_or(true, |b| e < b.1) {
                best = Some((k, e));
            }
        }
        match best {
            Some((k, e)) if e <= tol => {
                let tr = &mut tracks[k];
                tr.0 = t;
                tr.1.push(d);
                if tr.1.len() > 3 {
                    tr.1.remove(0);
                }
                tr.2.push(i);
            }
            _ => {
                tracks.push((t, vec![d], vec![i]));
                active.push(tracks.len() - 1);
            }
        }
    }
    tracks.into_iter().map(|t| t.2).collect()
}

fn doa_ctx(pulses: &[FlowPulse], groups: &[Vec<usize>], cfg: &ClassicCfg) -> DoaCtx {
    use rayon::prelude::*;
    let win = cfg.doa_win_ms * 1e6;
    let dup_ns = cfg.pri_min_gap_us * 1e3;
    // pass 1: time-adjacent noise + duplicate disagreement
    let per: Vec<(f64, f64, Vec<f64>, usize, f64)> = groups
        .par_iter()
        .filter_map(|g| {
            let o = doa_order(pulses, g);
            if o.len() < 50 {
                return None;
            }
            let (mut d1, mut dd) = (Vec::new(), Vec::new());
            for k in 0..o.len() - 1 {
                let a = &pulses[o[k]];
                let b = &pulses[o[k + 1]];
                if b.toa_ns - a.toa_ns < dup_ns {
                    dd.push(circ_diff(b.doa_deg, a.doa_deg).abs());
                    continue;
                }
                // closest bearing among the next pulses: a group may still hold
                // several interleaved radars; one of the next pulses is the same one
                let best = o[k + 1..o.len().min(k + 1 + cfg.doa_next)]
                    .iter()
                    .map(|&j| &pulses[j])
                    .filter(|q| q.toa_ns - a.toa_ns >= dup_ns && q.toa_ns - a.toa_ns <= win)
                    .map(|q| circ_diff(q.doa_deg, a.doa_deg).abs())
                    .fold(f64::INFINITY, f64::min);
                if best.is_finite() {
                    d1.push(best);
                }
            }
            if d1.len() < 20 {
                return None;
            }
            let np = d1.len() + dd.len();
            Some((o.len() as f64, quantile(&mut d1, 0.5), dd, np, 0.0))
        })
        .collect();
    if per.is_empty() {
        return DoaCtx::default();
    }
    let noise = weighted_median(per.iter().map(|x| (x.1, x.0)).collect());
    let mut dd: Vec<f64> = per.iter().flat_map(|x| x.2.iter().copied()).collect();
    let pairs: usize = per.iter().map(|x| x.3).sum();
    let dup_share = dd.len() as f64 / pairs.max(1) as f64;
    let dup_diff = if dd.is_empty() { 0.0 } else { quantile(&mut dd, 0.5) };
    let track_tol = cfg.doa_track_tol_min.max(cfg.doa_tol_k * noise);
    // two reports of the same pulse that disagree: bearings are not usable
    // (the same pulse at the same instant: must agree within the moment noise)
    let dup_bad = dup_share >= 0.05 && dup_diff > cfg.doa_tol_min.max(track_tol);
    let df = noise <= cfg.df_noise_max && !dup_bad;
    // bearing drift while a radar is heard: least-squares slope of each
    // bearing track (>= 20 pulses, >= 10 ms); pulse-weighted median
    let track_tol = cfg.doa_track_tol_min.max(cfg.doa_tol_k * noise);
    let slopes: Vec<(f64, f64)> = if df {
        groups
            .par_iter()
            .flat_map_iter(|g| {
                let o = doa_order(pulses, g);
                doa_tracks(pulses, &o, track_tol, cfg)
                    .into_iter()
                    .filter_map(|t| {
                        let (t0, t1) = (pulses[t[0]].toa_ns, pulses[*t.last().unwrap()].toa_ns);
                        if t.len() < 20 || t1 - t0 < 1e7 {
                            return None;
                        }
                        let d0 = pulses[t[0]].doa_deg;
                        let n = t.len() as f64;
                        let xs: Vec<f64> = t.iter().map(|&i| (pulses[i].toa_ns - t0) / 1e9).collect();
                        let ys: Vec<f64> = t.iter().map(|&i| circ_diff(pulses[i].doa_deg, d0)).collect();
                        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
                        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
                        let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
                        if sxx <= 0.0 {
                            return None;
                        }
                        Some(((sxy / sxx).abs(), n))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        Vec::new()
    };
    // no long track to measure: unknown drift -> treated as spinning (no splits)
    let rate = if slopes.iter().map(|x| x.1).sum::<f64>() < 200.0 { f64::INFINITY } else { weighted_median(slopes) };
    DoaCtx {
        df,
        fixed: df && rate <= cfg.doa_fixed_rate,
        noise,
        rate,
        track_tol,
        tol: cfg.doa_tol_min.max(track_tol),
        dup_diff,
        dup_share,
    }
}

/// TOA-sorted (toa_ns, doa_deg) of a group's pulses with a bearing.
fn doa_train(pulses: &[FlowPulse], g: &[usize]) -> Vec<(f64, f64)> {
    doa_order(pulses, g).into_iter().map(|i| (pulses[i].toa_ns, pulses[i].doa_deg)).collect()
}

/// Bearings of two trains compared at the same moments only: for sampled
/// pulses of the smaller train, the other train's bearing at that time
/// (interpolated between its neighbours, both within `doa_win_ms`).
/// Some(true) = different directions at the same time = different radars.
/// None = not on air together often enough (no evidence). Bearings seen at
/// different times are never compared: a spinning direction finder can hear
/// one radar from several directions (side lobes) on different looks.
fn doa_differs(a: &[(f64, f64)], b: &[(f64, f64)], ctx: &DoaCtx, cfg: &ClassicCfg, samples: usize) -> Option<bool> {
    let (a, b) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if a.len() < 20 || b.len() < 20 {
        return None;
    }
    let win = cfg.doa_win_ms * 1e6;
    let n = samples.min(a.len());
    let mut r = Vec::with_capacity(n);
    for k in 0..n {
        let i = if n == 1 { 0 } else { k * (a.len() - 1) / (n - 1) };
        let (t, d) = a[i];
        let j = b.partition_point(|x| x.0 < t);
        if j == 0 || j >= b.len() {
            continue;
        }
        let (p, q) = (b[j - 1], b[j]);
        if t - p.0 > win || q.0 - t > win || q.0 <= p.0 {
            continue;
        }
        let est = p.1 + circ_diff(q.1, p.1) * (t - p.0) / (q.0 - p.0);
        r.push(circ_diff(d, est).abs());
    }
    if r.len() >= cfg.doa_min_evidence {
        return Some(quantile(&mut r, 0.5) > ctx.tol);
    }
    if ctx.fixed {
        // fixed direction finder: a radar keeps its bearing, so the whole
        // trains can be compared even when they are not on air together
        let ca = circ_median(&a.iter().map(|x| (x.1, 1.0)).collect::<Vec<_>>());
        let cb = circ_median(&b.iter().map(|x| (x.1, 1.0)).collect::<Vec<_>>());
        return Some(circ_diff(ca, cb).abs() > ctx.tol);
    }
    None
}

/// Two trains heard from clearly different directions at the same time.
fn doa_conflict(a: &[(f64, f64)], b: &[(f64, f64)], ctx: &DoaCtx, cfg: &ClassicCfg) -> bool {
    ctx.df && doa_differs(a, b, ctx, cfg, 300) == Some(true)
}

/// v3 step C (direction finder): split a group whose pulses come from more
/// than one direction AT THE SAME TIME. Pulses are walked in time order into
/// bearing tracks; tracks go into parts (largest first: a track joins the
/// part with the most same-direction evidence and no conflict; a track with
/// no evidence either way joins the largest part without conflict; a track
/// in conflict with every part starts a new one). Parts below
/// `doa_split_share` of the group (or `min_group`) are kept apart (the
/// min_group filter drops them) unless `doa_fold_small`.
fn doa_track_split(pulses: &[FlowPulse], g: Vec<usize>, ctx: &DoaCtx, cfg: &ClassicCfg) -> Vec<Vec<usize>> {
    // spinning antenna: one radar can show at two bearings (side lobes,
    // reflections) -> bearings only veto joins, they never split a group
    if !ctx.fixed || g.len() < 2 * cfg.min_group {
        return vec![g];
    }
    let o = doa_order(pulses, &g);
    let mut tr: Vec<Vec<usize>> = doa_tracks(pulses, &o, ctx.track_tol, cfg).into_iter().filter(|t| t.len() >= 20).collect();
    if tr.len() < 2 {
        return vec![g];
    }
    tr.sort_by_key(|t| std::cmp::Reverse(t.len()));
    let trains: Vec<Vec<(f64, f64)>> =
        tr.iter().map(|t| t.iter().map(|&i| (pulses[i].toa_ns, pulses[i].doa_deg)).collect()).collect();
    let mut cl: Vec<Vec<usize>> = Vec::new();
    for x in 0..tr.len() {
        let mut pick: Option<(usize, usize)> = None;
        let mut free: Option<usize> = None;
        for (c, members) in cl.iter().enumerate() {
            let (mut same, mut diff) = (0usize, 0usize);
            for &y in members {
                match doa_differs(&trains[x], &trains[y], ctx, cfg, 100) {
                    Some(true) => diff += 1,
                    Some(false) => same += 1,
                    None => {}
                }
            }
            if diff > 0 {
                continue;
            }
            if same > 0 && pick.map_or(true, |p| same > p.1) {
                pick = Some((c, same));
            }
            if free.is_none() {
                free = Some(c);
            }
        }
        match pick.map(|p| p.0).or(free) {
            Some(c) => cl[c].push(x),
            None => cl.push(vec![x]),
        }
    }
    if std::env::var("CLASSIC_DOA_DEBUG").is_ok() {
        eprintln!("DOA split n={} parts {:?}", g.len(), cl.iter().map(|c| c.iter().map(|x| tr[*x].len()).sum::<usize>()).collect::<Vec<_>>());
    }
    if cl.len() < 2 {
        return vec![g];
    }
    let mut parts: Vec<Vec<usize>> = cl.iter().map(|c| c.iter().flat_map(|x| tr[*x].iter().copied()).collect()).collect();
    parts.sort_by_key(|p| std::cmp::Reverse(p.len()));
    let min_n = (cfg.doa_split_share * g.len() as f64).max(cfg.min_group as f64) as usize;
    let mut keep: Vec<Vec<usize>> = Vec::new();
    let mut small: Vec<Vec<usize>> = Vec::new();
    for p in parts {
        if keep.is_empty() || p.len() >= min_n {
            keep.push(p);
        } else {
            small.push(p);
        }
    }
    if keep.len() < 2 && (cfg.doa_fold_small || small.is_empty()) {
        return vec![g];
    }
    let placed: std::collections::HashSet<usize> = keep.iter().chain(small.iter()).flatten().copied().collect();
    if cfg.doa_fold_small {
        keep[0].extend(small.into_iter().flatten());
    } else {
        keep.extend(small);
    }
    // fixed direction finder: pulses of short tracks go to the part with the
    // nearest bearing; pulses without a bearing go to the largest part
    let cen: Vec<f64> = keep.iter().map(|p| circ_median(&p.iter().map(|&i| (pulses[i].doa_deg, 1.0)).collect::<Vec<_>>())).collect();
    for &i in &g {
        if placed.contains(&i) {
            continue;
        }
        let d = pulses[i].doa_deg;
        let k = if d.is_finite() {
            (0..cen.len()).min_by(|a, b| circ_diff(d, cen[*a]).abs().partial_cmp(&circ_diff(d, cen[*b]).abs()).unwrap()).unwrap()
        } else {
            0
        };
        keep[k].push(i);
    }
    for p in keep.iter_mut() {
        p.sort_unstable();
    }
    keep
}

/// v3 step C: two radars of the same type (same pulses, both agile) are one
/// group after the timing steps; with a direction finder their angles still
/// differ. Split a group only when its DOA histogram (5 deg bins) holds two
/// or more tight peaks >= 15 deg apart, each >= 15 % of pulses, and >= 90 %
/// of all pulses sit within +-1 bin of a peak. Omni / random DOA (flat
/// histogram) never splits.
fn doa_split(pulses: &[FlowPulse], g: Vec<usize>) -> Vec<Vec<usize>> {
    let n = g.len();
    if n < 100 {
        return vec![g];
    }
    let bin = |i: usize| (((pulses[i].doa_deg % 360.0 + 360.0) % 360.0) / 5.0).floor() as i64 % 72;
    let mut h = [0usize; 72];
    for &i in &g {
        if pulses[i].doa_deg.is_finite() {
            h[bin(i) as usize] += 1;
        }
    }
    let at = |k: i64| h[k.rem_euclid(72) as usize];
    let near3 = |k: i64| at(k - 1) + at(k) + at(k + 1);
    let mut peaks: Vec<i64> = (0..72i64)
        .filter(|&k| at(k) >= at(k - 1) && at(k) > at(k + 1) && near3(k) as f64 >= 0.15 * n as f64)
        .collect();
    peaks.sort_by_key(|k| std::cmp::Reverse(near3(*k)));
    let mut keep: Vec<i64> = Vec::new();
    for p in peaks {
        if keep.iter().all(|q| {
            let d = (p - q).rem_euclid(72);
            d.min(72 - d) >= 3
        }) {
            keep.push(p);
        }
    }
    let covered: usize = keep.iter().map(|k| near3(*k)).sum();
    if keep.len() < 2 || (covered as f64) < 0.9 * n as f64 {
        return vec![g];
    }
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); keep.len()];
    for i in g {
        let b = bin(i);
        let k = (0..keep.len())
            .min_by_key(|j| {
                let d = (b - keep[*j]).rem_euclid(72);
                d.min(72 - d)
            })
            .unwrap();
        out[k].push(i);
    }
    out
}

/// Main own gap values of a piece (ns): 2 % log bins holding >= 8 % of its
/// gaps (at most 6). None when too few gaps or no clear main gap (no evidence).
fn own_gap_set(pulses: &[FlowPulse], g: &[usize], cfg: &ClassicCfg) -> Option<Vec<f64>> {
    let ts = sorted_toas(pulses, g);
    let min_gap = cfg.pri_min_gap_us * 1e3;
    let mut gp: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|d| *d >= min_gap).collect();
    if gp.len() < 20 {
        return None;
    }
    // PRI-scale gaps only: long gaps (scan / dwell breaks) are shared by
    // unrelated radars with the same scan period
    let p5 = quantile(&mut gp.clone(), 0.05);
    gp.retain(|d| *d <= 5.0 * p5);
    if gp.len() < 20 {
        return None;
    }
    let lb = 1.02f64.ln();
    let mut h: HashMap<i64, usize> = HashMap::new();
    for d in &gp {
        *h.entry((d.ln() / lb).round() as i64).or_default() += 1;
    }
    let mut c: Vec<(i64, usize)> = h.into_iter().filter(|x| x.1 as f64 >= 0.08 * gp.len() as f64).collect();
    c.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if c.is_empty() {
        return None;
    }
    Some(c.iter().take(6).map(|x| (x.0 as f64 * lb).exp()).collect())
}

/// Quantile of a sorted train's own gaps (ns), duplicates skipped.
fn own_gap_q(ts: &[f64], cfg: &ClassicCfg, q: f64) -> Option<f64> {
    let min_gap = cfg.pri_min_gap_us * 1e3;
    let mut gp: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|d| *d >= min_gap).collect();
    if gp.len() < 20 {
        return None;
    }
    Some(quantile(&mut gp, q))
}

/// Merge the pieces' pulses into one train; gap histogram on log bins.
/// Returns (top-k share of all gaps, per-piece share of its pulses whose
/// gaps before/after fall in those top-k bins).
fn train_fit(pulses: &[FlowPulse], groups: &[Vec<usize>], cand: &[usize], cfg: &ClassicCfg) -> (f64, Vec<f64>, Vec<f64>) {
    let mut t: Vec<(f64, usize)> = Vec::new();
    for (k, g) in cand.iter().enumerate() {
        t.extend(groups[*g].iter().map(|&i| (pulses[i].toa_ns, k)));
    }
    t.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // the same pulse seen twice (< pri_min_gap) counts once
    let min_gap = cfg.pri_min_gap_us * 1e3;
    t.dedup_by(|b, a| b.0 - a.0 < min_gap);
    if t.len() < 3 {
        return (0.0, vec![0.0; cand.len()], vec![1.0; cand.len()]);
    }
    // shortest own gap (5th pct) over the pieces -> collision distance
    let mut own_min = f64::INFINITY;
    for g in cand {
        let ts = sorted_toas(pulses, &groups[*g]);
        let mut gp: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|d| *d >= min_gap).collect();
        if gp.len() >= 20 {
            own_min = own_min.min(quantile(&mut gp, 0.05));
        }
    }
    // interleaved pieces (n-modal, hops) each have own gaps of N x PRI:
    // cap by the merged train's median gap (= the PRI for one radar)
    let mut mg: Vec<f64> = t.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let med = quantile(&mut mg, 0.5);
    let thr = if own_min.is_finite() { cfg.coll_rel * own_min.min(med) } else { 0.0 };
    let (mut hit, mut cnt) = (vec![0.0f64; cand.len()], vec![0.0f64; cand.len()]);
    for j in 0..t.len() {
        let k = t[j].1;
        let near = |o: usize| t[o].1 != k && (t[o].0 - t[j].0).abs() < thr;
        cnt[k] += 1.0;
        if (j >= 1 && near(j - 1)) || (j + 1 < t.len() && near(j + 1)) {
            hit[k] += 1.0;
        }
    }
    let coll: Vec<f64> = hit.iter().zip(&cnt).map(|(h, c)| if *c > 0.0 { h / c } else { 1.0 }).collect();
    let lb = (1.0 + cfg.reg_bin_rel).ln();
    let bins: Vec<i64> = t.windows(2).map(|w| ((w[1].0 - w[0].0).max(1.0).ln() / lb).round() as i64).collect();
    let mut h: HashMap<i64, usize> = HashMap::new();
    for b in &bins {
        *h.entry(*b).or_default() += 1;
    }
    let mut c: Vec<(i64, usize)> = h.into_iter().collect();
    c.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top: std::collections::HashSet<i64> = c.iter().take(cfg.reg_top).map(|x| x.0).collect();
    let r = c.iter().take(cfg.reg_top).map(|x| x.1).sum::<usize>() as f64 / bins.len() as f64;
    let good: Vec<bool> = bins.iter().map(|b| top.contains(b)).collect();
    let (mut num, mut den) = (vec![0.0f64; cand.len()], vec![0.0f64; cand.len()]);
    for j in 0..t.len() {
        let k = t[j].1;
        if j >= 1 {
            num[k] += good[j - 1] as u8 as f64;
            den[k] += 1.0;
        }
        if j + 1 < t.len() {
            num[k] += good[j] as u8 as f64;
            den[k] += 1.0;
        }
    }
    (r, num.iter().zip(&den).map(|(n, d)| if *d > 0.0 { n / d } else { 0.0 }).collect(), coll)
}

/// Share of the smaller group's pulses that start while a pulse of the other
/// group is still on air (TOA within the other's pulse width) — impossible
/// for one transmitter.
fn pulse_overlap(pulses: &[FlowPulse], ga: &[usize], gb: &[usize]) -> f64 {
    let (small, big) = if ga.len() <= gb.len() { (ga, gb) } else { (gb, ga) };
    let mut bt: Vec<(f64, f64)> = big.iter().map(|&i| (pulses[i].toa_ns, pulses[i].duration_ns)).collect();
    bt.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let mut hit = 0usize;
    for &i in small {
        let t = pulses[i].toa_ns;
        let w = pulses[i].duration_ns;
        let j = bt.partition_point(|x| x.0 <= t);
        let prev_on = j > 0 && t - bt[j - 1].0 < bt[j - 1].1;
        let next_on = j < bt.len() && bt[j].0 - t < w;
        if prev_on || next_on {
            hit += 1;
        }
    }
    hit as f64 / small.len().max(1) as f64
}

fn regroup(groups: Vec<Vec<usize>>, uf: &mut Uf) -> Vec<Vec<usize>> {
    let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.into_iter().enumerate() {
        m.entry(uf.find(i)).or_default().extend(g);
    }
    m.into_values().collect()
}

fn sorted_toas(pulses: &[FlowPulse], g: &[usize]) -> Vec<f64> {
    let mut t: Vec<f64> = g.iter().map(|&i| pulses[i].toa_ns).collect();
    t.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    t
}

/// Detect emitters and write emitters/signals/clusters (PG) + CH clusters.
pub async fn run(state: &AiState, pulses: Vec<FlowPulse>) -> Result<()> {
    use rayon::slice::ParallelSliceMut;
    let t0 = Instant::now();
    let cfg = ClassicCfg::from_env();
    let emitters = detect(&pulses, &cfg);
    let t_detect = t0.elapsed();
    let db = Db::new(state);
    db.begin().await?;
    let empty: [f32; 0] = [];
    let mut all_groups: Vec<(i64, Vec<Member>)> = Vec::new();
    for (lib, mut idx) in emitters.iter().cloned() {
        idx.par_sort_by(|a, b| pulses[*a].toa_ns.partial_cmp(&pulses[*b].toa_ns).unwrap().then(a.cmp(b)));
        let ms: Vec<Member> = idx.iter().map(|&i| Member { p: &pulses[i], emb: &empty }).collect();
        let doas: Vec<f64> = ms.iter().map(|m| m.p.doa_deg).collect();
        let pows: Vec<f64> = ms.iter().map(|m| m.p.power_dbm).collect();
        let (angle, _) = angle_score(&doas, &pows);
        let em = match lib {
            Some(id) => db.insert_emitter(&id.to_string(), signature_id(id)).await?,
            None => db.insert_emitter("unknown_ai", None).await?,
        };
        let sid = db.insert_signal(&ms, Some(em), angle).await?;
        // time slices
        let t_first = ms[0].p.toa_ns;
        let slice_ns = cfg.cluster_sec * 1e9;
        let mut slices: BTreeMap<i64, Vec<Member>> = BTreeMap::new();
        for m in &ms {
            slices.entry(((m.p.toa_ns - t_first) / slice_ns).floor() as i64).or_default().push(m.clone());
        }
        // Fold thin slices (e.g. a trailing duplicate pair) into their
        // neighbour so every cluster row has a measurable pulse train.
        let mut merged: Vec<Vec<Member>> = Vec::new();
        for v in slices.into_values() {
            match merged.last_mut() {
                Some(last) if v.len() < cfg.min_slice || last.len() < cfg.min_slice => last.extend(v),
                _ => merged.push(v),
            }
        }
        all_groups.extend(merged.into_iter().map(|v| (sid, v)));
        db.set_doa(sid, cluster_angle(&doas, &pows)).await?;
    }
    db.write_clusters_gap(&all_groups, cfg.pri_min_gap_us).await?;
    db.commit().await?;
    db.flush_ch(true).await?;
    crate::layer_timing::emit_dur("classic.detect_total", t_detect);
    crate::layer_timing::emit_db("classic");
    crate::layer_timing::emit_dur("classic.total_wall", t0.elapsed());
    info!(
        "classic_v2: {} pulses → {} emitters ({} known) — detect {:.0} ms, total {:.0} ms",
        pulses.len(),
        emitters.len(),
        emitters.iter().filter(|e| e.0.is_some()).count(),
        t_detect.as_secs_f64() * 1e3,
        t0.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(toa_us: f64, f_mhz: f64, doa: f64) -> FlowPulse {
        FlowPulse {
            input_id: 0,
            freq_hz: f_mhz * 1e6,
            power_dbm: -70.0,
            bandwidth_hz: 1e6,
            duration_ns: 10_000.0,
            doa_deg: doa,
            snr_db: 20.0,
            toa_ns: toa_us * 1e3,
            modulation: "lfm_up".into(),
        }
    }

    fn v3cfg() -> ClassicCfg {
        let mut c = ClassicCfg::from_env();
        c.v3 = true;
        c
    }

    /// small deterministic noise in [-a, a]
    fn jit(i: usize, a: f64) -> f64 {
        (((i * 7919 + 13) % 1000) as f64 / 999.0 * 2.0 - 1.0) * a
    }

    #[test]
    fn doa_ctx_fixed_df_vs_spinning_vs_omni() {
        let cfg = v3cfg();
        // fixed DF: radar at 40 deg, noise +-1 deg, 10 s
        let fixed: Vec<FlowPulse> = (0..3000).map(|i| fp(i as f64 * 1000.0, 3000.0, 40.0 + jit(i, 1.0))).collect();
        let g = vec![(0..fixed.len()).collect::<Vec<_>>()];
        let c = doa_ctx(&fixed, &g, &cfg);
        assert!(c.df && c.fixed, "{c:?}");
        // spinning antenna: bearing sweeps 300 deg/s during 60 ms looks once a second
        let mut spin = Vec::new();
        for look in 0..10 {
            for k in 0..300 {
                let t = look as f64 * 1e6 + k as f64 * 200.0; // us
                spin.push(fp(t, 3000.0, (100.0 + 300.0 * k as f64 * 200.0 / 1e6 - 9.0) % 360.0));
            }
        }
        let g = vec![(0..spin.len()).collect::<Vec<_>>()];
        let c = doa_ctx(&spin, &g, &cfg);
        assert!(c.df && !c.fixed, "{c:?}");
        // omni: bearing random per pulse
        let omni: Vec<FlowPulse> = (0..3000).map(|i| fp(i as f64 * 1000.0, 3000.0, (((i * i * 7919 + i * 104729) / 7) % 360) as f64)).collect();
        let g = vec![(0..omni.len()).collect::<Vec<_>>()];
        assert!(!doa_ctx(&omni, &g, &cfg).df);
        // two receivers reporting the same pulse with different bearings
        let mut dup = Vec::new();
        for i in 0..3000 {
            dup.push(fp(i as f64 * 1000.0, 3000.0, 80.0 + jit(i, 1.0)));
            dup.push(fp(i as f64 * 1000.0 + 0.2, 3000.0, 190.0 + jit(i + 1, 1.0)));
        }
        let g = vec![(0..dup.len()).collect::<Vec<_>>()];
        assert!(!doa_ctx(&dup, &g, &cfg).df);
    }

    #[test]
    fn doa_differs_needs_same_moment_unless_fixed() {
        let cfg = v3cfg();
        let a: Vec<(f64, f64)> = (0..500).map(|i| (i as f64 * 1e6, 40.0 + jit(i, 1.0))).collect();
        let b_same: Vec<(f64, f64)> = (0..500).map(|i| (i as f64 * 1e6 + 3e5, 41.0 + jit(i + 3, 1.0))).collect();
        let b_far: Vec<(f64, f64)> = (0..500).map(|i| (i as f64 * 1e6 + 3e5, 120.0 + jit(i + 3, 1.0))).collect();
        let late: Vec<(f64, f64)> = (0..500).map(|i| (6e8 + i as f64 * 1e6, 120.0)).collect();
        let mut ctx = DoaCtx { df: true, fixed: false, tol: 6.0, track_tol: 5.0, ..Default::default() };
        assert_eq!(doa_differs(&a, &b_same, &ctx, &cfg, 300), Some(false));
        assert_eq!(doa_differs(&a, &b_far, &ctx, &cfg, 300), Some(true));
        // spinning: trains never on air together -> no evidence
        assert_eq!(doa_differs(&a, &late, &ctx, &cfg, 300), None);
        // fixed DF: whole-train bearings compared
        ctx.fixed = true;
        assert_eq!(doa_differs(&a, &late, &ctx, &cfg, 300), Some(true));
    }

    #[test]
    fn doa_split_two_same_type_radars_fixed_df_only() {
        let cfg = v3cfg();
        // identical radars interleaved, bearings 40 and 200
        let mut ps = Vec::new();
        for i in 0..2000 {
            ps.push(fp(i as f64 * 1000.0, 3000.0, 40.0 + jit(i, 1.0)));
            ps.push(fp(i as f64 * 1000.0 + 377.0, 3000.0, 200.0 + jit(i + 5, 1.0)));
        }
        let g: Vec<usize> = (0..ps.len()).collect();
        let fixed = DoaCtx { df: true, fixed: true, tol: 6.0, track_tol: 5.0, ..Default::default() };
        let parts = doa_track_split(&ps, g.clone(), &fixed, &cfg);
        assert_eq!(parts.len(), 2);
        for p in &parts {
            let b0 = ps[p[0]].doa_deg;
            assert!(p.iter().all(|&i| circ_diff(ps[i].doa_deg, b0).abs() < 10.0));
        }
        // spinning DF: never split (side lobes)
        let spin = DoaCtx { fixed: false, ..fixed };
        assert_eq!(doa_track_split(&ps, g, &spin, &cfg).len(), 1);
    }

    #[test]
    fn freq_mode_labels() {
        let fixed: Vec<FlowPulse> = (0..200).map(|i| fp(i as f64 * 1000.0, 3000.0, 0.0)).collect();
        let slide: Vec<FlowPulse> = (0..200).map(|i| fp(i as f64 * 1000.0, 3000.0 + i as f64 * 0.3, 0.0)).collect();
        let burst: Vec<FlowPulse> = (0..200).map(|i| fp(i as f64 * 1000.0, 3000.0 + 20.0 * ((i / 20) % 4) as f64, 0.0)).collect();
        let modal: Vec<FlowPulse> = (0..200).map(|i| fp(i as f64 * 1000.0, 3000.0 + 20.0 * (i % 2) as f64, 0.0)).collect();
        let r = |v: &Vec<FlowPulse>| freq_mode(&v.iter().collect::<Vec<_>>());
        assert_eq!(r(&fixed), "fixed");
        assert_eq!(r(&slide), "sliding");
        assert_eq!(r(&burst), "burst");
        assert_eq!(r(&modal), "n_modal");
    }

    #[test]
    fn hops_of_one_radar_fit_one_train_other_radar_collides() {
        // one radar, PRI 1000 us, hopping over 4 carriers in random order
        let mut ps: Vec<FlowPulse> = Vec::new();
        let mut s = 12345u64;
        for i in 0..2000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ps.push(fp(i as f64 * 1000.0, 3000.0 + 50.0 * ((s >> 33) % 4) as f64, 0.0));
        }
        // a different radar, same pulses, PRI 1370 us, own carrier
        for i in 0..1400 {
            ps.push(fp(333.0 + i as f64 * 1370.0, 3400.0, 0.0));
        }
        let cfg = v3cfg();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); 5];
        for (i, p) in ps.iter().enumerate() {
            let k = if p.freq_hz >= 3.39e9 { 4 } else { ((p.freq_hz / 1e6 - 3000.0) / 50.0).round() as usize };
            groups[k].push(i);
        }
        let (r, fit, coll) = train_fit(&ps, &groups, &[0, 1, 2, 3], &cfg);
        assert!(r > 0.9 && fit.iter().all(|f| *f > 0.9) && coll.iter().all(|c| *c == 0.0), "{r} {fit:?} {coll:?}");
        let (_, _, coll) = train_fit(&ps, &groups, &[0, 1, 2, 3, 4], &cfg);
        assert!(coll[4] > 0.3, "other radar must collide: {coll:?}");
    }

    #[test]
    fn doa_split_only_on_tight_bimodal_angles() {
        let two: Vec<FlowPulse> = (0..400).map(|i| fp(i as f64 * 100.0, 3000.0, if i % 2 == 0 { 40.0 } else { 120.0 })).collect();
        let g: Vec<usize> = (0..400).collect();
        assert_eq!(doa_split(&two, g.clone()).len(), 2);
        // omni: angle random per pulse -> never split
        let omni: Vec<FlowPulse> = (0..400).map(|i| fp(i as f64 * 100.0, 3000.0, (i * 137 % 360) as f64)).collect();
        assert_eq!(doa_split(&omni, g).len(), 1);
    }

    fn train(start_us: f64, pri_us: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (start_us + pri_us * i as f64) * 1e3).collect()
    }

    #[test]
    fn coherence_needs_enough_compared_pulses() {
        // A has 400 pulses over 8 s; B only 150 pulses at the very end, so
        // only a couple of A pulses have a B pulse before them.
        let a = train(0.0, 20_000.0, 400);
        let b = train(7_950_000.0, 1_840.0, 150);
        assert!(coherence(&a, &b, 4, 3000, 0) > 0.9, "old behaviour: tiny sample scores ~1.0");
        assert_eq!(coherence(&a, &b, 4, 3000, 20), 0.0, "fewer than 20 compared pulses -> no evidence");
    }

    #[test]
    fn same_train_is_coherent_different_trains_are_not() {
        let cfg = ClassicCfg::from_env();
        let one = train(0.0, 1_000.0, 2000);
        let (even, odd): (Vec<f64>, Vec<f64>) = {
            let e = one.iter().step_by(2).copied().collect();
            let o = one.iter().skip(1).step_by(2).copied().collect();
            (e, o)
        };
        assert!(same_clock(&even, &odd, &cfg), "two halves of one radar keep one clock");
        let other = train(137.0, 1_313.7, 1500);
        assert!(!same_clock(&even, &other, &cfg), "unrelated PRIs are different clocks");
    }

    #[test]
    fn builtin_naming_table_matches_old_tree_order() {
        let t: LibTable = serde_json::from_str(DEFAULT_LIBRARY).unwrap();
        assert_eq!(t.max_bw_mhz, 6.0);
        let ids: Vec<i64> = t.rules.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![1113, 1106, 1107, 1104, 1116, 1109, 1105, 1108]);
        // PSR4 rule needs the long pulse, PSR1 is the fallback at the same carrier
        assert_eq!(t.rules[2].pw_max_ge, Some(86.0));
        assert_eq!(t.rules[3].f, [2748.0, 2752.0]);
        // PSR2 needs short PRI inside the 2858-2892 block, after the MF04 lfm rule
        assert_eq!(t.rules[6].pri_lt, Some(140.0));
    }
}
