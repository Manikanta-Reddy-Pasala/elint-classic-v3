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
    pub min_group: usize,
    pub cluster_sec: f64,
    /// Intervals below this (µs) are the same pulse seen twice — skipped for PRI.
    pub pri_min_gap_us: f64,
    /// Time slices with fewer pulses are folded into their neighbour.
    pub min_slice: usize,
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
            min_group: env_or("CLASSIC_MIN_GROUP", 50),
            cluster_sec: env_or("CLASSIC_CLUSTER_SEC", 1.0),
            pri_min_gap_us: env_or("CLASSIC_PRI_MIN_GAP_US", 1.0),
            min_slice: env_or("CLASSIC_MIN_SLICE", 50),
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
fn coherence(ta: &[f64], tb: &[f64], top: usize, samples: usize) -> f64 {
    if ta.len() < 20 || tb.len() < 20 {
        return 0.0;
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
        return 0.0;
    }
    let mut c: Vec<usize> = h.into_values().collect();
    c.sort_unstable_by(|a, b| b.cmp(a));
    c.iter().take(top).sum::<usize>() as f64 / tot as f64
}

/// Library id from measured group features (no labels, no generator PRI).
fn library_id(ps: &[&FlowPulse], toas_sorted: &[f64]) -> Option<i64> {
    let f_mhz = mode_bin(ps.iter().map(|p| (p.freq_hz / 1e5).round() as i64))? as f64 / 10.0;
    let bw_mhz = mode_bin(ps.iter().map(|p| (p.bandwidth_hz / 1e5).round() as i64)).unwrap_or(0) as f64 / 10.0;
    let mut pw: Vec<f64> = ps.iter().map(|p| p.duration_ns / 1e3).collect();
    // 99th percentile by selection (O(n)) instead of a full sort
    let pw_max = quantile(&mut pw, 0.99);
    let lfm = ps.iter().filter(|p| norm_mod(&p.modulation).replace('_', "") == "lfmup").count() as f64 / ps.len() as f64;
    let pri = pri_mode_us(toas_sorted);
    if bw_mhz >= 6.0 {
        return None;
    }
    if (2898.0..=2905.0).contains(&f_mhz) {
        return Some(1113);
    }
    if (2848.0..=2852.0).contains(&f_mhz) {
        return Some(1106);
    }
    if (2748.0..=2752.0).contains(&f_mhz) {
        return Some(if pw_max >= 86.0 { 1107 } else { 1104 });
    }
    if (2798.0..=2822.0).contains(&f_mhz) {
        return Some(1116);
    }
    if (2858.0..=2892.0).contains(&f_mhz) {
        if lfm >= 0.8 {
            return Some(1109);
        }
        if (2874.0..=2876.5).contains(&f_mhz) && pri < 140.0 {
            return Some(1105);
        }
        return Some(1108);
    }
    None
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
    let pulse_mod: Vec<i64> = pulses
        .iter()
        .map(|p| {
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

    // 3. same-emitter split repair (carrier + BW overlap, PW within 2 %)
    let summ = |g: &Vec<usize>| {
        let mut f: Vec<f64> = g.iter().map(|&i| pulses[i].freq_hz).collect();
        let mut b: Vec<f64> = g.iter().map(|&i| pulses[i].bandwidth_hz).collect();
        let mut w: Vec<f64> = g.iter().map(|&i| pulses[i].duration_ns).collect();
        (quantile(&mut f, 0.02), quantile(&mut f, 0.98), quantile(&mut b, 0.02), quantile(&mut b, 0.98), quantile(&mut w, 0.5))
    };
    let s: Vec<_> = groups.par_iter().map(summ).collect();
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
    let mut uf = Uf::new(groups.len());
    {
        let pairs: Vec<(usize, usize)> =
            (0..groups.len()).flat_map(|a| (a + 1..groups.len()).map(move |b| (a, b))).collect();
        let hits: Vec<(usize, usize)> = pairs
            .par_iter()
            .filter(|(a, b)| {
                // Same emitter: coherent both ways. A one-way peak is a PRI
                // beat between two unrelated trains (e.g. 2116 µs vs 220 µs).
                let c1 = coherence(&toas[*a], &toas[*b], cfg.coh_top, cfg.coh_samples);
                let c2 = coherence(&toas[*b], &toas[*a], cfg.coh_top, cfg.coh_samples);
                c1.max(c2) >= cfg.coh_thr && c1.min(c2) >= cfg.coh_min
            })
            .copied()
            .collect();
        for (a, b) in hits {
            uf.union(a, b);
        }
    }
    groups = regroup(groups, &mut uf);
    groups.retain(|g| g.len() >= cfg.min_group);
    debug!("classic_v2: {} groups after coherence merge", groups.len());
    emit_dur("classic.coherence", t.elapsed());
    t = Instant::now();

    // 5. naming; one emitter per library id
    let mut known: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut out: Vec<(Option<i64>, Vec<usize>)> = Vec::new();
    let names: Vec<Option<i64>> = groups
        .par_iter()
        .map(|g| {
            let ps: Vec<&FlowPulse> = g.iter().map(|&i| &pulses[i]).collect();
            library_id(&ps, &sorted_toas(pulses, g))
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
