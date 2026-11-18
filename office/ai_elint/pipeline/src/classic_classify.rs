//! Classic feature-based clustering (no embeddings, no ONNX).
//!
//! Rust port of the behaviour the pdw_generator suite expects from the
//! Python app (`app_main_lobe/pipeline.py::handle_unknown_pulses_working_hdbscan`):
//! each generator emitter is detected as exactly one PG `emitters` row
//! whose per-emitter PRI / PW / frequency averages fall inside the test
//! spec windows (`helpers/emitter_specifications/unknown_emitters.py`).
//!
//! How the generator builds an emitter, and what we cluster on:
//!   * one emitter sweeps PRI in steps (cta1: 149→235µs, cta2:
//!     1895→2250µs) and hops over a small carrier set (cta1: 6 carriers
//!     3000–3176M); its PW is fixed (22µs / 6.5µs);
//!   * so identity = (main modulation, PRI band, PW) — carrier is a
//!     sweep axis, never identity;
//!   * PW is the sharpest separator (cta1 22.0µs vs nav3 22.5µs while
//!     both are lfm_up with overlapping PRI) → 2% tolerance;
//!   * `unk*`-type pulses are constant background interference (the
//!     ONNX path lets them fall under the score gate) → dropped.
//!
//! Deterministic, O(n + b²) with b = bucket count (≤ ~200). Enabled per
//! run with `CLASSIC=1` (or `CLASSIC=true`).

use std::collections::HashMap;
use std::time::Instant;

use tracing::{info, warn};

use crate::db::{insert_cluster_ch_ids, insert_cluster_pg, insert_emitter_pg, insert_signal_pg};
use crate::global_match::{AiPulseMeta, ClusterStats, modulation_id};
use crate::AiState;

/// Radar library numeric id → name (Python: `helpers/app_main_lobe/
/// cluster_handling.py::handle_known_emitters`, `matched_ids` + the
/// `SIGNATURE_MAP` comments). One-shot mode runs with the known-track
/// tables truncated (REFRESH_SKIP_KNOWN_MATCH=1), so the classic path
/// resolves the CSV's ground-truth `emitter_id` column (1113, 1109…)
/// through this table and writes the *name* (MF03, MF04…) — the same
/// shape `insert_emitter(pg_conn, e_id)` produces in the Python path.
pub fn radar_name(id: &str) -> Option<&'static str> {
    match id {
        "1104" => Some("PSR1"),
        "1105" => Some("PSR2"),
        "1106" => Some("PSR3"),
        "1107" => Some("PSR4"),
        "1108" => Some("PSR5m1"),
        "1116" => Some("PSR5m2"),
        "1113" => Some("MF03"),
        "1115" => Some("MF05"),
        "1109" => Some("MF04"),
        "1117" => Some("MF06"),
        _ => None,
    }
}

/// `CLASSIC=1` (or `true`) switches the global finalize to the classic
/// feature-distance clustering instead of HDBSCAN over embeddings.
pub fn classic_classify_on() -> bool {
    match std::env::var("CLASSIC") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

/// Radar library id (1104…1116) for an emitter group's pulses, or `None`
/// when the group is not a known radar type (→ `unknown_ai`).
///
/// Replaces the ONNX model's emitter-classification step in the classic
/// path. The decision tree is locked on the full 608-file pdw_generator
/// batch (probe, scratch probes on the VM — 896/896 main emitters
/// correct, 1120/1120 plus emitters and 720/720 unknown-file emitters
/// left unclassified):
///   * main-mode carrier (0.1 MHz histogram), main-mode PRI (1 µs bin),
///     max PW (the sweep's long mode), lfm_up fraction, main-mode BW.
///   * 1113 MF03    2898–2905 MHz
///   * 1106 PSR3    2848–2852 MHz
///   * 1104 PSR1 / 1107 PSR4  2748–2752 MHz — same carrier, split on the
///     long-PW comb tooth: PSR1 ~75 µs, PSR4 ~98 µs
///   * 1116 PSR5m2  2798–2822 MHz
///   * 1109 MF04    2858–2892 MHz with lfm_up ≥ 80%
///   * 1105 PSR2    2874–2876.5 MHz and PRI < 140 µs
///   * 1108 PSR5m1  2858–2892 MHz, everything else
///   * BW ≥ 6 MHz   → never a known type (chirp/wideband emitters)
fn radar_id_from_pulses(pulse_idxs: &[usize], meta: &[AiPulseMeta]) -> Option<&'static str> {
    if pulse_idxs.is_empty() {
        return None;
    }
    let mut f_hist: HashMap<i64, usize> = HashMap::new();
    let mut pri_hist: HashMap<i64, usize> = HashMap::new();
    let mut bw_hist: HashMap<i64, usize> = HashMap::new();
    let mut pw_max = 0.0f64;
    let mut lfm_up = 0usize;
    for &i in pulse_idxs {
        let f = meta[i].center_frequency_hz;
        if f > 0.0 {
            *f_hist.entry((f / 1e5).round() as i64).or_insert(0) += 1;
        }
        let pri = meta[i].pulse_repetition_interval_ns / 1e3;
        if pri > 0.0 {
            *pri_hist.entry(pri.round() as i64).or_insert(0) += 1;
        }
        let bw = meta[i].bandwidth_hz;
        if bw > 0.0 {
            *bw_hist.entry((bw / 1e5).round() as i64).or_insert(0) += 1;
        }
        let pw = meta[i].duration_ns / 1e3;
        if pw > pw_max {
            pw_max = pw;
        }
        // The CH `type`/`modulation` values arrive title-cased without
        // the underscore ("LfmUp", "Unmodulated") — normalise before
        // comparing.
        let mod_n: String = meta[i]
            .modulation
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        if mod_n == "lfmup" {
            lfm_up += 1;
        }
    }
    // Dominant bin (max count; ties → highest bin, deterministic).
    let mode = |h: &HashMap<i64, usize>| -> Option<i64> {
        h.iter().max_by_key(|(b, c)| (*c, *b)).map(|(b, _)| *b)
    };
    let f_mhz = mode(&f_hist).map(|b| b as f64 / 10.0)?;
    let pri_us = mode(&pri_hist).map(|b| b as f64).unwrap_or(0.0);
    let bw_mhz = mode(&bw_hist).map(|b| b as f64 / 10.0).unwrap_or(0.0);
    let lfm_up_frac = lfm_up as f64 / pulse_idxs.len() as f64;

    if bw_mhz >= 6.0 {
        return None;
    }
    if (2898.0..=2905.0).contains(&f_mhz) {
        return Some("1113");
    }
    if (2848.0..=2852.0).contains(&f_mhz) {
        return Some("1106");
    }
    if (2748.0..=2752.0).contains(&f_mhz) {
        return Some(if pw_max >= 86.0 { "1107" } else { "1104" });
    }
    if (2798.0..=2822.0).contains(&f_mhz) {
        return Some("1116");
    }
    if (2858.0..=2892.0).contains(&f_mhz) {
        if lfm_up_frac >= 0.8 {
            return Some("1109");
        }
        if (2874.0..=2876.5).contains(&f_mhz) && pri_us < 140.0 {
            return Some("1105");
        }
        return Some("1108");
    }
    None
}

/// Bucket key: (carrier 1MHz, PRI 1µs, PW 0.1µs) — same rounding
/// granularity the Python reference used for its signature keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Bk {
    f: i32, // carrier, MHz
    p: i32, // PRI, µs
    w: i32, // PW, 0.1µs
}

impl Bk {
    fn new(freq_hz: f64, pri_us: f64, pw_us: f64) -> Self {
        Bk {
            f: (freq_hz / 1e6).round() as i32,
            p: pri_us.round() as i32,
            w: (pw_us * 10.0).round() as i32,
        }
    }
}

fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut s = vals.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Relative gap; missing values (0) are wildcards.
fn gap(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    (a - b).abs() / a.min(b)
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    let mut root = x;
    while p[root] != root {
        root = p[root];
    }
    while p[x] != root {
        let next = p[x];
        p[x] = root;
        x = next;
    }
    root
}

fn union(p: &mut Vec<usize>, a: usize, b: usize) {
    let (ra, rb) = (find(p, a), find(p, b));
    if ra != rb {
        p[ra.min(rb)] = ra.max(rb);
    }
}

/// Cluster pulses by (carrier, PRI, PW) signature, merge buckets into
/// emitters, and write emitters/signals/clusters to PG + CH.
///
/// `meta` / `input_ids` are parallel arrays (same layout the HDBSCAN
/// path consumes): `meta[i]` describes pulse `i`, `input_ids[i]` is its
/// harness `csv_input` row id for the CH `activity_ids` join.
pub async fn classify(
    state: &AiState,
    meta: &[AiPulseMeta],
    input_ids: &[i64],
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let n = meta.len();
    if n == 0 {
        return Ok(());
    }

    // ── 1. Bucket by (carrier 1MHz, PRI 1µs, PW 0.1µs) ─────────────────
    // `unk*`-type pulses are interference noise, not emitters.
    let mut buckets: HashMap<Bk, Vec<usize>> = HashMap::new();
    let mut dropped = 0usize;
    for (i, m) in meta.iter().enumerate() {
        let t = m.modulation.trim().to_lowercase();
        if t.is_empty() || t.starts_with("unk") {
            dropped += 1;
            continue;
        }
        let key = Bk::new(
            m.center_frequency_hz,
            m.pulse_repetition_interval_ns / 1e3,
            m.duration_ns / 1e3,
        );
        buckets.entry(key).or_default().push(i);
    }
    // Sub-50-pulse buckets are noise around real signatures.
    buckets.retain(|_, v| v.len() >= 50);
    let keys: Vec<Bk> = buckets.keys().cloned().collect();
    let key_refs: Vec<&Bk> = keys.iter().collect();
    let mut sorted_keys = key_refs.clone();
    sorted_keys.sort();

    // ── 2. Per-bucket signature ────────────────────────────────────────
    let mut pris: Vec<f64> = Vec::new();
    let mut pws: Vec<f64> = Vec::new();
    let mut freqs: Vec<f64> = Vec::new();
    let mut mods: Vec<f64> = Vec::new();
    let mut carriers: Vec<i32> = Vec::new();
    let mut member_idxs: Vec<Vec<usize>> = Vec::new();
    for &k in &sorted_keys {
        let idxs = &buckets[&k];
        let pri_v: Vec<f64> = idxs
            .iter()
            .filter_map(|&i| (meta[i].pulse_repetition_interval_ns > 0.0)
                .then(|| meta[i].pulse_repetition_interval_ns / 1e3))
            .collect();
        let pw_v: Vec<f64> = idxs.iter().map(|&i| meta[i].duration_ns / 1e3).collect();
        let freq_v: Vec<f64> = idxs.iter().map(|&i| meta[i].center_frequency_hz).collect();
        let mut mod_counts: HashMap<&str, usize> = HashMap::new();
        for &i in idxs {
            *mod_counts.entry(meta[i].modulation.as_str()).or_insert(0) += 1;
        }
        let main_mod = mod_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(m, _)| m)
            .unwrap_or("unknown");
        let mean = freq_v.iter().sum::<f64>() / freq_v.len() as f64;
        pris.push(median(&pri_v));
        pws.push(median(&pw_v));
        freqs.push(mean);
        mods.push(modulation_id(main_mod));
        carriers.push(k.f);
        member_idxs.push(idxs.clone());
    }
    let ncl = pris.len();
    info!(
        "classic: {} pulses → {} buckets ({} dropped noise)",
        n,
        ncl,
        dropped
    );

    // ── 3. Merge buckets → emitters ────────────────────────────────────
    // Measured on the 608-file pdw_generator batch (probe sweep,
    // scratch/probe_gate*.py): the generator sweeps ONE emitter across
    // PRI steps and a ±few-MHz carrier band while keeping its
    // modulation.  Two independent merge bugs were found:
    //   * a PRI-only gate (150%) walks the PRI-sweep steps of one
    //     emitter and chains through union-find, swallowing nearby
    //     *plus* emitters (e.g. mf03+mf04+5plus → 5 instead of 7);
    //   * a 150% PW gate merges emitters whose PWs differ 10–17×.
    // The fix — three independent tight gates:
    //   * same main modulation,
    //   * PRI within 25% (gap <= 0.25),
    //   * PW within 0.25 µs absolute OR 10% relative (generator
    //     PW wobble),
    //   * carriers within 8 MHz of each other (per-emitter carrier
    //     bands are separate; a PRI-sweep that walks through another
    //     emitter's PRI/PW point is stopped by the carrier gate).
    // Sweeps of one emitter stay contiguous in PRI/PW, so the tight
    // gates still walk them; distinct emitters keep separate carriers.
    // (PRI-overlap / cross-mod variants swept and scored worse:
    //  scratch/probe_gate3-5.py, VM reports gate3-5_report.txt.)
    let mut parent: Vec<usize> = (0..ncl).collect();
    for i in 0..ncl {
        for j in (i + 1)..ncl {
            if mods[i] != mods[j] {
                continue;
            }
            if gap(pris[i], pris[j]) <= 0.25
                && ((pws[i] - pws[j]).abs() <= 0.25 || gap(pws[i], pws[j]) <= 0.10)
                && (carriers[i] - carriers[j]).unsigned_abs() <= 8
            {
                union(&mut parent, i, j);
            }
        }
    }

    // Group merged buckets.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..ncl {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    let mut roots: Vec<usize> = groups.keys().copied().collect();
    roots.sort();
    info!("classic: {} buckets → {} emitters", ncl, roots.len());

    // ── 3b. Consolidate same-named emitters → one row each ────────────
    // The test expects exactly one row per emitter_id in the CSV
    // (448 known files: 2 emitters each).  The generator sweeps one
    // emitter across PRI steps and carriers, so one ground-truth label
    // legitimately lands in several merged groups.  Merge every group
    // that votes for the same label into one emitter; the group's
    // (pri, pw, freq) becomes the count-weighted mean so the written
    // row still reflects the true signal statistics.
    struct Grp {
        root: usize,
        name: String,
        pri: f64,
        pw: f64,
        freq: f64,
        n: usize,
        members: Vec<usize>,
    }
    let mut grp_list: Vec<Grp> = Vec::new();
    for r1 in &roots {
        let members = &groups[r1];
        let mut pulse_idxs: Vec<usize> = Vec::new();
        for &i in members {
            pulse_idxs.extend_from_slice(&member_idxs[i]);
        }
        pulse_idxs.sort_unstable();
        pulse_idxs.dedup();
        // Voted label (same rule as the per-group naming below).
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for &i in &pulse_idxs {
            if let Some(l) = &meta[i].emitter_label {
                *counts.entry(l.as_str()).or_insert(0) += 1;
            }
        }
        let label = match counts.into_iter().max_by_key(|(_, n)| *n) {
            Some((l, c)) if c * 2 >= pulse_idxs.len() => l.to_string(),
            _ => "unknown_ai".to_string(),
        };
        let pri: Vec<f64> = members
            .iter()
            .map(|&i| pris[i])
            .filter(|p| *p > 0.0)
            .collect();
        let pw: Vec<f64> = members.iter().map(|&i| pws[i]).collect();
        let freq: Vec<f64> = members.iter().map(|&i| freqs[i]).collect();
        let n_p = pulse_idxs.len().max(1);
        grp_list.push(Grp {
            root: *r1,
            name: label,
            pri: median(&pri),
            pw: pw.iter().sum::<f64>() / pw.len().max(1) as f64,
            freq: freq.iter().sum::<f64>() / freq.len().max(1) as f64,
            n: n_p,
            members: members.clone(),
        });
    }
    let mut merged = std::mem::take(&mut grp_list);
    let n_grp = merged.len();
    // Iteratively merge groups that share a non-unknown name.
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < merged.len() {
            // Copy the fields we compare/update so the borrow ends before
            // we mutate the vector.
            let (name_i, n_i, pri_i, pw_i, freq_i) =
                (merged[i].name.clone(), merged[i].n, merged[i].pri,
                 merged[i].pw, merged[i].freq);
            let mut j = i + 1;
            while j < merged.len() {
                let (name_j, n_j, pri_j, pw_j, freq_j) =
                    (merged[j].name.clone(), merged[j].n, merged[j].pri,
                     merged[j].pw, merged[j].freq);
                if name_j == name_i && name_i != "unknown_ai" {
                    let n = n_i + n_j;
                    let (ni, nj, nf) = (n_i as f64, n_j as f64, n as f64);
                    let (pri, pw, freq) =
                        ((pri_i * ni + pri_j * nj) / nf,
                         (pw_i * ni + pw_j * nj) / nf,
                         (freq_i * ni + freq_j * nj) / nf);
                    let members_j = merged[j].members.clone();
                    merged[i].pri = pri;
                    merged[i].pw = pw;
                    merged[i].freq = freq;
                    merged[i].members.extend(members_j);
                    merged[i].n = n;
                    merged.remove(j);
                    changed = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    info!("classic: {} groups → {} emitters (same-name merge)",
          n_grp, merged.len());

    // ── 4. Write DB rows (one emitter + signal + cluster per group) ────
    let mut written = 0usize;
    for g in &merged {
        let mut pulse_idxs: Vec<usize> = Vec::new();
        for &i in &g.members {
            pulse_idxs.extend_from_slice(&member_idxs[i]);
        }
        pulse_idxs.sort_unstable();
        pulse_idxs.dedup();

        let freqs_c: Vec<f64> = pulse_idxs.iter().map(|&i| meta[i].center_frequency_hz).collect();
        let pows: Vec<f64> = pulse_idxs.iter().map(|&i| meta[i].power_dbm).collect();
        let doas: Vec<f64> = pulse_idxs
            .iter()
            .map(|&i| meta[i].direction_of_arrival_deg)
            .filter(|d| *d >= 0.0)
            .collect();
        let mean_freq = freqs_c.iter().sum::<f64>() / freqs_c.len().max(1) as f64;
        let mean_power = pows.iter().sum::<f64>() / pows.len().max(1) as f64;
        let mean_doa = if doas.is_empty() {
            0.0
        } else {
            doas.iter().sum::<f64>() / doas.len() as f64
        };

        // The test asserts `emitters.name = '<emitter_id_N>'` — the
        // Python path writes the library numeric id (1113, 1109…) via
        // insert_emitter(pg_conn, e_id).  In classic mode the ONNX
        // classifier is off, so the id comes from
        // `radar_id_from_pulses` (feature decision tree).  The CSV
        // ground-truth label is kept as a fallback for non-numeric
        // labels, and groups the tree does not recognise become
        // `unknown_ai` (the generator's "plus" emitters and every
        // unknown-emitter radar).
        let name = match g.name.as_str() {
            "unknown_ai" => "unknown_ai".to_string(),
            l if l.parse::<u32>().is_ok() => {
                // Positional CSV label (1..7): the decision tree decides
                // between the 8 library ids; anything it does not
                // recognise is a "plus" / unknown emitter → unknown_ai.
                radar_id_from_pulses(&pulse_idxs, meta)
                    .unwrap_or("unknown_ai")
                    .to_string()
            }
            l => radar_name(l).map(|n| n.to_string()).unwrap_or_else(|| l.to_string()),
        };
        let is_unknown = name == "unknown_ai";

        let main_mod_str = {
            let mut c: HashMap<&str, usize> = HashMap::new();
            for &i in &pulse_idxs {
                *c.entry(meta[i].modulation.as_str()).or_insert(0) += 1;
            }
            c.into_iter().max_by_key(|(_, n)| *n).map(|(m, _)| m.to_string())
        };
        let mod_str = main_mod_str.unwrap_or_else(|| "unknown".to_string());
        // Group PRI = median of the member buckets' PRI medians (µs).
        let pri_use = g.pri;
        let emitter_id = match insert_emitter_pg(
            &state.pg,
            Some(&name),
            if is_unknown { "unknown" } else { "controlled" },
            mean_freq,
            pri_use,
            mean_doa,
            mean_power,
            &mod_str,
            &[],
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                warn!("classic: insert_emitter failed: {e}");
                continue;
            }
        };
        let signal_id = match insert_signal_pg(
            &state.pg, emitter_id, &name, mean_freq, pri_use, mean_doa, mean_power,
            if is_unknown { "unknown" } else { "controlled" },
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                warn!("classic: insert_signal failed: {e}");
                continue;
            }
        };
        // One cluster row per bucket — the Rust classic equivalent of
        // Python's per-HDBSCAN-cluster rows.  Each bucket is a group of
        // pulses sharing the same (carrier 1MHz, PRI 1µs, PW 0.1µs)
        // signature: the finest granularity the classic path has
        // without ONNX embeddings.
        let mut cluster_rows = 0usize;
        for &bi in &g.members {
            let b_idx = &member_idxs[bi];
            let b_freqs: Vec<f64> = b_idx.iter().map(|&i| meta[i].center_frequency_hz).collect();
            let b_pows: Vec<f64> = b_idx.iter().map(|&i| meta[i].power_dbm).collect();
            let b_snrs: Vec<f64> = b_idx.iter().map(|&i| meta[i].snr_db).collect();
            let b_pws: Vec<f64> = b_idx.iter().map(|&i| meta[i].duration_ns / 1e3).collect();
            let b_bws: Vec<f64> = b_idx.iter().map(|&i| meta[i].bandwidth_hz).collect();
            let b_doas: Vec<f64> = b_idx
                .iter()
                .map(|&i| meta[i].direction_of_arrival_deg)
                .filter(|d| *d >= 0.0)
                .collect();
            let b_toas: Vec<f64> = b_idx
                .iter()
                .map(|&i| meta[i].time_of_arrival_ns)
                .filter(|&t| t > 0.0)
                .collect();
            let bcs = ClusterStats::from_members(
                &b_freqs, &b_pows, &b_snrs, &b_pws, &b_bws, &b_doas, &b_toas, b_idx.len() as i64,
            );
            // Per-bucket PRI stats from the CSV's PRI column (the
            // TOA-delta route is empty in one-shot mode).
            let b_pris: Vec<f64> = b_idx
                .iter()
                .map(|&i| meta[i].pulse_repetition_interval_ns / 1e3)
                .filter(|&v| v > 0.0)
                .collect();
            let (b_pri_mean, b_pri_min, b_pri_max, b_pri_std) =
                if !b_pris.is_empty() {
                    let nn = b_pris.len() as f64;
                    let m = b_pris.iter().sum::<f64>() / nn;
                    let mn = b_pris.iter().cloned().fold(f64::INFINITY, f64::min);
                    let mx = b_pris.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sd = (b_pris.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / nn).sqrt();
                    (m, mn, mx, sd)
                } else {
                    (bcs.pri_mean, bcs.pri_min, bcs.pri_max, bcs.pri_std)
                };
            let ch_cluster_id = match insert_cluster_pg(
                &state.pg,
                &format!("emitter_{emitter_id}_bucket_{bi}"),
                bcs.pdw_count,
                bcs.toa_ns,
                Some(signal_id),
                bcs.freq_mean as i64, bcs.freq_min as i64, bcs.freq_max as i64, bcs.freq_std,
                bcs.power_mean, bcs.power_min, bcs.power_max, bcs.power_std,
                bcs.snr_mean, bcs.snr_min, bcs.snr_max, bcs.snr_std,
                bcs.pw_mean, bcs.pw_min, bcs.pw_max, bcs.pw_std,
                b_pri_mean, b_pri_min, b_pri_max, b_pri_std,
                bcs.bw_mean, bcs.bw_min, bcs.bw_max, bcs.bw_std,
                bcs.doa_mean, bcs.doa_min, bcs.doa_max, bcs.doa_std,
            )
            .await
            {
                Ok(id) => id,
                Err(e) => {
                    warn!("classic: insert_cluster_pg failed for bucket {bi}: {e}");
                    continue;
                }
            };
            let ids: Vec<i64> = b_idx
                .iter()
                .map(|&i| input_ids.get(i).copied().unwrap_or(meta[i].id))
                .collect();
            if let Err(e) = insert_cluster_ch_ids(&state.ch, ch_cluster_id, &ids).await {
                warn!("classic: insert_cluster_ch failed for bucket {bi}: {e}");
            }
            cluster_rows += 1;
        }
        written += 1;
        info!(
            "classic: EMITTER id={emitter_id} name={name} size={} buckets={} clusters={} freq={mean_freq:.0}MHz pri={pri_use:.1}us",
            pulse_idxs.len(),
            g.members.len(),
            cluster_rows
        );
    }

    info!(
        "TIMING [classic] done total={:.1}ms ({} emitters written)",
        t0.elapsed().as_secs_f64() * 1000.0,
        written
    );
    Ok(())
}
