//! ONNX engine — window-by-window port of the Python reference
//! (`ai_elint_python_master/integration/app`: app.py → inference.py →
//! cluster_handling.handle_embedding → custom_clustering.simple_clustering).
//!
//! Flow per 512-pulse window (CSV order, tail < 512 dropped like Python):
//!
//!   1. features: `utils/create_features.generate_features_per_window`
//!      (freq, power, 200 PW RBFs, 256 TOA RBFs over a 41-pulse window).
//!   2. model: the window is ONE sequence (1, 512, 458) sent to the external
//!      model service (`ONNX_URL`, see `onnx_model_service/`). The service
//!      returns per pulse: 1024-d embedding, centre id, similarity.
//!   3. known emitters (`handle_known_emitters_vectorized`): centre-id remap,
//!      per-emitter medoid refinement, thresholds, signature whitelist.
//!      Known pulses → one emitter per library id (name = the id), one
//!      signal, one cluster row per window.
//!   4. unknown pulses (`handle_unknown_pulses_buffered`): buffered until
//!      512 pulses, then greedy cosine clustering → merge by medoid +
//!      angle → score gate → match against the unknown track cache → new
//!      or existing `unknown_ai` emitter/signal + cluster rows.
//!
//! Windows are independent for steps 1–2, so they run ahead in parallel on
//! worker threads (the model service spreads them over GPU + CPU). Steps
//! 3–4 carry state from window to window, so they run in window order.
//!
//! Nothing here reads the CSV's ground-truth `emitter_id` column.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use tracing::{info, warn};

use crate::AiState;

pub const WINDOW: usize = 512;
pub const IN_DIM: usize = 458;
pub const OUT_DIM: usize = 1024;
const NUM_PW: usize = 200;
const NUM_TOA: usize = 256;
const FEAT_WIN: usize = 40; // Python WINDOW_SIZE; radius 20 → 41 rows

/// One pulse as the ONNX flow sees it (CSV row order).
#[derive(Clone, Debug)]
pub struct FlowPulse {
    /// `signals.csv_input.id` (1-based row number) — the CH join key.
    pub input_id: i64,
    pub freq_hz: f64,
    pub power_dbm: f64,
    pub bandwidth_hz: f64,
    pub duration_ns: f64,
    pub doa_deg: f64,
    pub snr_db: f64,
    pub toa_ns: f64,
    pub modulation: String,
}

/// Tunables (Python config.py values; env-overridable for experiments).
#[derive(Clone, Debug)]
pub struct FlowCfg {
    pub known_dot: f32,           // KNOWN_DOT_PRODUCT_THRESHOLD
    pub sim_thr: f32,             // SIMILARITY_THRESHOLD
    pub sim_thr_1117: f32,        // SIMILARITY_THRESHOLD_1117
    pub known_min_signal: usize,  // KNOWN_MIN_SIGNAL_SIZE
    pub unknown_min_signal: usize, // UNKNOWN_MIN_SIGNAL_SIZE
    pub min_cluster: usize,       // MIN_CLUSTER_SIZE (custom_clustering)
    pub cluster_thr: f32,         // simple_clustering threshold
    pub merge_thr: f64,           // merge_clusters sim_threshold
    pub merged_min_score: f64,    // UNKNOWN_MERGED_CLUSTER_MIN_SCORE
    pub track_thr: f64,           // custom_clustering UNKNOWN_DOT_PRODUCT_THRESHOLD
    pub accum_ns: f64,            // UNKNOWN_ACCUMULATION_TIME_SEC
    pub buffer_threshold: usize,  // BUFFER_THRESHOLD
    pub medoid_min_rows: usize,   // medoid refresh needs > this many rows
    /// Cluster PRI ignores intervals below this (µs): the same pulse seen by
    /// two receivers. 0 = Python's exact formula.
    pub pri_min_gap_us: f64,
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl FlowCfg {
    pub fn from_env() -> Self {
        Self {
            known_dot: env_or("ONNX_KNOWN_DOT", 0.6),
            sim_thr: env_or("ONNX_SIM_THR", 0.7),
            sim_thr_1117: env_or("ONNX_SIM_THR_1117", 0.5),
            known_min_signal: env_or("ONNX_KNOWN_MIN_SIGNAL", 20),
            unknown_min_signal: env_or("ONNX_UNKNOWN_MIN_SIGNAL", 20),
            min_cluster: env_or("ONNX_MIN_CLUSTER", 20),
            cluster_thr: env_or("ONNX_CLUSTER_THR", 0.8),
            merge_thr: env_or("ONNX_MERGE_THR", 0.7),
            merged_min_score: env_or("ONNX_MERGED_MIN_SCORE", 0.3),
            track_thr: env_or("ONNX_TRACK_THR", 0.5),
            accum_ns: env_or::<f64>("ONNX_ACCUM_SEC", 200.0) * 1e9,
            buffer_threshold: env_or("ONNX_BUFFER_THRESHOLD", 512),
            medoid_min_rows: env_or("ONNX_MEDOID_MIN_ROWS", 30),
            pri_min_gap_us: env_or("ONNX_PRI_MIN_GAP_US", 1.0),
        }
    }
}

/// Library ids the known path accepts (`matched_ids` in Python).
const MATCHED_IDS: [i64; 10] = [1104, 1105, 1106, 1107, 1108, 1116, 1113, 1115, 1109, 1117];

/// emitters.signature_matched_id per library id (`SIGNATURE_MAP`).
pub(crate) fn signature_id(id: i64) -> Option<i32> {
    Some(match id {
        1104 => 22,
        1105 => 23,
        1106 => 24,
        1107 => 25,
        1108 => 26,
        1116 => 28,
        1113 => 32,
        1115 => 33,
        1109 => 29,
        1117 => 38,
        _ => return None,
    })
}

fn modulation_type_id(m: &str) -> i32 {
    match m.trim().to_ascii_uppercase().as_str() {
        "UNKNOWN" => 1,
        "UNMODULATED" => 2,
        "LFM_UP" => 3,
        "NLFM_UP" => 4,
        "LFM_DOWN" => 5,
        "NLFM_DOWN" => 6,
        "FMCW" => 7,
        "BARKER_2A" => 8,
        "BARKER_2B" => 9,
        "BARKER_3" => 10,
        "BARKER_4A" => 11,
        "BARKER_4B" => 12,
        "BARKER_5" => 13,
        "BARKER_7" => 14,
        "BARKER_11" => 15,
        "BARKER_13" => 16,
        "AM" => 17,
        s => s.parse().unwrap_or(1),
    }
}

// ───────────────────────── features ─────────────────────────

struct FeatConsts {
    pw_c: Vec<f64>,
    pw_den: Vec<f64>,
    toa_c: Vec<f64>,
    toa_den: Vec<f64>,
}

fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let (a, b) = (lo.log10(), hi.log10());
    (0..n).map(|i| 10f64.powf(a + (b - a) * i as f64 / (n - 1) as f64)).collect()
}

/// numpy: diffs of consecutive centres; first gap repeated at the front,
/// last gap repeated at the end; sigma = min(prev, next) * 2/3.
fn denom(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let d: Vec<f64> = (0..n - 1).map(|i| c[i + 1] - c[i]).collect();
    (0..n)
        .map(|i| {
            let next = if i < n - 1 { d[i] } else { d[n - 2] };
            let prev = if i == 0 { d[0] } else { d[i - 1] };
            let s = next.min(prev) * (2.0 / 3.0);
            2.0 * s * s
        })
        .collect()
}

fn feat_consts() -> &'static FeatConsts {
    static C: std::sync::OnceLock<FeatConsts> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let pw_c = logspace(0.1, 300.0, NUM_PW);
        let toa_c = logspace(10.0, 5000.0, NUM_TOA);
        FeatConsts { pw_den: denom(&pw_c), toa_den: denom(&toa_c), pw_c, toa_c }
    })
}

/// Exponent below which a kernel term is skipped (exp(-30) ≈ 9e-14): far
/// below f32 resolution of any feature value, so the f32 output matches
/// the full 256-kernel sum. `FULL_TOA_KERNELS=1` evaluates every kernel.
const SKIP_EXPONENT: f64 = 30.0;

/// Add the 256 TOA RBF kernels for one lag. Centres are log-spaced and each
/// kernel's width follows its spacing, so only kernels near `d0` matter:
/// start at the nearest centre and walk outwards until terms vanish.
#[inline]
fn toa_kernels(d0: f64, k: &FeatConsts, acc: &mut [f64; NUM_TOA]) {
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *FULL.get_or_init(|| std::env::var("FULL_TOA_KERNELS").map(|v| v == "1").unwrap_or(false)) {
        for c in 0..NUM_TOA {
            let d = d0 - k.toa_c[c];
            acc[c] += (-(d * d) / k.toa_den[c]).exp();
        }
        return;
    }
    let start = k.toa_c.partition_point(|c| *c < d0).min(NUM_TOA - 1);
    let mut c = start as isize;
    while c >= 0 {
        let cu = c as usize;
        let d = d0 - k.toa_c[cu];
        let e = d * d / k.toa_den[cu];
        if e > SKIP_EXPONENT && cu < start {
            break;
        }
        acc[cu] += (-e).exp();
        c -= 1;
    }
    for cu in start + 1..NUM_TOA {
        let d = d0 - k.toa_c[cu];
        let e = d * d / k.toa_den[cu];
        if e > SKIP_EXPONENT {
            break;
        }
        acc[cu] += (-e).exp();
    }
}

/// Features for exactly one 512-pulse window → 512 * 458 f32, row-major.
pub fn window_features(p: &[FlowPulse]) -> Vec<f32> {
    assert_eq!(p.len(), WINDOW);
    let k = feat_consts();
    let r = FEAT_WIN / 2;
    let toa: Vec<f64> = p.iter().map(|x| x.toa_ns / 1000.0).collect();
    let mut out = vec![0f32; WINDOW * IN_DIM];
    for i in 0..WINDOW {
        let row = &mut out[i * IN_DIM..(i + 1) * IN_DIM];
        // Python casts freq/power/pw to float32 first (app.py .astype(np.float32)).
        let f32_freq = (p[i].freq_hz as f32) / 1e6f32;
        row[0] = (f32_freq - 50.0) / 10950.0;
        row[1] = (p[i].power_dbm as f32 + 100.0) / 100.0;
        let pw = ((p[i].duration_ns / 1000.0) as f32 as f64).clamp(0.1, 300.0);
        for c in 0..NUM_PW {
            let d = pw - k.pw_c[c];
            row[2 + c] = (-(d * d) / k.pw_den[c]).exp() as f32;
        }
        let centre = i.clamp(r, WINDOW - 1 - r);
        let mut acc = [0f64; NUM_TOA];
        for j in centre - r..=centre + r {
            let d0 = (toa[i] - toa[j]).abs();
            toa_kernels(d0, k, &mut acc);
        }
        for c in 0..NUM_TOA {
            row[2 + NUM_PW + c] = acc[c] as f32;
        }
    }
    out
}

// ───────────────────────── model service client ─────────────────────────

pub struct ModelOut {
    pub emb: Vec<f32>, // WINDOW * OUT_DIM
    pub ids: Vec<i64>,
    pub sims: Vec<f32>,
}

fn split_url(url: &str) -> Result<(String, String)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("ONNX_URL must start with http:// ({url})"))?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let base = path.trim_end_matches('/');
    let base = base.strip_suffix("/predict").or_else(|| base.strip_suffix("/embed")).unwrap_or(base);
    Ok((hostport.to_string(), format!("{base}/predict")))
}

/// `predict` with retries (connection resets / service restarts): 5 tries,
/// 0.2 s → 3.2 s backoff.
pub fn predict(url: &str, feats: &[f32], n: usize) -> Result<ModelOut> {
    let mut wait = Duration::from_millis(200);
    let mut last = None;
    for attempt in 0..5 {
        match predict_once(url, feats, n) {
            Ok(o) => return Ok(o),
            Err(e) => {
                warn!("model service call failed (attempt {}): {e:#}", attempt + 1);
                last = Some(e);
                std::thread::sleep(wait);
                wait *= 2;
            }
        }
    }
    Err(last.unwrap())
}

/// POST one or more whole windows to `/predict`; parse the framed reply.
fn predict_once(url: &str, feats: &[f32], n: usize) -> Result<ModelOut> {
    let (hostport, path) = split_url(url)?;
    let mut s = TcpStream::connect(&hostport).with_context(|| format!("model service {hostport}"))?;
    s.set_read_timeout(Some(Duration::from_secs(600)))?;
    s.set_nodelay(true)?;
    let body: Vec<u8> = feats.iter().flat_map(|v| v.to_le_bytes()).collect();
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/octet-stream\r\n\
         X-Window-Size: {WINDOW}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(&body)?;
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line)?;
    if !line.split_whitespace().nth(1).map_or(false, |c| c.starts_with('2')) {
        bail!("model service: {}", line.trim());
    }
    let mut clen: Option<usize> = None;
    loop {
        line.clear();
        r.read_line(&mut line)?;
        let l = line.trim();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                clen = v.trim().parse().ok();
            }
        }
    }
    let mut buf = Vec::new();
    match clen {
        Some(c) => {
            buf.resize(c, 0);
            r.read_exact(&mut buf)?;
        }
        None => {
            r.read_to_end(&mut buf)?;
        }
    }
    if buf.len() < 24 {
        bail!("model service: short /predict reply ({} bytes)", buf.len());
    }
    let ne = u64::from_le_bytes(buf[0..8].try_into()?) as usize;
    let nl = u64::from_le_bytes(buf[8..16].try_into()?) as usize;
    let ns = u64::from_le_bytes(buf[16..24].try_into()?) as usize;
    if ne != n * OUT_DIM * 4 || nl != n * 8 || ns != n * 4 || buf.len() != 24 + ne + nl + ns {
        bail!("model service: bad /predict sections ({ne},{nl},{ns}) for n={n}");
    }
    let b = &buf[24..];
    let emb = b[..ne].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let ids = b[ne..ne + nl].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    let sims = b[ne + nl..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(ModelOut { emb, ids, sims })
}

// ───────────────────────── vector helpers ─────────────────────────

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..16 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Rows of an embedding set (borrowed views into window buffers).
type Rows<'a> = Vec<&'a [f32]>;

/// argmax_i sum_j <x_i, x_j>  ==  argmax_i <x_i, sum_j x_j>  (O(n·d)).
fn medoid_idx(rows: &[&[f32]]) -> usize {
    let mut s = vec![0f32; OUT_DIM];
    for r in rows {
        for (a, b) in s.iter_mut().zip(r.iter()) {
            *a += *b;
        }
    }
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, r) in rows.iter().enumerate() {
        let v = dot(r, &s);
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

fn mean_vec(rows: &[&[f32]]) -> Vec<f32> {
    let mut s = vec![0f64; OUT_DIM];
    for r in rows {
        for (a, b) in s.iter_mut().zip(r.iter()) {
            *a += *b as f64;
        }
    }
    s.iter().map(|v| (*v / rows.len() as f64) as f32).collect()
}

fn std0(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// numpy.percentile (linear interpolation) on a sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

// ───────────────────────── angle / scoring ─────────────────────────

/// `get_cluster_angle`: power-weighted mean bearing around the peak pulse.
pub(crate) fn cluster_angle(doas: &[f64], powers: &[f64]) -> f64 {
    let mw: Vec<f64> = powers.iter().map(|p| 10f64.powf(p / 10.0)).collect();
    let mut peak = 0usize;
    for i in 1..mw.len() {
        if mw[i] > mw[peak] {
            peak = i;
        }
    }
    let (s, e) = (peak.saturating_sub(2), (peak + 3).min(mw.len()));
    let (mut x, mut y, mut w) = (0.0, 0.0, 0.0);
    for i in s..e {
        let a = doas[i].to_radians();
        x += a.cos() * mw[i];
        y += a.sin() * mw[i];
        w += mw[i];
    }
    (y / w).atan2(x / w).to_degrees().rem_euclid(360.0)
}

/// `get_cluster_angle_score` → (major angle, fraction within ±40°).
pub(crate) fn angle_score(doas: &[f64], powers: &[f64]) -> (f64, f64) {
    if doas.is_empty() {
        return (0.0, 0.0);
    }
    let major = cluster_angle(doas, powers);
    let inside = doas
        .iter()
        .filter(|a| {
            let mut d = (*a - major).abs();
            if d > 180.0 {
                d = 360.0 - d;
            }
            d <= 40.0
        })
        .count();
    (major, inside as f64 / doas.len() as f64)
}

/// `merge_clusters.get_angle_score`: 1 inside 20°, cos² fall-off to 40°.
fn merge_angle_score(d: f64) -> f64 {
    if d <= 20.0 {
        1.0
    } else if d <= 40.0 {
        (((d - 20.0) / 20.0) * std::f64::consts::FRAC_PI_2).cos().powi(2)
    } else {
        0.0
    }
}

fn wrap_diff(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

// ───────────────────────── cluster metrics + DB ─────────────────────────

/// One pulse inside a group being written (metadata + order preserved).
#[derive(Clone)]
pub(crate) struct Member<'a> {
    pub p: &'a FlowPulse,
    pub emb: &'a [f32],
}

/// `calculate_iqr_pri_stats_per_group` on TOAs in member order.
/// `min_gap_us` > 0 drops near-zero intervals (the same pulse reported by
/// two receivers); 0 keeps Python's exact behaviour.
/// Share of the tallest interval bin a shorter bin needs to count as the
/// fundamental PRI (`fundamental` estimator).
const FUNDAMENTAL_SHARE: f64 = 0.25;

/// `min_gap_us` > 0 drops near-zero intervals; `fundamental` picks the
/// shortest well-populated interval bin instead of the tallest (Python).
fn pri_stats(toas_ns: &[f64], min_gap_us: f64) -> (f64, f64, f64, f64) {
    pri_stats_ex(toas_ns, min_gap_us, false)
}

fn pri_stats_ex(toas_ns: &[f64], min_gap_us: f64, fundamental: bool) -> (f64, f64, f64, f64) {
    let mut v: Vec<f64> = toas_ns
        .windows(2)
        .map(|w| (w[1] - w[0]) / 1000.0)
        .filter(|d| min_gap_us <= 0.0 || d.abs() >= min_gap_us)
        .collect();
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    if v.len() >= 4 {
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (q25, q75) = (percentile(&s, 25.0), percentile(&s, 75.0));
        let iqr = q75 - q25;
        let (lo, hi) = (q25 - 1.5 * iqr, q75 + 1.5 * iqr);
        v.retain(|x| *x >= lo && *x <= hi);
    }
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bw = 3.0;
    // np.arange(mn, mx + bw, bw)
    let n_edges = (((mx + bw) - mn) / bw).ceil().max(0.0) as usize;
    let mut edges: Vec<f64> = (0..n_edges).map(|k| mn + k as f64 * bw).collect();
    if edges.len() < 2 {
        edges = vec![mn, mn + bw];
    }
    let nb = edges.len() - 1;
    let mut counts = vec![0usize; nb];
    for &x in &v {
        if x < edges[0] || x > edges[nb] {
            continue;
        }
        let mut b = edges.partition_point(|e| *e <= x).saturating_sub(1);
        if b >= nb {
            b = nb - 1;
        }
        counts[b] += 1;
    }
    let mut mb = 0;
    for i in 1..nb {
        if counts[i] > counts[mb] {
            mb = i;
        }
    }
    if fundamental {
        // smallest interval bin with real support: a cluster that holds only
        // every 2nd/3rd pulse of an emitter has its tallest bin at a multiple
        // of the PRI; the fundamental is the shortest well-populated interval
        let need = (counts[mb] as f64 * FUNDAMENTAL_SHARE).ceil() as usize;
        if let Some(i) = (0..nb).find(|&i| counts[i] >= need.max(2)) {
            mb = i;
        }
    }
    let (bs, be) = (edges[mb], edges[mb + 1]);
    let pts: Vec<f64> = v
        .iter()
        .cloned()
        .filter(|x| *x >= bs && if mb == nb - 1 { *x <= be } else { *x < be })
        .collect();
    let dom = if pts.is_empty() {
        (bs + be) / 2.0
    } else if fundamental {
        // median inside the bin: jitter / stray pulses at the bin edge do not pull it
        let mut q = pts.clone();
        q.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = q.len() / 2;
        if q.len() % 2 == 1 { q[m] } else { (q[m - 1] + q[m]) / 2.0 }
    } else {
        pts.iter().sum::<f64>() / pts.len() as f64
    };
    (dom, mn, mx, std0(&v))
}

struct Stats {
    n: usize,
    toa_ns: i64,
    f: (f64, f64, f64, f64),
    p: (f64, f64, f64, f64),
    pw: (f64, f64, f64, f64),
    bw: (f64, f64, f64, f64),
    doa: (f64, f64, f64, f64),
    pri: (f64, f64, f64, f64),
}

fn agg(v: &[f64]) -> (f64, f64, f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (m, mn, mx, std0(v))
}

fn stats(ms: &[Member], pri_min_gap_us: f64, fundamental: bool) -> Stats {
    let col = |f: &dyn Fn(&FlowPulse) -> f64| ms.iter().map(|m| f(m.p)).collect::<Vec<f64>>();
    let toas = col(&|p| p.toa_ns);
    let pos_min = toas.iter().cloned().filter(|t| *t > 0.0).fold(f64::INFINITY, f64::min);
    Stats {
        n: ms.len(),
        toa_ns: if pos_min.is_finite() { pos_min as i64 } else { 0 },
        f: agg(&col(&|p| p.freq_hz)),
        p: agg(&col(&|p| p.power_dbm)),
        pw: agg(&col(&|p| p.duration_ns / 1000.0)),
        bw: agg(&col(&|p| p.bandwidth_hz)),
        doa: agg(&col(&|p| p.doa_deg)),
        pri: pri_stats_ex(&toas, pri_min_gap_us, fundamental),
    }
}

/// Adds the elapsed time to the DB-writes layer when dropped.
struct DbTimer(Instant);
impl Drop for DbTimer {
    fn drop(&mut self) {
        crate::layer_timing::add_db(self.0.elapsed());
    }
}

/// Buffered CH cluster rows are sent once they reach this size.
const CH_FLUSH_BYTES: usize = 8 << 20;

pub(crate) struct Db<'a> {
    pub st: &'a AiState,
    /// One pinned PG connection so a run's (or a window's) writes can share
    /// one transaction — one disk flush instead of one per statement.
    pub client: &'a tokio_postgres::Client,
    /// Pending CH `signals.clusters` rows ("(id,[...])"), sent in large
    /// batches by `flush_ch` — ClickHouse is much faster with few big inserts.
    pub ch_buf: std::sync::Mutex<String>,
    /// cluster PRI estimator: fundamental (true) or Python's tallest bin
    pub pri_fundamental: bool,
    /// For the optional clock merge (ONNX_CLOCK_MERGE=1): signal -> emitter,
    /// the unknown_ai emitters, and every written pulse TOA per signal.
    pub sig_emitter: std::sync::Mutex<HashMap<i64, i64>>,
    pub unknown_emitters: std::sync::Mutex<BTreeSet<i64>>,
    pub sig_toas: std::sync::Mutex<HashMap<i64, Vec<f64>>>,
    pub track_toas: bool,
}

impl<'a> Db<'a> {
    pub(crate) fn new(st: &'a AiState) -> Self {
        Db {
            st,
            client: st.pg.acquire(),
            ch_buf: std::sync::Mutex::new(String::new()),
            pri_fundamental: false,
            sig_emitter: Default::default(),
            unknown_emitters: Default::default(),
            sig_toas: Default::default(),
            track_toas: false,
        }
    }

    pub(crate) async fn begin(&self) -> Result<()> {
        let _t = DbTimer(Instant::now());
        self.client.batch_execute("BEGIN").await?;
        Ok(())
    }

    /// Send buffered CH cluster rows (`force` = regardless of size).
    pub(crate) async fn flush_ch(&self, force: bool) -> Result<()> {
        let body = {
            let mut b = self.ch_buf.lock().unwrap();
            if b.is_empty() || (!force && b.len() < CH_FLUSH_BYTES) {
                return Ok(());
            }
            std::mem::take(&mut *b)
        };
        let _t = DbTimer(Instant::now());
        let tq = Instant::now();
        self.st.ch.execute(&format!("INSERT INTO signals.clusters (cluster_id, activity_ids) VALUES {body}")).await?;
        crate::layer_timing::add_ch_clusters(tq.elapsed());
        Ok(())
    }

    /// Cluster PRI de-harmonization: a cluster that holds only every k-th
    /// pulse of its signal measures k × PRI. Divide such clusters (k ≥ 2,
    /// within 2 %) by k, using the signal's median cluster PRI as reference.
    pub(crate) async fn deharmonize_pri(&self) -> Result<u64> {
        let _t = DbTimer(Instant::now());
        let n = self
            .pg()
            .execute(
                "UPDATE clusters c SET \
                   pulse_repetition_interval_us = c.pulse_repetition_interval_us / k.k, \
                   min_pulse_repetition_interval_us = c.min_pulse_repetition_interval_us / k.k, \
                   max_pulse_repetition_interval_us = c.max_pulse_repetition_interval_us / k.k, \
                   std_pulse_repetition_interval_us = c.std_pulse_repetition_interval_us / k.k \
                 FROM (SELECT c2.id, round(c2.pulse_repetition_interval_us / m.med) AS k \
                         FROM clusters c2 JOIN (SELECT signal_id, percentile_cont(0.5) WITHIN GROUP \
                              (ORDER BY pulse_repetition_interval_us) AS med FROM clusters \
                              WHERE pulse_repetition_interval_us > 0 GROUP BY signal_id) m \
                           ON m.signal_id = c2.signal_id \
                        WHERE m.med > 0 AND round(c2.pulse_repetition_interval_us / m.med) >= 2 \
                          AND abs(c2.pulse_repetition_interval_us / m.med \
                                  - round(c2.pulse_repetition_interval_us / m.med)) < 0.02) k \
                 WHERE c.id = k.id",
                &[],
            )
            .await?;
        Ok(n)
    }

    /// Optional post-pass: join unknown_ai emitters whose pulse trains keep
    /// the same clock (classic's clock test, both directions), so one unknown
    /// radar split into pieces across windows becomes one emitter again.
    /// Returns the number of emitters removed.
    pub(crate) async fn clock_merge_unknown(&self) -> Result<usize> {
        let cfg = crate::classic_v2::ClassicCfg::from_env();
        let sig_em = self.sig_emitter.lock().unwrap().clone();
        let unknown = self.unknown_emitters.lock().unwrap().clone();
        let mut em_toas: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for (sid, toas) in self.sig_toas.lock().unwrap().iter() {
            if let Some(e) = sig_em.get(sid) {
                if unknown.contains(e) {
                    em_toas.entry(*e).or_default().extend_from_slice(toas);
                }
            }
        }
        let ems: Vec<i64> = em_toas.keys().copied().collect();
        let toas: Vec<Vec<f64>> = em_toas
            .into_values()
            .map(|mut v| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v.dedup();
                v
            })
            .collect();
        use rayon::prelude::*;
        let pairs: Vec<(usize, usize)> = (0..ems.len()).flat_map(|a| (a + 1..ems.len()).map(move |b| (a, b))).collect();
        let hits: Vec<(usize, usize)> = pairs
            .par_iter()
            .filter(|(a, b)| crate::classic_v2::same_clock(&toas[*a], &toas[*b], &cfg))
            .copied()
            .collect();
        let mut parent: Vec<usize> = (0..ems.len()).collect();
        fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for (a, b) in hits {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
        let mut removed = 0usize;
        for i in 0..ems.len() {
            let r = find(&mut parent, i);
            if r != i {
                self.pg()
                    .execute("UPDATE signals SET emitter_id = $1 WHERE emitter_id = $2", &[&ems[r], &ems[i]])
                    .await?;
                self.pg().execute("DELETE FROM emitters WHERE id = $1", &[&ems[i]]).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    pub(crate) async fn commit(&self) -> Result<()> {
        let _t = DbTimer(Instant::now());
        self.client.batch_execute("COMMIT").await?;
        Ok(())
    }

    fn pg(&self) -> &tokio_postgres::Client {
        self.client
    }

    pub(crate) async fn insert_emitter(&self, name: &str, sig: Option<i32>) -> Result<i64> {
        let _t = DbTimer(Instant::now());
        let row = match sig {
            Some(s) => self
                .pg()
                .query_one(
                    "INSERT INTO emitters (name, created_at, updated_at, signature_matched_id) \
                     VALUES ($1, NOW(), NOW(), $2) RETURNING id",
                    &[&name, &s],
                )
                .await?,
            None => self
                .pg()
                .query_one(
                    "INSERT INTO emitters (name, created_at, updated_at) VALUES ($1, NOW(), NOW()) RETURNING id",
                    &[&name],
                )
                .await?,
        };
        let id = row.get::<_, i64>(0);
        if self.track_toas && name == "unknown_ai" {
            self.unknown_emitters.lock().unwrap().insert(id);
        }
        Ok(id)
    }

    /// `insert_signal`: first-row metadata, angle as doa, TOA span as duration.
    pub(crate) async fn insert_signal(&self, ms: &[Member<'_>], emitter_id: Option<i64>, angle: f64) -> Result<i64> {
        let _t = DbTimer(Instant::now());
        let f = ms[0].p;
        let tmin = ms.iter().map(|m| m.p.toa_ns).fold(f64::INFINITY, f64::min);
        let tmax = ms.iter().map(|m| m.p.toa_ns).fold(f64::NEG_INFINITY, f64::max);
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let row = self
            .pg()
            .query_one(
                "INSERT INTO signals (frequency_hz, bandwidth_hz, receiver_id, power_dbm, snr_db, doa_deg, \
                 emitter_id, is_ai, modulation_type_id, toa_nanoseconds, duration_ms, emission_type, toa_seconds) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,true,$8,$9,$10,'PULSE',NOW()) RETURNING id",
                &[
                    &(f.freq_hz.trunc()),
                    &(f.bandwidth_hz.trunc()),
                    &1i32,
                    &(f.power_dbm.trunc()),
                    &(f.snr_db.trunc()),
                    &angle,
                    &emitter_id,
                    &modulation_type_id(&f.modulation),
                    &now_ns,
                    &((tmax - tmin) / 1e6),
                ],
            )
            .await?;
        let sid = row.get::<_, i64>(0);
        if self.track_toas {
            if let Some(e) = emitter_id {
                self.sig_emitter.lock().unwrap().insert(sid, e);
            }
        }
        Ok(sid)
    }

    pub(crate) async fn touch_signal(&self, sid: i64) -> Result<()> {
        let _t = DbTimer(Instant::now());
        self.pg()
            .execute(
                "UPDATE signals SET duration_ms = EXTRACT(EPOCH FROM (NOW()::timestamp - toa_seconds)) * 1000.0 WHERE id = $1",
                &[&sid],
            )
            .await?;
        Ok(())
    }

    pub(crate) async fn set_doa(&self, sid: i64, angle: f64) -> Result<()> {
        let _t = DbTimer(Instant::now());
        self.pg().execute("UPDATE signals SET doa_deg = $1 WHERE id = $2", &[&angle, &sid]).await?;
        Ok(())
    }

    /// One `clusters` row per signal (bulk_insert_clusters_to_postgres) and
    /// one CH `signals.clusters` row with the member input ids.
    pub(crate) async fn write_clusters(&self, groups: &[(i64, Vec<Member<'_>>)]) -> Result<Vec<i64>> {
        self.write_clusters_gap(groups, 0.0).await
    }

    pub(crate) async fn write_clusters_gap(&self, groups: &[(i64, Vec<Member<'_>>)], pri_min_gap_us: f64) -> Result<Vec<i64>> {
        let _t = DbTimer(Instant::now());
        if groups.is_empty() {
            return Ok(Vec::new());
        }
        if self.track_toas {
            let mut st = self.sig_toas.lock().unwrap();
            for (sid, ms) in groups {
                st.entry(*sid).or_default().extend(ms.iter().map(|m| m.p.toa_ns));
            }
        }
        // Stats per group in parallel (PRI histogram + std are the costly part).
        use rayon::prelude::*;
        let fundamental = self.pri_fundamental;
        let stats_v: Vec<Stats> = groups.par_iter().map(|(_, ms)| stats(ms, pri_min_gap_us, fundamental)).collect();
        // One multi-row INSERT for every cluster row (PG keeps VALUES order
        // for RETURNING in a single INSERT), then one CH INSERT.
        fn f(x: f64) -> String {
            if x.is_finite() { format!("{x:?}") } else { "'NaN'::float8".to_string() }
        }
        let mut sql = String::with_capacity(groups.len() * 400);
        sql.push_str(
            "INSERT INTO clusters (name, pdw_count, toa_ns, signal_id, frequency_hz, min_frequency_hz, \
             max_frequency_hz, std_frequency_hz, power_dbm, min_power_dbm, max_power_dbm, std_power_dbm, \
             pulse_width_us, min_pulse_width_us, max_pulse_width_us, std_pulse_width_us, \
             pulse_repetition_interval_us, min_pulse_repetition_interval_us, max_pulse_repetition_interval_us, \
             std_pulse_repetition_interval_us, bandwidth_hz, min_bandwidth_hz, max_bandwidth_hz, std_bandwidth_hz, \
             doa_deg, min_doa_deg, max_doa_deg, std_doa_deg, is_ai, created_at, updated_at) VALUES ",
        );
        for (k, ((sid, _), st)) in groups.iter().zip(stats_v.iter()).enumerate() {
            if k > 0 {
                sql.push(',');
            }
            sql.push_str(&format!(
                "('emitter_{sid}_summary',{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},true,NOW(),NOW())",
                st.n as i32,
                st.toa_ns,
                *sid as i32,
                st.f.0 as i64,
                st.f.1 as i64,
                st.f.2 as i64,
                f(st.f.3),
                f(st.p.0), f(st.p.1), f(st.p.2), f(st.p.3),
                f(st.pw.0), f(st.pw.1), f(st.pw.2), f(st.pw.3),
                f(st.pri.0), f(st.pri.1), f(st.pri.2), f(st.pri.3),
                f(st.bw.0), f(st.bw.1), f(st.bw.2), f(st.bw.3),
                f(st.doa.0), f(st.doa.1), f(st.doa.2), f(st.doa.3),
            ));
        }
        sql.push_str(" RETURNING id");
        let tq = Instant::now();
        let rows = self.pg().query(sql.as_str(), &[]).await?;
        crate::layer_timing::add_pg_clusters(tq.elapsed());
        let ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();
        if ids.len() != groups.len() {
            bail!("clusters insert returned {} ids for {} rows", ids.len(), groups.len());
        }
        let mut ch = String::with_capacity(groups.iter().map(|(_, m)| m.len() * 9).sum::<usize>() + 128);
        for ((_, ms), cid) in groups.iter().zip(ids.iter()) {
            if !ch.is_empty() {
                ch.push(',');
            }
            ch.push_str(&format!("({cid},["));
            let mut seen = std::collections::HashSet::with_capacity(ms.len());
            let mut first = true;
            for m in ms {
                if seen.insert(m.p.input_id) {
                    if !first {
                        ch.push(',');
                    }
                    first = false;
                    ch.push_str(&format!("'{}'", m.p.input_id));
                }
            }
            ch.push_str("])");
        }
        {
            let mut buf = self.ch_buf.lock().unwrap();
            if !buf.is_empty() {
                buf.push(',');
            }
            buf.push_str(&ch);
        }
        self.flush_ch(false).await?;
        Ok(ids)
    }
}

// ───────────────────────── stateful flow ─────────────────────────

#[derive(Clone)]
struct Track {
    vector: Vec<f32>,
    angle: f64,
    card: usize,
    signal_id: i64,
}

/// Unknown pulse waiting in the buffer (owned copies — windows are dropped).
#[derive(Clone)]
struct Pending {
    p: FlowPulse,
    emb: Vec<f32>,
}

struct Flow<'a> {
    cfg: FlowCfg,
    db: Db<'a>,
    medoids: HashMap<i64, Vec<f32>>,
    emitter_signal: HashMap<i64, i64>,
    known_cache: HashMap<i64, (Vec<f64>, Vec<f64>)>, // signal → (powers, doas)
    buffer: Vec<Pending>,
    tracks: Vec<(i64, Track)>, // insertion-ordered dict
    track_hist: HashMap<i64, (Vec<f64>, Vec<f64>)>,
    pub n_known: usize,
    pub n_unknown_runs: usize,
}

impl<'a> Flow<'a> {
    fn track_get(&self, k: i64) -> Option<&Track> {
        self.tracks.iter().find(|(kk, _)| *kk == k).map(|(_, t)| t)
    }

    fn track_set(&mut self, k: i64, t: Track) {
        if let Some(slot) = self.tracks.iter_mut().find(|(kk, _)| *kk == k) {
            slot.1 = t;
        } else {
            self.tracks.push((k, t));
        }
    }

    /// `handle_embedding` for one window.
    async fn window(&mut self, pulses: &[FlowPulse], out: &ModelOut) -> Result<()> {
        let n = pulses.len();
        let emb = |i: usize| &out.emb[i * OUT_DIM..(i + 1) * OUT_DIM];
        let mut ids: Vec<i64> = out.ids.clone();
        let mut sims: Vec<f32> = out.sims.clone();
        for i in 0..n {
            if (1117..=1137).contains(&ids[i]) && pulses[i].freq_hz > 9e9 {
                ids[i] = 1117;
            }
        }
        let pre: Vec<bool> = sims.iter().map(|s| *s >= self.cfg.known_dot).collect();
        let mut known = pre.clone();
        if pre.iter().any(|b| *b) {
            let uniq: BTreeSet<i64> = (0..n).filter(|&i| pre[i]).map(|i| ids[i]).collect();
            for eid in uniq {
                let idx: Vec<usize> = (0..n).filter(|&i| pre[i] && ids[i] == eid).collect();
                let rows: Rows = idx.iter().map(|&i| emb(i)).collect();
                let (sim_old, mean_old): (Vec<f32>, f64) = match self.medoids.get(&eid) {
                    Some(m) => {
                        let s: Vec<f32> = rows.iter().map(|r| dot(r, m)).collect();
                        let mean = s.iter().map(|v| *v as f64).sum::<f64>() / s.len() as f64;
                        (s, mean)
                    }
                    None => {
                        let s: Vec<f32> = idx.iter().map(|&i| sims[i]).collect();
                        let mean = s.iter().map(|v| *v as f64).sum::<f64>() / s.len().max(1) as f64;
                        (s, mean)
                    }
                };
                let c = rows[medoid_idx(&rows)].to_vec();
                let sim_new: Vec<f32> = rows.iter().map(|r| dot(r, &c)).collect();
                let mean_new = sim_new.iter().map(|v| *v as f64).sum::<f64>() / sim_new.len() as f64;
                let take = if mean_old < mean_new && rows.len() > self.cfg.medoid_min_rows {
                    self.medoids.insert(eid, c);
                    &sim_new
                } else {
                    &sim_old
                };
                for (k, &i) in idx.iter().enumerate() {
                    sims[i] = take[k];
                }
            }
            for i in 0..n {
                let thr = if ids[i] == 1117 { self.cfg.sim_thr_1117 } else { self.cfg.sim_thr };
                known[i] = sims[i] >= thr && MATCHED_IDS.contains(&ids[i]);
            }
        }

        // ── known path ──
        let known_idx: Vec<usize> = (0..n).filter(|&i| known[i]).collect();
        if !known_idx.is_empty() {
            self.n_known += known_idx.len();
            let all_doa: Vec<f64> = known_idx.iter().map(|&i| pulses[i].doa_deg).collect();
            let all_pow: Vec<f64> = known_idx.iter().map(|&i| pulses[i].power_dbm).collect();
            let (angle_all, _) = angle_score(&all_doa, &all_pow);
            // pandas .unique(): order of first appearance
            let mut order: Vec<i64> = Vec::new();
            for &i in &known_idx {
                if !order.contains(&ids[i]) {
                    order.push(ids[i]);
                }
            }
            let mut sig_of: BTreeMap<i64, i64> = BTreeMap::new(); // signal → emitter key
            let mut emitter_to_signal: HashMap<i64, i64> = HashMap::new();
            for eid in order {
                match self.emitter_signal.get(&eid).copied() {
                    None => {
                        let slice: Vec<Member> = known_idx
                            .iter()
                            .filter(|&&i| ids[i] == eid)
                            .map(|&i| Member { p: &pulses[i], emb: emb(i) })
                            .collect();
                        if slice.len() > self.cfg.known_min_signal {
                            let em = self.db.insert_emitter(&eid.to_string(), signature_id(eid)).await?;
                            let sid = self.db.insert_signal(&slice, Some(em), angle_all).await?;
                            self.known_cache.insert(sid, (all_pow.clone(), all_doa.clone()));
                            self.emitter_signal.insert(eid, sid);
                            emitter_to_signal.insert(eid, sid);
                        }
                    }
                    Some(sid) => {
                        let e = self.known_cache.entry(sid).or_default();
                        e.0.extend_from_slice(&all_pow);
                        e.1.extend_from_slice(&all_doa);
                        let a = cluster_angle(&e.1, &e.0);
                        self.db.touch_signal(sid).await?;
                        self.db.set_doa(sid, a).await?;
                        emitter_to_signal.insert(eid, sid);
                    }
                }
            }
            for (e, s) in &emitter_to_signal {
                sig_of.insert(*s, *e);
            }
            // groupby target_signal_id (sorted), members in row order
            let groups: Vec<(i64, Vec<Member>)> = sig_of
                .iter()
                .map(|(sid, eid)| {
                    let ms = known_idx
                        .iter()
                        .filter(|&&i| ids[i] == *eid)
                        .map(|&i| Member { p: &pulses[i], emb: emb(i) })
                        .collect();
                    (*sid, ms)
                })
                .collect();
            self.db.write_clusters_gap(&groups, self.cfg.pri_min_gap_us).await?;
        }

        // ── unknown path (handle_unknown_pulses_buffered) ──
        let unk: Vec<usize> = (0..n).filter(|&i| !known[i]).collect();
        if unk.is_empty() {
            return Ok(());
        }
        for &i in &unk {
            self.buffer.push(Pending { p: pulses[i].clone(), emb: emb(i).to_vec() });
        }
        if self.buffer.len() >= self.cfg.buffer_threshold {
            let max_t = self.buffer.iter().map(|b| b.p.toa_ns).fold(f64::NEG_INFINITY, f64::max);
            let batch: Vec<Pending> =
                std::mem::take(&mut self.buffer).into_iter().filter(|b| b.p.toa_ns >= max_t - self.cfg.accum_ns).collect();
            self.n_unknown_runs += 1;
            self.simple_clustering(&batch).await?;
        }
        Ok(())
    }

    /// `simple_clustering` → greedy clusters → merge → per-cluster emitter.
    async fn simple_clustering(&mut self, b: &[Pending]) -> Result<()> {
        let n = b.len();
        let x: Vec<&[f32]> = b.iter().map(|p| p.emb.as_slice()).collect();
        // cluster_based_on_embedding_similarities
        let mut labels = vec![-1i64; n];
        let mut cid = 0i64;
        let mut clusters: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
        for i in 0..n {
            if labels[i] != -1 {
                continue;
            }
            let m: Vec<usize> = (0..n).filter(|&j| labels[j] == -1 && dot(x[i], x[j]) >= self.cfg.cluster_thr).collect();
            if m.len() >= self.cfg.min_cluster {
                for &j in &m {
                    labels[j] = cid;
                }
                let rows: Rows = m.iter().map(|&j| x[j]).collect();
                let mv = mean_vec(&rows);
                let d: Vec<f64> = m.iter().map(|&j| b[j].p.doa_deg).collect();
                let p: Vec<f64> = m.iter().map(|&j| b[j].p.power_dbm).collect();
                if angle_score(&d, &p).1 > 0.3 {
                    clusters.push((m, mv));
                    cid += 1;
                }
            }
        }
        if clusters.is_empty() {
            return Ok(());
        }
        // merge_clusters
        let k = clusters.len();
        let angles: Vec<f64> = clusters
            .iter()
            .map(|(m, _)| {
                let d: Vec<f64> = m.iter().map(|&j| b[j].p.doa_deg).collect();
                let p: Vec<f64> = m.iter().map(|&j| b[j].p.power_dbm).collect();
                angle_score(&d, &p).0
            })
            .collect();
        let mut adj: Vec<BTreeSet<usize>> = (0..k).map(|i| BTreeSet::from([i])).collect();
        for i in 0..k {
            for j in i + 1..k {
                let sim = dot(&clusters[i].1, &clusters[j].1) as f64;
                let fs = (sim * 8.0 + merge_angle_score(wrap_diff(angles[i], angles[j])) * 2.0) / 10.0;
                if fs > self.cfg.merge_thr {
                    adj[i].insert(j);
                    adj[j].insert(i);
                }
            }
        }
        let mut seen = vec![false; k];
        let mut merged: Vec<(Vec<usize>, Vec<f32>, f64, f64)> = Vec::new(); // members, medoid, angle, score
        for i in 0..k {
            if seen[i] {
                continue;
            }
            let mut comp = Vec::new();
            let mut q = VecDeque::from([i]);
            seen[i] = true;
            while let Some(c) = q.pop_front() {
                comp.push(c);
                for &nb in &adj[c] {
                    if !seen[nb] {
                        seen[nb] = true;
                        q.push_back(nb);
                    }
                }
            }
            let members: Vec<usize> = comp.iter().flat_map(|&c| clusters[c].0.iter().copied()).collect();
            let rows: Rows = members.iter().map(|&j| x[j]).collect();
            let med = rows[medoid_idx(&rows)].to_vec();
            let avg = rows.iter().map(|r| dot(&med, r) as f64).sum::<f64>() / rows.len() as f64;
            let d: Vec<f64> = members.iter().map(|&j| b[j].p.doa_deg).collect();
            let p: Vec<f64> = members.iter().map(|&j| b[j].p.power_dbm).collect();
            let (ang, asc) = angle_score(&d, &p);
            merged.push((members, med, ang, (avg + asc) / 3.0));
        }
        for (members, med, ang, score) in merged {
            if score > self.cfg.merged_min_score {
                self.new_emitter(b, &members, med, ang).await?;
            }
        }
        Ok(())
    }

    /// `create_instert_new_signal_emitter`.
    async fn write_signal(&mut self, ms: &[Member<'_>], angle: f64, existing: Option<i64>) -> Result<i64> {
        let sid = match existing {
            Some(s) => s,
            None => {
                let em = self.db.insert_emitter("unknown_ai", None).await?;
                self.db.insert_signal(ms, Some(em), angle).await?
            }
        };
        let groups = vec![(sid, ms.to_vec())];
        self.db.write_clusters_gap(&groups, self.cfg.pri_min_gap_us).await?;
        self.db.touch_signal(sid).await?;
        Ok(sid)
    }

    /// `handle_new_emitter`.
    async fn new_emitter(&mut self, b: &[Pending], members: &[usize], med: Vec<f32>, angle: f64) -> Result<()> {
        let ms: Vec<Member> = members.iter().map(|&j| Member { p: &b[j].p, emb: &b[j].emb }).collect();
        let len = ms.len();
        if self.tracks.is_empty() {
            let sid = if len > self.cfg.unknown_min_signal { self.write_signal(&ms, angle, None).await? } else { 0 };
            self.track_set(0, Track { vector: med, angle, card: len, signal_id: sid });
            return Ok(());
        }
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, (_, t)) in self.tracks.iter().enumerate() {
            let s = dot(&med, &t.vector);
            if s > best.1 {
                best = (i, s);
            }
        }
        let mut key = self.tracks[best.0].0;
        let tr = self.tracks[best.0].1.clone();
        let ad = wrap_diff(angle, tr.angle);
        let asc = if ad <= 40.0 { ((ad / 40.0) * std::f64::consts::FRAC_PI_2).cos().powi(2) } else { 0.0 };
        let fs = (best.1 as f64 * 8.0 + asc * 2.0) / 10.0;
        let mut old_card = 0;
        let mut signal_id = 0i64;
        let mut new_sid = 0i64;
        if fs > self.cfg.track_thr {
            old_card = tr.card;
            if tr.signal_id != 0 {
                signal_id = tr.signal_id;
                new_sid = self.write_signal(&ms, angle, Some(signal_id)).await?;
            } else if old_card + len > self.cfg.unknown_min_signal {
                new_sid = self.write_signal(&ms, angle, None).await?;
                signal_id = new_sid;
            }
        } else {
            key = self.tracks.len() as i64;
            if len > self.cfg.unknown_min_signal {
                new_sid = self.write_signal(&ms, angle, None).await?;
                signal_id = new_sid;
            }
        }
        self.track_set(key, Track { vector: med, angle, card: old_card + len, signal_id });
        let _ = self.track_get(key);
        if new_sid != 0 {
            let h = self.track_hist.entry(key).or_default();
            h.0.extend(ms.iter().map(|m| m.p.power_dbm));
            h.1.extend(ms.iter().map(|m| m.p.doa_deg));
            let a = cluster_angle(&h.1, &h.0);
            self.db.set_doa(new_sid, a).await?;
        }
        Ok(())
    }
}

/// Run the whole ONNX flow over `pulses` (CSV order).
pub async fn run(state: &AiState, pulses: Vec<FlowPulse>) -> Result<()> {
    let t0 = Instant::now();
    let url = std::env::var("ONNX_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let n_win = pulses.len() / WINDOW;
    if n_win == 0 {
        warn!("onnx_flow: {} pulses < one {WINDOW}-pulse window — nothing to do", pulses.len());
        return Ok(());
    }
    let workers: usize = env_or("ONNX_CONCURRENCY", 8);
    let ahead: usize = env_or("ONNX_MAX_AHEAD", 64);
    // Result writes are committed every N windows (one disk flush per N).
    let commit_every: usize = env_or::<usize>("ONNX_COMMIT_EVERY", 32).max(1);
    info!(
        "onnx_flow: {} pulses → {n_win} windows of {WINDOW} (tail {} dropped), model={url}, {workers} workers",
        pulses.len(),
        pulses.len() - n_win * WINDOW
    );
    let pulses = Arc::new(pulses);
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Result<ModelOut>, f64, f64)>();
    let mut handles = Vec::new();
    for _ in 0..workers.max(1) {
        let (pulses, next, done, tx, url) = (pulses.clone(), next.clone(), done.clone(), tx.clone(), url.clone());
        handles.push(std::thread::spawn(move || loop {
            let w = next.fetch_add(1, Ordering::SeqCst);
            if w >= n_win {
                break;
            }
            while w >= done.load(Ordering::SeqCst) + ahead {
                std::thread::sleep(Duration::from_millis(2));
            }
            let t = Instant::now();
            let f = window_features(&pulses[w * WINDOW..(w + 1) * WINDOW]);
            let tf = t.elapsed().as_secs_f64();
            let r = predict(&url, &f, WINDOW);
            let tm = t.elapsed().as_secs_f64() - tf;
            if tx.send((w, r, tf, tm)).is_err() {
                break;
            }
        }));
    }
    drop(tx);

    let mut flow = Flow {
        cfg: FlowCfg::from_env(),
        db: Db {
            pri_fundamental: std::env::var("ONNX_PRI_ESTIMATOR").map(|v| v != "python").unwrap_or(true),
            track_toas: std::env::var("ONNX_CLOCK_MERGE").map(|v| v == "1").unwrap_or(false),
            ..Db::new(state)
        },
        medoids: HashMap::new(),
        emitter_signal: HashMap::new(),
        known_cache: HashMap::new(),
        buffer: Vec::new(),
        tracks: Vec::new(),
        track_hist: HashMap::new(),
        n_known: 0,
        n_unknown_runs: 0,
    };
    let mut pending: BTreeMap<usize, ModelOut> = BTreeMap::new();
    let (mut want, mut t_feat, mut t_model, mut t_post) = (0usize, 0f64, 0f64, 0f64);
    while want < n_win {
        let (w, r, tf, tm) = rx.recv().await.ok_or_else(|| anyhow!("onnx_flow: workers stopped early"))?;
        t_feat += tf;
        t_model += tm;
        pending.insert(w, r.with_context(|| format!("window {w}"))?);
        while let Some(out) = pending.remove(&want) {
            let t = Instant::now();
            if want % commit_every == 0 {
                flow.db.begin().await?;
            }
            flow.window(&pulses[want * WINDOW..(want + 1) * WINDOW], &out).await?;
            if want % commit_every == commit_every - 1 || want + 1 == n_win {
                flow.db.commit().await?;
            }
            t_post += t.elapsed().as_secs_f64();
            want += 1;
            done.store(want, Ordering::SeqCst);
        }
    }
    for h in handles {
        let _ = h.join();
    }
    flow.db.flush_ch(true).await?;
    if flow.db.track_toas {
        let t = Instant::now();
        let n = flow.db.clock_merge_unknown().await?;
        crate::layer_timing::emit_dur("onnx.clock_merge", t.elapsed());
        info!("onnx_flow: clock merge joined {n} unknown_ai emitters into others");
    }
    if std::env::var("ONNX_PRI_HARMONIC").map(|v| v != "0").unwrap_or(true) {
        let n = flow.db.deharmonize_pri().await?;
        info!("onnx_flow: de-harmonized PRI of {n} clusters");
    }
    {
        use crate::layer_timing::{emit, emit_dur};
        emit("onnx.windows", n_win as f64);
        emit("onnx.pulses_in_windows", (n_win * WINDOW) as f64);
        emit("onnx.features_cpu_sum", t_feat * 1e3);
        emit("onnx.model_service_sum", t_model * 1e3);
        emit("onnx.model_per_window", t_model * 1e3 / n_win as f64);
        emit("onnx.postprocess_inorder", t_post * 1e3);
        crate::layer_timing::emit_db("onnx");
        emit_dur("onnx.total_wall", t0.elapsed());
    }
    info!(
        "onnx_flow: done in {:.2}s — {n_win} windows, known pulses={}, unknown clustering runs={}, \
         worker time features={:.2}s model={:.2}s, in-order post-processing={:.2}s",
        t0.elapsed().as_secs_f64(),
        flow.n_known,
        flow.n_unknown_runs,
        t_feat,
        t_model,
        t_post
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denom_matches_numpy_layout() {
        let c = vec![1.0, 2.0, 4.0, 8.0];
        let d = denom(&c);
        // gaps 1,2,4 → prev/next mins: 1,1,2,4 → sigma*2/3
        let s: Vec<f64> = [1.0, 1.0, 2.0, 4.0].iter().map(|g| g * 2.0 / 3.0).collect();
        for i in 0..4 {
            assert!((d[i] - 2.0 * s[i] * s[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn fast_toa_kernels_match_full_sum() {
        let k = feat_consts();
        for &d0 in &[0.0, 0.3, 9.9, 10.0, 57.3, 148.9, 1000.0, 4999.0, 5000.0, 7000.0, 1e7] {
            let mut fast = [0f64; NUM_TOA];
            toa_kernels(d0, k, &mut fast);
            for c in 0..NUM_TOA {
                let d = d0 - k.toa_c[c];
                let full = (-(d * d) / k.toa_den[c]).exp();
                assert!((fast[c] - full).abs() < 1e-12, "d0={d0} c={c} fast={} full={full}", fast[c]);
            }
        }
    }

    #[test]
    fn pri_stats_dominant_bin() {
        let toas: Vec<f64> = (0..20).map(|i| i as f64 * 100_000.0).collect();
        let (m, mn, mx, sd) = pri_stats(&toas, 0.0);
        assert!((m - 100.0).abs() < 1e-9 && mn == 100.0 && mx == 100.0 && sd == 0.0);
    }

    #[test]
    fn angle_wraps() {
        assert!((wrap_diff(350.0, 10.0) - 20.0).abs() < 1e-9);
        let (a, s) = angle_score(&[10.0, 12.0, 11.0], &[-50.0, -40.0, -45.0]);
        assert!(a > 10.0 && a < 12.5 && s == 1.0);
    }
}
