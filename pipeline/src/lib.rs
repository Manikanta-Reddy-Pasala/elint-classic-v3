//! ai-processor — full POC-replicated AI pipeline (direct CH → PG/CH).
//!
//! Flow (all Rust, no Kafka, no Go):
//!   ClickHouse `signals.pdw` (direct HTTP read, polled cursor)
//!     → features (458-dim RBF, the model's real input width)
//!     → ONNX inference (1024-dim embedding)
//!     → known_match (cosine vs in-memory tracks, AoA gate)
//!         ├─ PASS → PG `signals` (emitter_id) + EMA medoid update
//!         └─ FAIL → HDBSCAN + agglomerative merge + score gate
//!             └─ PASS → PG `emitters` (new, unknown_ai) + CH `clusters`
//!
//! The `ai_processor` library is used by the `engine` crate
//! (`ai_processor::AiState` / `ai_processor::run_batch`) and also builds
//! a standalone binary that polls ClickHouse on its own.

pub mod cache;
pub mod ch_read;
pub mod classic_classify;
pub mod classic_embed;
pub mod refresh;
pub mod refresh_logger;
pub mod cluster_merge;
pub mod config;
pub mod db;
pub mod features;
pub mod global_match;
pub mod hdbscan;
pub mod engine;
pub mod inference;
pub mod model_fetch;
pub mod known_match;
pub mod onnx_predict;
pub mod onnx_flow;
pub mod classic_v2;
pub mod pri_pattern;
pub mod layer_timing;
pub mod refresh_api_core;

// ── Unit tests, kept in separate files under `src/tests/` ────────────────────
// (moved out of the inline `#[cfg(test)]` blocks in each module)

use std::sync::Arc;
use std::time::{Duration, Instant};



use anyhow::Result;
use crate::hdbscan::dot;
use rayon::prelude::*;
use tracing::{info, warn};

use crate::cache::Cache;
use crate::ch_read::host_port;
use crate::config::Config;
use crate::features::Pulse as FeaturePulse;
use crate::inference::MODEL_INPUT_DIM;
use crate::known_match::{apply_ema, match_known};

/// Default agglomerative merge threshold (overridable via MERGE_THRESHOLD env).
/// A cosine similarity >= this causes two clusters to be merged.
/// Lower = more merging (fewer emitters). The Python reference used 0.7.
const DEFAULT_MERGE_THRESHOLD: f64 = 0.7;
/// Default POC cluster score gate (overridable via SCORE_GATE env).
/// Clusters with a score below this are dropped (not written to DB).
/// Lower = more emitters pass through. The Python reference used 0.6.
const DEFAULT_SCORE_GATE: f64 = 0.6;

/// Read the effective merge threshold (env MERGE_THRESHOLD or default).
#[allow(unused_variables)]
pub fn merge_threshold(cfg: &Config) -> f64 {
    std::env::var("MERGE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MERGE_THRESHOLD)
}
/// Read the effective score gate (env SCORE_GATE or default).
#[allow(unused_variables)]
pub fn score_gate(cfg: &Config) -> f64 {
    std::env::var("SCORE_GATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SCORE_GATE)
}

/// One PDW pulse as consumed by the AI stage (band-routed by the caller).
#[derive(Debug, Clone)]
pub struct AiPulse {
    pub id: i64,
    pub center_frequency_hz: f64,
    pub power_dbm: f64,
    pub snr_db: f64,
    pub duration_ns: f64,
    pub bandwidth_hz: f64,
    pub time_of_arrival_ns: f64,
    pub direction_of_arrival_deg: f64,
    pub modulation: String,
    /// Ground-truth emitter label from the CSV `emitter_id` column
    /// (None when absent) — carried into `AiPulseMeta` for naming.
    pub emitter_label: Option<String>,
}

/// Shared AI state for in-process callers (the engine).
pub struct AiState {
    pub cache: Arc<Cache>,
    pub tracks: tokio::sync::Mutex<Vec<crate::cache::EmitterTrack>>,
    /// Swappable embedding engine (ONNX default; classic opt-in).
    pub engine: Arc<dyn crate::engine::EmbeddingEngine>,
    pub cfg: Config,
    pub ch_http: String,
    /// Persistent Postgres connection set (all PG reads/writes go through it).
    pub pg: db::PgState,
    /// Persistent keep-alive ClickHouse write socket.
    pub ch: db::ChState,
    /// Buffer of (embeddings, pulse_metadata) collected across all
    /// 1024-pulse windows. Consumed by `global_match::finalize` at the
    /// end of the run for one global HDBSCAN + merge + gate + DB writes.
    pub unknowns_buffer: std::sync::Mutex<
        Vec<crate::global_match::BufferedBatch>,
    >,
}

impl AiState {
    /// PG connections to open at startup (default 4).
    pub async fn new_with_db(cfg: Config, pg: db::PgState, ch: db::ChState) -> Result<Self> {
        db::ensure_schema(&pg).await?;
        let cache = Arc::new(Cache::new());
        let mut tracks = db::load_tracks(&pg).await?;
        for t in &tracks {
            cache.save_track(t);
        }
        info!("ai_processor: seeded {} known tracks from PG", tracks.len());
        // ONNX is the default embedding engine (like the Python repo).
        // `CLASSIC=1` / `EMBED=classic` select the in-house engine.
        let engine: Arc<dyn crate::engine::EmbeddingEngine> = crate::engine::build_engine_from_env(
            &cfg.onnx_path,
            cfg.onnx_session_count,
            cfg.onnx_threads,
        );
        info!("embedding engine: {} (out_dim={})", engine.name(), engine.out_dim());
        let (ch_host, ch_port) = host_port(&cfg.ch_url);
        let ch_http = format!("http://{ch_host}:{ch_port}");
        let ch_db = std::env::var("CLICKHOUSE_DB").ok().filter(|d| !d.is_empty());
        let ch = db::ChState::new(&cfg.ch_url, ch_db);
        db::ensure_clusters_ch(&ch).await;
        Ok(AiState {
            cache,
            tracks: tokio::sync::Mutex::new(tracks),
            engine,
            cfg,
            ch_http,
            pg,
            ch,
            unknowns_buffer: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Convenience: open the PG pool + CH write socket from cfg and build
    /// the state in one call.
    pub async fn new(cfg: Config) -> Result<Self> {
        let pg_n: usize = std::env::var("PG_CONNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let pg = db::PgState::connect(&cfg.pg_url, pg_n).await?;
        let ch_db = std::env::var("CLICKHOUSE_DB").ok().filter(|d| !d.is_empty());
        let ch = db::ChState::new(&cfg.ch_url, ch_db);
        Self::new_with_db(cfg, pg, ch).await
    }
}

/// Run the full AI stage on a batch of pulses, EXCEPT the unknown-pulse
/// clustering (HDBSCAN + merge + sinks). Used by the engine in E2E_MATCH
/// mode: embeddings are collected across batches, then a single global
/// HDBSCAN is run at the end so the cluster output matches the old
/// Python pipeline (one global HDBSCAN over the whole file).
pub async fn run_batch_no_cluster(state: &AiState, pulses: &[AiPulse]) -> Result<Vec<Vec<f32>>> {
    let t0 = Instant::now();

    // 1. Feature vectors (458-dim, the model's real input width).
    let f_pulses: Vec<FeaturePulse> = pulses
        .iter()
        .map(|m| FeaturePulse {
            freq: m.center_frequency_hz / 10e5,     // Python app.py: freq_raw / 10e5
            power: m.power_dbm,
            raw_pw: m.duration_ns / 1e3,           // us   (Python: ns/1000)
            raw_toa: m.time_of_arrival_ns / 1e3,   // us   (Python: pd.Timestamp ns -> us)
        })
        .collect();
    let features = features::generate_features(&f_pulses);
    let t_features = t0.elapsed();

    // 2. Embeddings — swappable engine (ONNX default, classic when
    //    EMBED=classic / CLASSIC=1). One call: pad to model width, embed.
    let out_dim = state.engine.out_dim();
    let flat_dim = crate::engine::model_input_dim();
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|row| {
            let mut padded: Vec<f32> = row.iter().map(|&x| x as f32).collect();
            padded.resize(flat_dim, 0.0);
            padded
        })
        .collect();
    let emb_flat = state.engine.embed(&flat, pulses.len(), flat_dim).unwrap_or_else(|e| {
        warn!(
            "ai_processor: engine '{}' failed ({e}); using zero embeddings",
            state.engine.name()
        );
        vec![0.0; pulses.len() * out_dim]
    });
    let embeddings: Vec<Vec<f32>> = (0..pulses.len())
        .map(|i| emb_flat[i * out_dim..(i + 1) * out_dim].to_vec())
        .collect();
    let t_onnx = t0.elapsed();
    info!(
        "TIMING [nomatch] features={:.1}ms onnx_total={:.1}ms ({} msgs, {} sessions)",
        t_features.as_secs_f64() * 1000.0,
        t_onnx.as_secs_f64() * 1000.0,
        pulses.len(),
        state.engine.parallelism()
    );

    // 3. Known match (same as run_batch: PG signal writes + EMA updates).
    // In no-cluster mode (global match), we skip the PG signal writes
    // entirely — they would write 20k rows to PG and block on locks.
    // All pulses are treated as unknown and collected for global HDBSCAN.
    info!(
        "run_batch_no_cluster: skipping known-match + PG writes, collecting all {} as unknown",
        pulses.len()
    );
    let unk: Vec<Vec<f32>> = embeddings;
    Ok(unk)
}

/// True when the in-house classic bag-of-windows embedding replaces ONNX.
/// `EMBED=classic` or `CLASSIC=1` forces it; otherwise ONNX is the
/// default engine (like the Python repo).
pub fn classic_embed_on() -> bool {
    crate::classic_classify::classic_classify_on()
        || std::env::var("EMBED").map(|v| v == "classic").unwrap_or(false)
}



/// Like `run_batch`, but every pulse is routed straight to HDBSCAN —
/// no known-match pass, no PG `signals` writes, no EMA updates.
///
/// Used by the refresh API in `REFRESH_SKIP_KNOWN_MATCH` mode: after a
/// wipe the known-track table is empty (or irrelevant), so the match
/// pass is pure overhead. Results (clusters → emitters → CH clusters)
/// are identical to `run_batch` with zero known matches.
pub async fn run_batch_no_known(state: &AiState, pulses: &[AiPulse]) -> Result<()> {
    let t0 = Instant::now();
    let n = pulses.len();

    // 1. Feature vectors (458-dim, the model's real input width) — write
    //    straight into the flat model buffer, no per-row Vec allocation.
    let flat_dim = crate::engine::model_input_dim();
    let flat: Vec<f32>;
    let (emb_flat, onnx_bs) = {
        // Match the Python feature conventions exactly
        // (app_main_lobe/app.py): freq Hz/10e5, pw ns/1000 (µs),
        // toa ns/1000 (µs).
        let f_pulses: Vec<FeaturePulse> = pulses
            .iter()
            .map(|m| FeaturePulse {
                freq: m.center_frequency_hz / 10e5, // Python: Hz/10e5
                power: m.power_dbm,
                raw_pw: m.duration_ns / 1e3,       // µs
                raw_toa: m.time_of_arrival_ns / 1e3, // µs
            })
            .collect();
        let feature_rows = features::generate_features(&f_pulses);
        let mut flat_buf: Vec<f32> = vec![0.0f32; n * flat_dim];
        for (i, row) in feature_rows.iter().enumerate() {
            let d = row.len().min(flat_dim);
            for (slot, &v) in flat_buf[i * flat_dim..i * flat_dim + d].iter_mut().zip(row.iter()) {
                *slot = v as f32;
            }
        }
        let onnx_bs = if classic_embed_on() { 0 } else { state.cfg.onnx_batch_size.max(1) };
        let out_dim = state.engine.out_dim();
        let mut out = Vec::with_capacity(n * out_dim);
        if onnx_bs == 0 {
            // Classic in-house engine — one whole-file call.
            for v in state.engine.embed(&flat_buf, n, flat_dim).unwrap_or_else(|e| {
                warn!(
                    "ai_processor: engine '{}' failed ({e}); using zero embeddings",
                    state.engine.name()
                );
                vec![0.0; n * out_dim]
            }) {
                out.push(v);
            }
        } else {
            for (i, chunk) in flat_buf.chunks(onnx_bs * flat_dim).enumerate() {
                let chunk_n = chunk.len() / flat_dim;
                state.engine.embed(chunk, chunk_n, flat_dim).unwrap_or_else(|e| {
                    warn!("ai_processor: engine '{}' sub-batch {i} failed ({e}); using zero embeddings", state.engine.name());
                    vec![0.0; chunk_n * out_dim]
                })
                .drain(..)
                .for_each(|v| out.push(v));
            }
        }
        flat = flat_buf;
        (out, onnx_bs)
    };
    let t_features = t0.elapsed();
    let out_dim = if n > 0 { emb_flat.len() / n } else { 0 };
    let t_onnx = t_features;
    info!(
        "TIMING features={:.1}ms onnx_total={:.1}ms ({} msgs, onnx_batch={}, {} sessions)",
        t_features.as_secs_f64() * 1000.0,
        t_onnx.as_secs_f64() * 1000.0,
        n,
        onnx_bs.max(1),
        state.engine.parallelism()
    );

    // ONNX-only predictions are routed to emitters/signals before HDBSCAN.
    // This block is reached only in non-classic mode; classic uses its own
    // classifier and keeps the original whole-file embedding call above.
    //
    // The labels MUST travel with the batch they belong to: a static
    // "last call wins" store (the old way) let a later batch's labels be
    // applied to an earlier batch's pulses, which is why known emitters
    // went missing on multi-window files.
    //
    // Reuse the feature rows already built for the embedding call above —
    // calling `generate_features` a second time for the same pulses
    // wasted ~1% of the pipeline time.
    let batch_predictions: Option<crate::onnx_predict::OnnxPredictions> =
        if state.engine.name() == "onnx" {
            let mut predicted = Vec::with_capacity(n);
            let mut scores = Vec::with_capacity(n);
            let mut uncovered = 0;
            for chunk in flat.chunks(state.cfg.onnx_batch_size.max(1) * flat_dim) {
                let cn = chunk.len() / flat_dim;
                let out = state.engine.embedding_output(chunk, cn, flat_dim).unwrap_or_else(|e| {
                    warn!("onnx: prediction output unavailable: {e}");
                    crate::engine::EmbeddingOutput { embeddings: vec![], labels: vec![-1; cn], similarities: vec![0.0; cn] }
                });
                if out.embeddings.len() != cn * out_dim { uncovered += cn; }
                predicted.extend(out.labels); scores.extend(out.similarities);
            }
            if predicted.len() == n && scores.len() == n {
                info!("onnx: prediction batch rows={} known={} uncovered={}", n, scores.iter().zip(predicted.iter()).filter(|(s, l)| **s >= state.cfg.known_threshold && **l >= 0).count(), uncovered);
                Some(crate::onnx_predict::OnnxPredictions { labels: predicted, similarities: scores })
            } else {
                None
            }
        } else {
            None
        };
    // 3. Store embeddings + metadata in the state buffer for global merge.
    //    No per-window HDBSCAN — the global finalize at the end runs ONE
    //    HDBSCAN over all pulses, then merge + gate + DB writes.
    let dim = out_dim;
    let embeddings: Vec<Vec<f32>> = (0..pulses.len())
        .map(|i| emb_flat[i * dim..(i + 1) * dim].to_vec())
        .collect();
    let meta: Vec<crate::global_match::AiPulseMeta> = pulses
        .iter()
        .map(|p| {
            let pri_ns = crate::refresh::pri_by_id_pub().read().unwrap().get(&p.id).copied().unwrap_or(0.0);
            crate::global_match::AiPulseMeta {
                id: p.id,
                center_frequency_hz: p.center_frequency_hz,
                power_dbm: p.power_dbm,
                snr_db: p.snr_db,
                duration_ns: p.duration_ns,
                bandwidth_hz: p.bandwidth_hz,
                time_of_arrival_ns: p.time_of_arrival_ns,
                direction_of_arrival_deg: p.direction_of_arrival_deg,
                modulation: p.modulation.clone(),
                pulse_repetition_interval_ns: pri_ns,
                emitter_label: p.emitter_label.clone(),
            }
        })
        .collect();
    // Harness `signals.csv_input.id` per pulse in batch order — the CH
    // signals.clusters.activity_ids join key (None in the map = keep the
    // engine's own pulse id).
    let input_ids: Vec<i64> = {
        let map = crate::refresh::PULSE_INPUT_IDS
            .get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()));
        let g = map.read().unwrap();
        pulses
            .iter()
            .map(|p| g.get(&p.id).copied().unwrap_or(p.id))
            .collect()
    };
    state
        .unknowns_buffer
        .lock()
        .unwrap()
        .push(crate::global_match::BufferedBatch {
            embeddings,
            meta,
            input_ids,
            predictions: batch_predictions,
        });

    let t_total = t0.elapsed();
    info!(
        "TIMING batch_no_known buffered={:.1}ms ({} pulses → global buffer, total {} pulses buffered)",
        t_total.as_secs_f64() * 1000.0,
        n,
        state
            .unknowns_buffer
            .lock()
            .unwrap()
            .iter()
            .map(|b| b.embeddings.len())
            .sum::<usize>()
    );
    Ok(())
}

/// Like `run_batch`, but every pulse is routed straight to HDBSCAN —
/// no known-match pass, no PG `signals` writes, no EMA updates.
///
/// Used by the refresh API in `REFRESH_SKIP_KNOWN_MATCH` mode: after a
/// wipe the known-track table is empty (or irrelevant), so the match
/// pass is pure overhead. Results (clusters → emitters → CH clusters)
/// are identical to `run_batch` with zero known matches.
///
/// NOTE: this is the OLD per-window path kept for backward compat.
/// Prefer `run_batch_no_known` + `global_match::finalize`.
pub async fn run_batch_no_known_legacy(state: &AiState, pulses: &[AiPulse]) -> Result<()> {
    let t0 = Instant::now();
    let n = pulses.len();

    let flat_dim = crate::engine::model_input_dim();
    let mut flat: Vec<f32> = vec![0.0f32; n * flat_dim];
    {
        let f_pulses: Vec<FeaturePulse> = pulses
            .iter()
            .map(|m| FeaturePulse {
                freq: m.center_frequency_hz / 10e5,     // Python: Hz/10e5
                power: m.power_dbm,
                raw_pw: m.duration_ns / 1e3,           // us   (Python: ns/1000)
                raw_toa: m.time_of_arrival_ns / 1e3,   // us   (Python: pd.Timestamp ns -> us)
            })
            .collect();
        let feature_rows = features::generate_features(&f_pulses);
        for (i, row) in feature_rows.iter().enumerate() {
            let d = row.len().min(flat_dim);
            for (slot, &v) in flat[i * flat_dim..i * flat_dim + d].iter_mut().zip(row.iter()) {
                *slot = v as f32;
            }
        }
    }
    let onnx_bs = state.cfg.onnx_batch_size.max(1);
    let out_dim = state.engine.out_dim();
    let mut emb_flat: Vec<f32> = Vec::with_capacity(n * out_dim);
    for (i, chunk) in flat.chunks(onnx_bs * flat_dim).enumerate() {
        let chunk_n = chunk.len() / flat_dim;
        state.engine.embed(chunk, chunk_n, flat_dim).unwrap_or_else(|e| {
            warn!(
                "ai_processor: engine '{}' sub-batch {i} failed ({e}); using zero embeddings",
                state.engine.name()
            );
            vec![0.0; chunk_n * out_dim]
        })
        .drain(..)
        .for_each(|v| emb_flat.push(v));
    }
    let dim = if !emb_flat.is_empty() { emb_flat.len() / n } else { out_dim };

    let unk_doas: Vec<f64> = pulses.iter().map(|m| m.direction_of_arrival_deg).collect();
    let unk_mods: Vec<String> = pulses.iter().map(|p| p.modulation.clone()).collect();
    let unk_freqs: Vec<f64> = pulses.iter().map(|p| p.center_frequency_hz).collect();
    let unk_ids: Vec<i64> = pulses.iter().map(|p| p.id).collect();

    let hd = hdbscan::hdbscan_flat(
        &emb_flat,
        n,
        dim,
        state.cfg.min_cluster_size,
        state.cfg.max_samples,
    );
    let n_clusters = hd.labels.iter().filter(|&&l| l >= 0).collect::<std::collections::HashSet<_>>().len();
    info!(
        "TIMING hdbscan_legacy={} unknowns, {} clusters",
        n, n_clusters
    );

    let mut medoids: Vec<Vec<f32>> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut angles: Vec<f64> = Vec::new();
    let mut mods: Vec<f64> = Vec::new();
    let mut freqs: Vec<f64> = Vec::new();
    let mut member_ids: Vec<Vec<i64>> = Vec::new();

    let mut by_label: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
    for (ci, &label) in hd.labels.iter().enumerate() {
        if label >= 0 {
            by_label.entry(label).or_default().push(ci);
        }
    }

    struct ClusterStats {
        label: i64,
        medoid: Vec<f32>,
        size: usize,
        angle: f64,
        mod_: f64,
        freq: f64,
        ids: Vec<i64>,
    }
    let stats: Vec<ClusterStats> = by_label
        .par_iter()
        .filter(|(_, idxs)| idxs.len() >= state.cfg.min_cluster_size)
        .map(|(&label, idxs)| {
            let row = |ci: usize| -> &[f32] { &emb_flat[ci * dim..(ci + 1) * dim] };
            let mut best_score = f64::MIN;
            let mut best_i = *idxs.first().unwrap_or(&0);
            for &ci in idxs.iter() {
                let mut sum = 0f64;
                for &oj in idxs.iter() {
                    sum += dot(row(ci), row(oj)) as f64;
                }
                let mean = sum / idxs.len() as f64;
                if mean > best_score {
                    best_score = mean;
                    best_i = ci;
                }
            }
            let mut angle_sum = 0f64;
            let mut w_sum = 0f64;
            for &ci in idxs.iter() {
                let w = row(ci).iter().sum::<f32>() as f64;
                angle_sum += unk_doas[ci] * w.max(1.0);
                w_sum += w.max(1.0);
            }
            let angle = if w_sum > 0.0 { angle_sum / w_sum } else { 0.0 };
            let mod_ = unk_mods[*idxs.first().unwrap_or(&0)].parse::<f64>().unwrap_or(0.0);
            let freq: f64 = idxs.iter().map(|&ci| unk_freqs[ci]).sum::<f64>() / idxs.len() as f64;
            ClusterStats {
                label,
                medoid: row(best_i).to_vec(),
                size: idxs.len(),
                angle,
                mod_,
                freq,
                ids: idxs.iter().map(|&ci| unk_ids[ci]).collect(),
            }
        })
        .collect();

    for s in &stats {
        medoids.push(s.medoid.clone());
        sizes.push(s.size);
        angles.push(s.angle);
        mods.push(s.mod_);
        freqs.push(s.freq);
        member_ids.push(s.ids.clone());
    }

    let mt = merge_threshold(&state.cfg);
    let sg = score_gate(&state.cfg);
    let merged = cluster_merge::merge_clusters(
        &medoids, &sizes, &angles, &mods, &freqs, &member_ids, mt,
    );
    info!(
        "ai_processor: merged {} → {} clusters (threshold={})",
        medoids.len(), merged.len(), mt
    );

    let gated: Vec<&cluster_merge::MergedCluster> =
        merged.iter().map(|(mc, _)| mc).filter(|mc| mc.score >= sg).collect();
    let t_sink = Instant::now();
    for mc in &gated {
        let mc_freq = mc.mean_freq_hz;
        let emitter_id = match db::insert_emitter_pg(
            &state.pg, None, "unknown_ai", mc_freq, 0.0, mc.angle, 0.0, "unknown", &mc.medoid,
        ).await {
            Ok(id) => id,
            Err(e) => {
                warn!("ai_processor: insert_emitter failed: {e}");
                continue;
            }
        };
        if let Err(e) = db::insert_cluster_ch(&state.ch, emitter_id, mc, mc_freq, 0.0).await {
            warn!("ai_processor: insert_cluster failed: {e}");
        }
        info!(
            "ai_processor: NEW EMITTER id={emitter_id} score={:.3} size={} freq={:.0}Hz",
            mc.score, mc.size, mc_freq
        );
    }
    let t_sink_ms = t_sink.elapsed().as_secs_f64() * 1000.0;

    let t_total = t0.elapsed();
    let sink_pg_ms = crate::db::pg_sink_ns().swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let sink_ch_ms = crate::db::ch_sink_ns().swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    info!(
        "TIMING batch_done total={:.1}ms sink_pg={:.1}ms sink_ch={:.1}ms sink_wall={:.1}ms ({} pulses, no known-match, {} emitters)",
        t_total.as_secs_f64() * 1000.0,
        sink_pg_ms,
        sink_ch_ms,
        t_sink_ms,
        n,
        gated.len()
    );
    Ok(())
}

/// Run the full AI stage on a batch of pulses (in-process entry point).
pub async fn run_batch(state: &AiState, pulses: &[AiPulse]) -> Result<()> {
    let t0 = Instant::now();

    // 1. Feature vectors (458-dim, the model's real input width).
    let f_pulses: Vec<FeaturePulse> = pulses
        .iter()
        .map(|m| FeaturePulse {
            freq: m.center_frequency_hz / 10e5,     // Python app.py: freq_raw / 10e5
            power: m.power_dbm,
            raw_pw: m.duration_ns / 1e3,           // us   (Python: ns/1000)
            raw_toa: m.time_of_arrival_ns / 1e3,   // us   (Python: pd.Timestamp ns -> us)
        })
        .collect();
    let features = features::generate_features(&f_pulses);
    let t_features = t0.elapsed();

    // 2. Embeddings (1024-dim each for ONNX) — swappable engine pool.
    let flat_dim = MODEL_INPUT_DIM;
    let flat: Vec<f32> = features
        .iter()
        .flat_map(|row| {
            let mut padded: Vec<f32> = row.iter().map(|&x| x as f32).collect();
            padded.resize(flat_dim, 0.0);
            padded
        })
        .collect();
    let out_dim = state.engine.out_dim();
    let emb_flat = state.engine.embed(&flat, pulses.len(), flat_dim).unwrap_or_else(|e| {
        warn!("ai_processor: engine '{}' failed ({e}); using zero embeddings", state.engine.name());
        vec![0.0; pulses.len() * out_dim]
    });
    let embeddings: Vec<Vec<f32>> = (0..pulses.len())
        .map(|i| emb_flat[i * out_dim..(i + 1) * out_dim].to_vec())
        .collect();
    let t_onnx = t0.elapsed();
    info!(
        "TIMING features={:.1}ms onnx_total={:.1}ms ({} msgs, {} sessions)",
        t_features.as_secs_f64() * 1000.0,
        t_onnx.as_secs_f64() * 1000.0,
        pulses.len(),
        state.engine.parallelism()
    );

    // 3. Known match.
    let doas: Vec<f64> = pulses.iter().map(|m| m.direction_of_arrival_deg).collect();
    let tracks = state.tracks.lock().await.clone();
    let track_name_by_id: std::collections::HashMap<i64, String> = tracks
        .iter()
        .map(|t| (t.signal_id, track_name(&t.modulation).to_string()))
        .collect();
    let matches = match_known(
        &embeddings,
        &doas,
        &tracks,
        state.cfg.known_threshold,
        state.cfg.angle_gate,
    );

    let mut known_count = 0u64;
    let mut unknown_idx: Vec<usize> = Vec::new();
    let mut tracks_mut = state.tracks.lock().await;

    for (i, m) in matches.iter().enumerate() {
        let m = m.clone();
        if m.matched {
            known_count += 1;
            apply_ema(&state.cache, m.signal_id, &embeddings[i], state.cfg.ema_alpha);
            if let Some(t) = tracks_mut.iter_mut().find(|t| t.signal_id == m.signal_id) {
                for (d, &nv) in embeddings[i].iter().enumerate() {
                    if d < t.vector.len() {
                        t.vector[d] =
                            (state.cfg.ema_alpha * nv + (1.0 - state.cfg.ema_alpha) * t.vector[d])
                                as f32;
                    }
                }
                t.cardinality += 1;
            }
            // Direct PG write (insert_signal_pg accumulates into TIMING_PG_NS).
            // Copy the track's display name onto the signal row — the test
            // harness filters `signals.name`.
            let sig_name = track_name_by_id.get(&m.signal_id).cloned().unwrap_or_default();
            if let Err(e) = db::insert_signal_pg(
                &state.pg,
                m.signal_id,
                &sig_name,
                pulses[i].center_frequency_hz,
                0.0,
                pulses[i].direction_of_arrival_deg,
                pulses[i].power_dbm,
                &pulses[i].modulation,
            )
            .await
            {
                warn!("ai_processor: insert_signal failed: {e}");
            }
        } else {
            unknown_idx.push(i);
        }
    }
    let t_known = t0.elapsed();
    info!(
        "TIMING known_match_done total={:.1}ms ({known_count} known, {} unknown)",
        t_known.as_secs_f64() * 1000.0,
        unknown_idx.len()
    );

    // 4. HDBSCAN on the unknowns.
    if !unknown_idx.is_empty() {
        let unk_embeds: Vec<Vec<f32>> =
            unknown_idx.iter().map(|&i| embeddings[i].clone()).collect();
        let unk_doas: Vec<f64> = unknown_idx.iter().map(|&i| doas[i]).collect();
        let unk_mods: Vec<String> =
            unknown_idx.iter().map(|&i| pulses[i].modulation.clone()).collect();
        let unk_freqs: Vec<f64> =
            unknown_idx.iter().map(|&i| pulses[i].center_frequency_hz).collect();
        let unk_ids: Vec<i64> = unknown_idx.iter().map(|&i| pulses[i].id).collect();

        let t_hd = Instant::now();
        let hd = hdbscan::hdbscan_cosine(
            &unk_embeds,
            state.cfg.min_cluster_size,
            state.cfg.max_samples,
        );
        let t_hdbscan = t_hd.elapsed();
        let n_clusters = hd
            .labels
            .iter()
            .filter(|&&l| l >= 0)
            .collect::<std::collections::HashSet<_>>()
            .len();
        info!(
            "TIMING hdbscan={:.1}ms ({} unknowns, {} clusters, min_size={})",
            t_hdbscan.as_secs_f64() * 1000.0,
            unk_embeds.len(),
            n_clusters,
            state.cfg.min_cluster_size
        );
        let mut medoids: Vec<Vec<f32>> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        let mut angles: Vec<f64> = Vec::new();
        let mut mods: Vec<f64> = Vec::new();
        let mut freqs: Vec<f64> = Vec::new();
        let mut member_ids: Vec<Vec<i64>> = Vec::new();

        let mut by_label: std::collections::HashMap<i64, Vec<usize>> =
            std::collections::HashMap::new();
        for (ci, &label) in hd.labels.iter().enumerate() {
            if label >= 0 {
                by_label.entry(label).or_default().push(ci);
            }
        }

        // Parallel per-cluster stats: medoid, size, angle, mod, freq, member ids.
        struct ClusterStats {
            label: i64,
            medoid: Vec<f32>,
            size: usize,
            angle: f64,
            mod_: f64,
            freq: f64,
            ids: Vec<i64>,
        }
        let dim = unk_embeds[0].len();
        let stats: Vec<ClusterStats> = by_label
            .par_iter()
            .filter(|(_, idxs)| idxs.len() >= state.cfg.min_cluster_size)
            .map(|(&label, idxs)| {
                // Medoid = point with the best mean similarity to its cluster.
                let (mut best_i, mut best_score) = (0usize, f64::NEG_INFINITY);
                for &ci in idxs.iter() {
                    let mut sum = 0.0f64;
                    for &cj in idxs.iter() {
                        let mut dot = 0f32;
                        for d in 0..dim {
                            dot += unk_embeds[ci][d] * unk_embeds[cj][d];
                        }
                        sum += dot as f64;
                    }
                    let mean = sum / idxs.len() as f64;
                    if mean > best_score {
                        best_score = mean;
                        best_i = ci;
                    }
                }
                // Power-weighted major angle.
                let mut angle_sum = 0f64;
                let mut w_sum = 0f64;
                for &ci in idxs.iter() {
                    let w = unk_embeds[ci].iter().sum::<f32>() as f64;
                    angle_sum += unk_doas[ci] * w.max(1.0);
                    w_sum += w.max(1.0);
                }
                let angle = if w_sum > 0.0 { angle_sum / w_sum } else { 0.0 };
                let mod_ = unk_mods[*idxs.first().unwrap_or(&0)]
                    .parse::<f64>()
                    .unwrap_or(0.0);
                // Mean centre frequency of this HDBSCAN cluster.
                let freq: f64 =
                    idxs.iter().map(|&ci| unk_freqs[ci]).sum::<f64>() / idxs.len() as f64;
                let ids: Vec<i64> = idxs.iter().map(|&ci| unk_ids[ci]).collect();
                ClusterStats {
                    label,
                    medoid: unk_embeds[best_i].clone(),
                    size: idxs.len(),
                    angle,
                    mod_,
                    freq,
                    ids,
                }
            })
            .collect();

        for s in stats {
            medoids.push(s.medoid);
            sizes.push(s.size);
            angles.push(s.angle);
            mods.push(s.mod_);
            freqs.push(s.freq);
            member_ids.push(s.ids);
        }
        drop(tracks_mut);

        // 5. Agglomerative merge + score gate → new emitters.
        let mt = merge_threshold(&state.cfg);
        let sg = score_gate(&state.cfg);
        let merged = cluster_merge::merge_clusters(
            &medoids,
            &sizes,
            &angles,
            &mods,
            &freqs,
            &member_ids,
            mt,
        );
        info!(
            "ai_processor: merged {} → {} clusters (threshold={})",
            medoids.len(),
            merged.len(),
            mt
        );

        for (mc, _members) in merged {
            if mc.score < sg {
                continue;
            }
            let freq = mc.mean_freq_hz;
            let emitter_id = match db::insert_emitter_pg(
                &state.pg,
                None,
                "unknown_ai",
                freq,
                0.0,
                mc.angle,
                0.0,
                "unknown",
                &mc.medoid,
            )
            .await
            {
                Ok(id) => id,
                Err(e) => {
                    warn!("ai_processor: insert_emitter failed: {e}");
                    continue;
                }
            };
            if let Err(e) = db::insert_cluster_ch(&state.ch, emitter_id, &mc, freq, 0.0).await {
                warn!("ai_processor: insert_cluster failed: {e}");
            }
            info!(
                "ai_processor: NEW EMITTER id={emitter_id} score={:.3} size={} angle={:.1}",
                mc.score, mc.size, mc.angle
            );
        }
    }

    let t_total = t0.elapsed();
    let sink_pg_ms = crate::db::pg_sink_ns().swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let sink_ch_ms = crate::db::ch_sink_ns().swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    info!(
        "TIMING batch_done total={:.1}ms sink_pg={:.1}ms sink_ch={:.1}ms ({} pulses, {} known, {} unknown)",
        t_total.as_secs_f64() * 1000.0,
        sink_pg_ms,
        sink_ch_ms,
        pulses.len(),
        known_count,
        unknown_idx.len()
    );
    Ok(())
}

/// Display name for a known-emitter track, derived from its modulation
/// field (Python writes the profile id, e.g. `PSR1` / `MF-03`).
pub fn track_name(modulation: &str) -> &str {
    let m = modulation.trim();
    if m.is_empty() || m.eq_ignore_ascii_case("unknown") || m.eq_ignore_ascii_case("known") {
        "unknown_ai"
    } else {
        m
    }
}
