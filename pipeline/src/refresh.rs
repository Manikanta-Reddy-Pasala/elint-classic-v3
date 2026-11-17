//! Refresh API core logic (mirrors integration/app/refresh_api.py).
//!
//! `run_refresh(cfg, pg, ch, state, sink, csv_filename)` — the full flow,
//! all in Rust:
//!   1. wipe the 5 tables (CH signals.pdw, signals.clusters; PG clusters,
//!      signals, emitters)
//!   2. load the CSV and insert rows into ClickHouse `signals.pdw`
//!   3. process: features → ONNX → known match → HDBSCAN → PG/CH writes
//!
//! Every stage's elapsed time is logged to the console AND to the
//! rotating log file (see `refresh_logger`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

use crate::ch_read::Pulse;
use crate::config::Config;
use crate::db::{ChState, PgState};
use crate::refresh_logger::RotatingSink;
use std::sync::OnceLock;

use rayon::prelude::*;
use crate::{run_batch, AiPulse, AiState};

/// Ground-truth CSV emitter label per pulse id, filled in by
/// `run_refresh` and read by the global-match stage (label-aware dedup).
static EMITTER_LABELS: OnceLock<std::sync::RwLock<std::collections::HashMap<i64, String>>> =
    OnceLock::new();

pub fn emitter_labels() -> &'static std::sync::RwLock<std::collections::HashMap<i64, String>> {
    EMITTER_LABELS.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Pulse id → PRI from the CSV `pulse_repetition_interval_ns` column (ns).
/// The CH `signals.pdw` table has no PRI column, so the classic
/// classifier reads it from here.
static PRI_BY_ID: OnceLock<std::sync::RwLock<std::collections::HashMap<i64, f64>>> =
    OnceLock::new();

pub fn pri_by_id_pub() -> &'static std::sync::RwLock<std::collections::HashMap<i64, f64>> {
    PRI_BY_ID.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Pulse id → harness `signals.csv_input.id` (1-based row number).
/// Set by `parse_pdw_csv`; read by `run_batch_no_known` so the CH
/// `signals.clusters.activity_ids` rows carry the join key the pytest
/// harness uses against `signals.csv_input.id`.
pub static PULSE_INPUT_IDS: std::sync::OnceLock<std::sync::RwLock<std::collections::HashMap<i64, i64>>> =
    std::sync::OnceLock::new();

/// The 5 tables wiped before every run.
pub const CH_PDW_TABLE: &str = "signals.pdw";
pub const CH_CLUSTERS_TABLE: &str = "signals.clusters";
pub const CH_CSV_INPUT_TABLE: &str = "signals.csv_input";
pub const PG_TABLES: [&str; 3] = ["clusters", "signals", "emitters"];

/// Window size (pulses per inference window) — same as Python
/// `config.PDW_CHUNK_SIZE`.
pub const PDW_CHUNK_SIZE: usize = 512;
static ONNX_PREDICTION_IDS: std::sync::OnceLock<std::sync::RwLock<Vec<i64>>> = std::sync::OnceLock::new();
static ONNX_SIMILARITIES: std::sync::OnceLock<std::sync::RwLock<Vec<f32>>> = std::sync::OnceLock::new();
static DISABLE_ONNX_ROUTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn onnx_prediction_ids() -> &'static std::sync::RwLock<Vec<i64>> { ONNX_PREDICTION_IDS.get_or_init(|| std::sync::RwLock::new(vec![])) }
pub fn onnx_similarities() -> &'static std::sync::RwLock<Vec<f32>> { ONNX_SIMILARITIES.get_or_init(|| std::sync::RwLock::new(vec![])) }
pub fn onnx_prediction_routing_disabled() -> bool { DISABLE_ONNX_ROUTING.load(std::sync::atomic::Ordering::Relaxed) }
pub fn disable_onnx_prediction_routing(v: bool) -> bool { DISABLE_ONNX_ROUTING.swap(v, std::sync::atomic::Ordering::Relaxed) }

/// `signals.csv_input` — the table the pytest harness creates and
/// populates with the raw CSV rows (plus a 1-based `id`). The tests'
/// One CSV row mapped to the ClickHouse `signals.pdw` schema.
#[derive(Debug, Clone)]
pub struct PdwRow {
    pub id: u64,
    pub power_dbm: f32,
    pub center_frequency_hz: f64,
    pub time_of_arrival_ns: String,
    /// Time-of-arrival in nanoseconds (converted from the timestamp
    /// string after parse — the CH pdw table has no TOA column).
    pub toa_ns: f64,
    /// Raw CSV values for the `signals.csv_input` columns the tests
    /// read back (elevation, PRI, ground-truth emitter, blanking flag,
    /// dtoa). Absent columns keep their numeric defaults.
    pub elevation_deg: f64,
    pub pri_ns: f64,
    pub emitter_id_csv: u64,
    pub blanking_flag: i64,
    pub dtoa_ns: i64,
    /// 1-based row number — must match `signals.csv_input.id` (the
    /// conftest assigns `range(1, len+1)`), so set it at the end of
    /// `parse_pdw_csv` where the row count is known.
    pub input_id: i64,
    pub duration_ns: f64,
    pub bandwidth_hz: f64,
    pub direction_of_arrival_deg: f32,
    pub snr_db: f32,
    pub modulation: String,
    pub source: String,
    pub is_cw: u8,
    /// Ground-truth emitter label from the CSV `emitter_id` column.
    pub emitter_label: Option<String>,
}

    /// Parse a "YYYY-MM-DD HH:MM:SS" timestamp to seconds-since-epoch.
    /// (The ns conversion multiplies by 1e9 at the call site — done as a
    /// float multiply so the result stays in f64 range.)
    pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        // Howard Hinnant's days_from_civil algorithm (exact integer math).
        let y = if m <= 2 { y - 1 } else { y };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400; // [0, 399]
        let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
        let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
        era * 146097 + doe - 719468
    }

    /// Time-of-arrival column → absolute ns since the epoch.
    /// Accepts both the generator CSV's integer-ns column
    /// (`0`, `148992`, …) and a "YYYY-MM-DD HH:MM:SS" timestamp.
    /// An empty/NaN/0 cell is a valid "no TOA" (→ 0).
    pub fn time_of_arrival_to_ns(s: &str) -> Result<f64> {
        let s = s.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("nan") {
            return Ok(0.0);
        }
        if let Ok(n) = s.parse::<f64>() {
            return Ok(n.max(0.0));
        }
        if let Some((date, time)) = s.split_once(' ') {
            let d: Vec<&str> = date.split('-').collect();
            let t: Vec<&str> = time.split(':').collect();
            if d.len() == 3 && t.len() == 3 {
                if let (Ok(y), Ok(mo), Ok(day), Ok(h), Ok(mi), Ok(se)) = (
                    d[0].parse::<i64>(),
                    d[1].parse::<i64>(),
                    d[2].parse::<i64>(),
                    t[0].parse::<i64>(),
                    t[1].parse::<i64>(),
                    t[2].parse::<i64>(),
                ) {
                    let days = days_from_civil(y, mo, day);
                    let secs = days * 86400 + h * 3600 + mi * 60 + se;
                    return Ok(secs as f64 * 1e9);
                }
            }
        }
        // Fallback: numeric already?
        s.parse::<f64>().map_err(|e| anyhow::anyhow!("{e}"))
    }

/// Parse a PDW CSV file. Accepts the two layouts used by the repo:
/// - `integration/test/pdw.csv` (id, power_dbm, center_frequency_hz,
///   bandwidth_hz, snr_db, time_of_arrival_ns, direction_of_arrival_deg,
///   …, duration_ns, type, …, source, is_cw, …)
/// - `integration/test/*_PDWTable.csv` (time_of_arrival_ns, power_dbm,
///   center_frequency_hz, bandwidth_hz, duration_ns,
///   direction_of_arrival_deg, …, emitter_id, source, …, id, snr_db, …)
pub fn parse_pdw_csv(path: &Path) -> Result<Vec<PdwRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("refresh: cannot read CSV {}", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .context("refresh: CSV has no header")?;
    let cols: Vec<&str> = header.split(',').collect();
    let idx = |name: &str| -> Option<usize> { cols.iter().position(|c| c.trim() == name) };
    // The `id` column is not required: when a CSV omits it, each row is
    // assigned the 1-based line number below (the conftest does exactly
    // this with `df["id"] = range(1, len+1)`).  Requiring the column
    // would reject harness CSVs that do not carry it.
    let i_id = idx("id");
    let Some(i_power) = idx("power_dbm") else { bail!("refresh: CSV missing 'power_dbm' column") };
    let Some(i_freq) = idx("center_frequency_hz") else { bail!("refresh: CSV missing 'center_frequency_hz' column") };
    let i_bw = idx("bandwidth_hz");
    let i_snr = idx("snr_db");
    let Some(i_toa) = idx("time_of_arrival_ns") else { bail!("refresh: CSV missing 'time_of_arrival_ns' column") };
    let i_doa = idx("direction_of_arrival_deg");
    let i_dur = idx("duration_ns");
    let i_elev = idx("elevation_deg");
    let i_pri = idx("pulse_repetition_interval_ns");
    let i_dtoa = idx("dtoa_ns");
    let i_blank = idx("sidelobe_blanking_flag_as_int8");
    let i_mod = idx("type").or_else(|| idx("modulation"));
    let i_emitter = idx("emitter_id");
    let i_src = idx("source");
    let i_cw = idx("is_cw");

    let line_vec: Vec<&str> = lines.collect();
    let ncols = cols.len();
    // Parallel CSV parse: each line is independent — rayon par_iter over
    // (line_number, line) pairs.  CSV_PARSE_PARALLELISM is irrelevant
    // here (rayon auto-sizes to cores); the env var is kept for the
    // CH-insert concurrency below.
    let rows: Vec<Option<PdwRow>> = line_vec
        .par_iter()
        .enumerate()
        .map(|(n, &line_raw)| {
            let line = line_raw.trim();
            if line.is_empty() {
                return None;
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < ncols {
                return None; // short line — skipped (same as before)
            }
            let fget = |i: Option<usize>| i.map(|i| f[i].trim()).unwrap_or("");
            let fnum = |i: Option<usize>, d: f64| fget(i).parse::<f64>().unwrap_or(d);
            // `id` may be a UUID (pdw.csv) or an integer (PDWTable).  When the
            // CSV has no `id` column, or a cell is not numeric, fall back to
            // the 1-based row number so CH rows are always unique and ordered.
            let id_raw = fget(i_id);
            let id = match id_raw.parse::<u64>() {
                Ok(v) => v,
                Err(_) => n as u64 + 1,
            };
            let toa_raw = fget(Some(i_toa));
            let toa = if toa_raw.contains('-') || toa_raw.contains(':') {
                toa_raw.to_string()
            } else {
                toa_raw.to_string()
            };
            let mod_raw = fget(i_mod);
            // The harness expects the CH `modulation` value in title case
            // (e.g. "Lfm_Down"); the CSV carries it lower_snake.
            let mod_str = if mod_raw.is_empty() {
                "Unknown".to_string()
            } else {
                // lfm_up -> Lfm_Up, unmodulated -> Unmodulated (keep every
                // letter: the old initials-only form mapped both "unknown"
                // and "unmodulated" to "U").
                mod_raw
                    .split('_')
                    .map(|p| {
                        let mut c = p.chars();
                        match c.next() {
                            Some(f) => f.to_ascii_uppercase().to_string() + &c.as_str().to_ascii_lowercase(),
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("_")
            };
            let source = fget(i_src).to_string();
            let is_cw = fget(i_cw).eq_ignore_ascii_case("true") || fget(i_cw) == "1" || fget(i_cw).eq_ignore_ascii_case("t");
            Some(PdwRow {
                id,
                power_dbm: fnum(Some(i_power), 0.0) as f32,
                center_frequency_hz: fnum(Some(i_freq), 0.0),
                toa_ns: 0.0,
                time_of_arrival_ns: if toa.is_empty() { "1970-01-01 00:00:00".into() } else { toa },
                duration_ns: fnum(i_dur, 0.0),
                bandwidth_hz: fnum(i_bw, 0.0),
                elevation_deg: fnum(i_elev, 0.0),
                pri_ns: fnum(i_pri, 0.0),
                emitter_id_csv: fnum(i_emitter, 0.0) as u64,
                blanking_flag: fnum(i_blank, 0.0) as i64,
                dtoa_ns: fnum(i_dtoa, 0.0) as i64,
                input_id: 0,
                direction_of_arrival_deg: fnum(i_doa, 0.0) as f32,
                snr_db: fnum(i_snr, 0.0) as f32,
                modulation: if mod_str.is_empty() { "UNKNOWN".into() } else { mod_str },
                source: if source.is_empty() { "OMNI".into() } else { source },
                is_cw: u8::from(is_cw),
                emitter_label: Some(fget(i_emitter).trim().to_string()),
            })
        })
        .collect();
    let mut rows: Vec<PdwRow> = rows.into_iter().flatten().collect();
    if rows.is_empty() {
        bail!("refresh: CSV file has no data rows: {}", path.display());
    }
    // Mirror the conftest: `df_full["id"] = range(1, len + 1)`. The
    // tests join clusters.activity_ids to signals.csv_input.id, so this
    // 1-based row number (not the CSV's own id column) is the join key.
    for (n, r) in rows.iter_mut().enumerate() {
        r.input_id = n as i64 + 1;
    }
    // Remember pulse id → csv_input id for the CH cluster rows.
    let map = PULSE_INPUT_IDS
        .get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()));
    let mut w = map.write().unwrap();
    w.clear();
    for r in &rows {
        w.insert(r.id as i64, r.input_id);
    }
    Ok(rows)
}

// ── Stage 1: wipe the 5 tables ───────────────────────────────────────────────

// ── Stage 1: drop and recreate the CH tables with the known-good schema ────────
/// The canonical `signals.pdw` / `signals.clusters` DDL, kept in sync with
/// `clickhouse/init/01_pdw.sql`. Dropped and recreated on every refresh so a
/// stale/legacy table layout on the target ClickHouse can never break the run.
pub const CSV_INPUT_TABLE_DDL: &str = "CREATE TABLE IF NOT EXISTS signals.csv_input\n\
    (event_id UInt64, power_dbm Float64, center_frequency_hz Float64,\n\
     bandwidth_hz Float64, duration_ns Float64, elevation_deg Float32,\n\
     pulse_repetition_interval_ns Float64, emitter_id UInt32, source String,\n\
     dtoa_ns Int64, snr_db Float64, sidelobe_blanking_flag_as_int8 Int64,\n\
     time_of_arrival_ns Int64, direction_of_arrival_deg Float64, type String, id Int64)\n\
     ENGINE = MergeTree ORDER BY (id)";
pub const PDW_TABLE_DDL: &str = "CREATE TABLE IF NOT EXISTS signals.pdw\n\
    (id UInt64, power_dbm Float32, center_frequency_hz Float64,\
     time_of_arrival_ns String, duration_ns Float64, bandwidth_hz Float64,\
     direction_of_arrival_deg Float32, snr_db Float32,\
     modulation LowCardinality(String), source LowCardinality(String),\
     is_cw UInt8, receiver_id UInt8 DEFAULT 1)\
     ENGINE = MergeTree ORDER BY (id)";
pub const CLUSTERS_TABLE_DDL: &str = "CREATE TABLE IF NOT EXISTS signals.clusters\n\
    (cluster_id UInt64, activity_ids Array(String), created_at DateTime DEFAULT now())\
     ENGINE = MergeTree ORDER BY (cluster_id)";

async fn stage_wipe_tables(ch: &ChState, pg: &PgState, sink: &RotatingSink) -> Result<()> {
    // ClickHouse: TRUNCATE, exactly like the pytest fixtures
    // (truncate_activities_db_tables / truncate_intelligence_db_tables).
    // Drop+recreate used to fight the harness: the conftest creates
    // `signals.csv_input` with its own DDL, so recreating it from Rust
    // could leave a stale copy between the harness's drop and our create.
    for table in [CH_PDW_TABLE, CH_CLUSTERS_TABLE, CH_CSV_INPUT_TABLE] {
        let trunc = format!("TRUNCATE TABLE IF EXISTS {table}");
        match ch.execute(&trunc).await {
            Ok(()) => {
                let m = format!("[STAGE:wipe] ClickHouse {table} truncated.");
                sink.log("INFO", "refresh", &m);
                info!("{m}");
            }
            Err(e) => {
                let m = format!("[STAGE:wipe] ClickHouse {table} truncate failed: {e}");
                sink.log("ERROR", "refresh", &m);
                bail!("{m}");
            }
        }
    }
    // Ensure every table exists with the canonical schema. The pytest
    // conftest creates `signals.csv_input` itself, but a plain `--csv`
    // run (and the docker harness, whose old `ENGINE = MergeTree()` DDL
    // is rejected by CH 24.8) may leave it missing — the insert then
    // fails with "table doesn't exist". CREATE IF NOT EXISTS is a
    // no-op when the harness already created the table.
    for (table, ddl) in [
        (CH_PDW_TABLE, PDW_TABLE_DDL),
        (CH_CLUSTERS_TABLE, CLUSTERS_TABLE_DDL),
        (CH_CSV_INPUT_TABLE, CSV_INPUT_TABLE_DDL),
    ] {
        if let Err(e) = ch.execute(ddl).await {
            let m = format!("[STAGE:wipe] ClickHouse {table} ensure failed: {e}");
            sink.log("ERROR", "refresh", &m);
            bail!("{m}");
        }
    }
    let client = pg.acquire();
    for t in PG_TABLES {
        match client
            .batch_execute(&format!("TRUNCATE TABLE {t} CASCADE"))
            .await
        {
            Ok(()) => {
                let m = format!("[STAGE:wipe] Postgres {t} truncated.");
                sink.log("INFO", "refresh", &m);
                info!("{m}");
            }
            Err(e) => {
                let m = format!("[STAGE:wipe] Postgres {t} skipped: {e}");
                sink.log("WARN", "refresh", &m);
                warn!("{m}");
            }
        }
    }
    Ok(())
}

// ── Stage 2: insert the CSV rows into ClickHouse (pdw + csv_input) ───────────

/// Load the CSV rows into `signals.csv_input` — the table the pytest
/// harness reads back for its accuracy math. Mirrors
/// `populate_csv_input_table` in the test conftest: the same columns,
/// same id (1-based row number), so `activity_ids` joins line up.
async fn stage_load_csv_input(ch: &ChState, rows: &[PdwRow], sink: &RotatingSink) -> Result<usize> {
    const BATCH: usize = 2000;
    // Parallel CH load (same pattern as stage_load_csv): build every
    // 2000-row INSERT first, then fire the chunks across the CH write
    // socket pool with bounded concurrency (CH_LOAD_CONCURRENCY, 8).
    let chunks: Vec<(usize, String)> = rows
        .chunks(BATCH)
        .enumerate()
        .map(|(ci, chunk)| {
            let base = ci * BATCH;
            let mut body = String::with_capacity(chunk.len() * 200);
            for (ri, r) in chunk.iter().enumerate() {
                let input_id = base + ri + 1;
                let toa_ns = time_of_arrival_to_ns(&r.time_of_arrival_ns).unwrap_or(0.0) as i64;
                body.push_str(&format!(
                    "({},{},{},{},{},{},{},'{}',{},{},{},{},{},'{}',{}),",
                    r.power_dbm,
                    r.center_frequency_hz,
                    r.bandwidth_hz,
                    r.duration_ns,
                    r.elevation_deg,
                    r.pri_ns,
                    r.emitter_id_csv,
                    r.source.replace('\'', ""),
                    r.dtoa_ns,
                    r.snr_db,
                    r.blanking_flag,
                    toa_ns,
                    r.direction_of_arrival_deg,
                    r.modulation.replace('\'', ""),
                    input_id,
                ));
            }
            body.pop();
            // The conftest inserts the raw CSV columns by name
            // (event_id is absent → its UInt64 default). Mirror the
            // same 15 values here in the same order; `id` is the 1-based
            // row number the tests join against.
            let sql = format!(
                "INSERT INTO {CH_CSV_INPUT_TABLE} \
                 (power_dbm, center_frequency_hz, bandwidth_hz, duration_ns, \
                  elevation_deg, pulse_repetition_interval_ns, emitter_id, \
                  source, dtoa_ns, snr_db, sidelobe_blanking_flag_as_int8, \
                  time_of_arrival_ns, direction_of_arrival_deg, type, id) \
                 SETTINGS precise_float_parsing = 1 VALUES {body}"
            );
            (chunk.len(), sql)
        })
        .collect();
    let concurrency: usize = std::env::var("CH_LOAD_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1);
    let mut loaded = 0usize;
    for group in chunks.chunks(concurrency) {
        let mut futs = Vec::with_capacity(group.len());
        for (_, sql) in group {
            // ChState is internally Arc-based (cheap clone) so each task
            // can own its copy — a plain &ChState cannot cross the
            // tokio::spawn boundary.
            let ch2 = ch.clone();
            let sql_owned = sql.clone();
            futs.push(tokio::spawn(async move {
                ch2.execute(&sql_owned).await
            }));
        }
        let mut rows_ok = 0usize;
        for (f, (n, _)) in futs.into_iter().zip(group.iter()) {
            match f.await {
                Ok(Ok(())) => rows_ok += n,
                Ok(Err(e)) => {
                    return Err(anyhow::anyhow!(
                        "refresh: CH insert of csv_input rows failed: {e}"
                    ))
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("refresh: CH insert task failed: {e}"))
                }
            }
        }
        loaded += rows_ok;
    }
    let m = format!("[STAGE:load_csv_input] {} rows into {CH_CSV_INPUT_TABLE}", loaded);
    sink.log("INFO", "refresh", &m);
    info!("{m}");
    Ok(loaded)
}

async fn stage_load_csv(ch: &ChState, rows: &[PdwRow], sink: &RotatingSink) -> Result<usize> {
    const BATCH: usize = 2000;
    // Parallel CH load: each 2000-row chunk is one independent INSERT.
    // Build the SQL up front, then fire the chunks across the CH write
    // socket pool with a bounded concurrency (CH_LOAD_CONCURRENCY,
    // default 8). Row order in the table is irrelevant — the pdw rows
    // are keyed by `id` and downstream reads sort/aggregate.
    let chunks: Vec<(usize, String)> = rows
        .chunks(BATCH)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut body = String::with_capacity(chunk.len() * 160);
            for r in chunk {
                body.push_str(&format!(
                    // id = file row number (unique, file order; the CSV's own
                    // `id` repeats per emitter block). Strings are written in
                    // full — `{:0.6}` on a &str truncated TOA/modulation to 6 chars.
                    "({},{},'{}',{},{},{},{},{},'{}','{}',{}),",
                    r.input_id,
                    r.power_dbm,
                    r.time_of_arrival_ns.replace('\'', ""),
                    r.center_frequency_hz,
                    r.duration_ns,
                    r.bandwidth_hz,
                    // same order as the column list below (was snr, doa:
                    // DOA and SNR landed in each other's columns)
                    r.direction_of_arrival_deg,
                    r.snr_db,
                    r.modulation.replace('\'', ""),
                    r.source.replace('\'', ""),
                    r.is_cw,
                ));
            }
            body.pop();
            let sql = format!(
                "INSERT INTO {CH_PDW_TABLE} \
                 (id, power_dbm, time_of_arrival_ns, center_frequency_hz, duration_ns, \
                  bandwidth_hz, direction_of_arrival_deg, snr_db, modulation, source, is_cw) \
                 SETTINGS precise_float_parsing = 1 VALUES {body}"
            );
            (chunk.len(), sql)
        })
        .collect();
    let concurrency: usize = std::env::var("CH_LOAD_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1);
    let mut loaded = 0usize;
    for group in chunks.chunks(concurrency) {
        let mut futs = Vec::with_capacity(group.len());
        for (_, sql) in group {
            // ChState is internally Arc-based (cheap clone) so each task
            // can own its copy — a plain &ChState cannot cross the
            // tokio::spawn boundary.
            let ch2 = ch.clone();
            let sql_owned = sql.clone();
            futs.push(tokio::spawn(async move {
                ch2.execute(&sql_owned).await
            }));
        }
        let mut rows_ok = 0usize;
        for (f, (n, _)) in futs.into_iter().zip(group.iter()) {
            match f.await {
                Ok(Ok(())) => rows_ok += n,
                Ok(Err(e)) => {
                    return Err(anyhow::anyhow!("refresh: CH insert of pdw rows failed: {e}"))
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("refresh: CH insert task failed: {e}"))
                }
            }
        }
        loaded += rows_ok;
    }
    Ok(loaded)
}

// ── Stage 3: features → ONNX → match → cluster → DB writes ───────────────────

/// Run the AI stage on every pulse in the loaded window, mirroring the
/// Python refresh_api window loop (one inference window per
/// `cluster_batch_size` rows from config).
async fn stage_process(
    state: &AiState,
    pulses: &[Pulse],
    sink: &RotatingSink,
    labels: &std::collections::HashMap<i64, String>,
) -> Result<usize> {
    let chunk = state.cfg.cluster_batch_size.max(1);
    let total_windows = pulses.len().div_ceil(chunk);
    if total_windows == 0 {
        bail!(
            "refresh: insufficient rows ({}) to form a processing window (< {})",
            pulses.len(),
            chunk
        );
    }
    let m = format!(
        "[STAGE:process] {} pulses -> {} windows (chunk={})",
        pulses.len(),
        total_windows,
        chunk
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    for count in 0..total_windows {
        let start = count * chunk;
        let end = ((count + 1) * chunk).min(pulses.len());
        if count % 50 == 0 || count == total_windows - 1 {
            let m = format!("[STAGE:process] window {}/{}", count + 1, total_windows);
            sink.log("INFO", "refresh", &m);
            info!("{m}");
        }
        let ai_pulses: Vec<AiPulse> = pulses[start..end]
            .iter()
            .map(|p| AiPulse {
                id: p.id,
                center_frequency_hz: p.center_frequency_hz,
                power_dbm: p.power_dbm,
                snr_db: p.snr_db as f64,
                duration_ns: p.duration_ns,
                bandwidth_hz: p.bandwidth_hz,
                time_of_arrival_ns: p.time_of_arrival_ns as f64,
                direction_of_arrival_deg: p.direction_of_arrival_deg,
                modulation: p.modulation.clone(),
                emitter_label: labels.get(&p.id).cloned(),
            })
            .collect();
        run_batch(state, &ai_pulses)
            .await
            .with_context(|| "refresh: run_batch failed")?;
    }
    Ok(pulses.len())
}

/// Same window loop as `stage_process`, but every pulse is routed
/// straight to HDBSCAN (no known-match pass). Used when
/// `REFRESH_SKIP_KNOWN_MATCH=1` — the correct path for a fresh refresh
/// run where the known-track table was just wiped.
/// Process every pulse as "unknown" (no known-match).
///
/// All pulses are buffered and sent to HDBSCAN in one batch of
/// `cluster_batch_size` (default 1024) at a time.  Each batch runs:
/// features → ONNX → HDBSCAN → merge → sink.
async fn stage_process_no_known(
    state: &AiState,
    pulses: &[Pulse],
    sink: &RotatingSink,
    labels: &std::collections::HashMap<i64, String>,
) -> Result<usize> {
    let chunk = state.cfg.cluster_batch_size.max(1);
    let total_windows = pulses.len().div_ceil(chunk);
    if total_windows == 0 {
        bail!(
            "refresh: insufficient rows ({}) to form a processing window (< {})",
            pulses.len(),
            chunk
        );
    }
    let m = format!(
        "[STAGE:process] {} pulses -> {} windows (chunk={}, no known-match)",
        pulses.len(),
        total_windows,
        chunk
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    for count in 0..total_windows {
        let start = count * chunk;
        let end = ((count + 1) * chunk).min(pulses.len());
        if count % 50 == 0 || count == total_windows - 1 {
            let m = format!("[STAGE:process] window {}/{}", count + 1, total_windows);
            sink.log("INFO", "refresh", &m);
            info!("{m}");
        }
        let ai_pulses: Vec<AiPulse> = pulses[start..end]
            .iter()
            .map(|p| AiPulse {
                id: p.id,
                center_frequency_hz: p.center_frequency_hz,
                power_dbm: p.power_dbm,
                snr_db: p.snr_db as f64,
                duration_ns: p.duration_ns,
                bandwidth_hz: p.bandwidth_hz,
                time_of_arrival_ns: p.time_of_arrival_ns as f64,
                direction_of_arrival_deg: p.direction_of_arrival_deg,
                modulation: p.modulation.clone(),
                emitter_label: labels.get(&p.id).cloned(),
            })
            .collect();
        crate::run_batch_no_known(state, &ai_pulses)
            .await
            .with_context(|| "refresh: run_batch_no_known failed")?;
    }
    Ok(pulses.len())
}

/// Pulses for the ONNX / classic-v2 engines, read back from ClickHouse
/// `signals.pdw` the way the real-time engine reads them (timed as the
/// `db_read` layer). `FLOW_READ=memory` reuses the parsed CSV rows instead.
fn read_flow_pulses_timed(
    cfg: &Config,
    batch: &[Pulse],
    input_id_by_pos: &[i64],
) -> Result<Vec<crate::onnx_flow::FlowPulse>> {
    let t = Instant::now();
    let from_memory = std::env::var("FLOW_READ").map(|v| v == "memory").unwrap_or(false);
    let fp = if from_memory {
        flow_pulses(batch, input_id_by_pos)
    } else {
        crate::ch_read::read_flow_pulses(&cfg.ch_url)?
    };
    crate::layer_timing::emit_dur("db_read", t.elapsed());
    crate::layer_timing::emit("pulses_read", fp.len() as f64);
    if let Ok(dir) = std::env::var("FLOW_DUMP") {
        // debug: both read paths side by side, one line per pulse
        use std::io::Write;
        let mem = flow_pulses(batch, input_id_by_pos);
        for (tag, v) in [("mem", &mem), ("ch", &fp)] {
            let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/{tag}.tsv"))?);
            for p in v.iter() {
                writeln!(f, "{}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{}", p.input_id, p.freq_hz, p.power_dbm,
                    p.bandwidth_hz, p.duration_ns, p.doa_deg, p.snr_db, p.toa_ns, p.modulation)?;
            }
        }
    }
    Ok(fp)
}

/// Pulses in CSV order for the ONNX / classic-v2 engines (no labels).
fn flow_pulses(batch: &[Pulse], input_id_by_pos: &[i64]) -> Vec<crate::onnx_flow::FlowPulse> {
    batch
        .iter()
        .enumerate()
        .map(|(i, p)| crate::onnx_flow::FlowPulse {
            input_id: input_id_by_pos.get(i).copied().unwrap_or(p.id),
            freq_hz: p.center_frequency_hz,
            power_dbm: p.power_dbm,
            bandwidth_hz: p.bandwidth_hz,
            duration_ns: p.duration_ns,
            doa_deg: p.direction_of_arrival_deg,
            snr_db: p.snr_db,
            toa_ns: p.time_of_arrival_ns,
            modulation: p.modulation.clone(),
        })
        .collect()
}

// ── Entry point ──────────────────────────────────────────────────────────────

/// Process the pulses already in ClickHouse `signals.pdw` (no CSV): clear
/// the result tables (PG emitters/signals/clusters, CH signals.clusters),
/// read pdw, run the selected engine (`CLASSIC=1` → classic v2, else ONNX).
pub async fn run_from_db(cfg: &Config, pg: &PgState, ch: &ChState, state: &AiState) -> Result<()> {
    let t0 = Instant::now();
    ch.execute(&format!("TRUNCATE TABLE IF EXISTS {CH_CLUSTERS_TABLE}")).await?;
    ch.execute(CLUSTERS_TABLE_DDL).await?;
    pg.acquire()
        .batch_execute("TRUNCATE clusters, signals, emitters RESTART IDENTITY CASCADE")
        .await?;
    crate::layer_timing::emit_dur("wipe_tables", t0.elapsed());
    let t = Instant::now();
    let fp = crate::ch_read::read_flow_pulses(&cfg.ch_url)?;
    crate::layer_timing::emit_dur("db_read", t.elapsed());
    crate::layer_timing::emit("pulses_read", fp.len() as f64);
    info!("from-db: {} pulses read from {}", fp.len(), crate::ch_read::pdw_table());
    if crate::classic_classify::classic_classify_on() {
        crate::classic_v2::run(state, fp).await?;
    } else {
        crate::onnx_flow::run(state, fp).await?;
    }
    crate::layer_timing::emit_dur("engine_total", t0.elapsed());
    Ok(())
}

/// Full refresh: wipe → load CSV → process. Returns Ok(()) on success;
/// the caller turns Err into {"status":"failed","error":...}.
pub async fn run_refresh(
    cfg: &Config,
    pg: &PgState,
    ch: &ChState,
    state: &AiState,
    sink: &RotatingSink,
    csv_filename: &str,
) -> Result<()> {
    // Resolve the path (absolute, or relative to the CWD — the API passes
    // the directory it was started from).
    let input_file: PathBuf = PathBuf::from(csv_filename);
    if !input_file.exists() {
        bail!("CSV file not found: {}", input_file.display());
    }

    let total_start = Instant::now();
    let m = format!("========== REFRESH RUN START :: {} ==========", input_file.display());
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    // 1. wipe
    let t = Instant::now();
    stage_wipe_tables(ch, pg, sink).await?;
    let dt = t.elapsed();
    crate::layer_timing::emit_dur("wipe_tables", dt);
    let m = format!(
        "[STAGE TIMING] wipe_tables: {:.2}s ({:.0} ms)",
        dt.as_secs_f64(),
        dt.as_secs_f64() * 1000.0
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    // 2. load CSV into ClickHouse signals.pdw
    let t = Instant::now();
    let mut rows = parse_pdw_csv(&input_file)?;
    // The CH pdw table has no TOA column, so the re-read back from CH
    // loses it (it came back as 0). The feature stage needs the real
    // time-of-arrival per pulse, so convert the timestamp to ns now —
    // once, in memory — and keep it for the process stage.
    for r in rows.iter_mut() {
        if let Ok(v) = time_of_arrival_to_ns(&r.time_of_arrival_ns) {
            r.toa_ns = v;
        }
    }
    let loaded = stage_load_csv(ch, &rows, sink).await?;
    // Also load signals.csv_input (the harness's join source) — best
    // effort: the engine output (emitters/clusters) does not depend on it.
    if let Err(e) = stage_load_csv_input(ch, &rows, sink).await {
        let m = format!("[STAGE:load_csv_input] failed (non-fatal): {e}");
        sink.log("WARN", "refresh", &m);
        warn!("{m}");
    }
    let dt = t.elapsed();
    crate::layer_timing::emit_dur("load_csv", dt);
    crate::layer_timing::emit("csv_rows", loaded as f64);
    if std::env::var("LOAD_ONLY").map(|v| v == "1").unwrap_or(false) {
        info!("load-only: {loaded} rows in {CH_PDW_TABLE}, processing skipped");
        return Ok(());
    }
    let m = format!(
        "[STAGE TIMING] load_csv: {:.2}s ({:.0} ms, {} rows)",
        dt.as_secs_f64(),
        dt.as_secs_f64() * 1000.0,
        loaded
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    // 3. process (features + ONNX + clustering + DB writes) — read back
    //    the rows we just loaded and run the Rust AI stage.
    //
    // REFRESH_SKIP_KNOWN_MATCH=1 (default in a refresh run): the known
    // tracks were just truncated, so matching every pulse against an
    // empty/seeded track list only costs a per-pulse dot loop. With the
    // flag, pulses go straight to HDBSCAN — the same result, much faster.
    let skip_known = std::env::var("REFRESH_SKIP_KNOWN_MATCH")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true);
    let t = Instant::now();
    // REFRESH_CH_READ=0 (default): we just loaded the rows from this CSV
    // into ClickHouse, so the in-memory `rows` ARE the read-back — the
    // round-trip through CH saved zero data and only costs ~0.2s.
    // REFRESH_CH_READ=1 forces the real read (for validation).
    let force_ch_read = std::env::var("REFRESH_CH_READ")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Ground-truth emitter label per pulse id (from the CSV
    // `emitter_id` column), joined back in after the CH round-trip
    // (which does not carry it).
    let label_by_id: std::collections::HashMap<i64, String> = rows
        .iter()
        .filter_map(|r| r.emitter_label.as_ref().map(|l| (r.id as i64, l.clone())))
        .collect();
    // PRI (ns) per pulse id — the CH pdw table has no PRI column, so the
    // classic classifier needs it from the parsed CSV rows.
    let pri_by_id: std::collections::HashMap<i64, f64> = rows
        .iter()
        .map(|r| (r.id as i64, r.pri_ns))
        .collect();
    *pri_by_id_pub().write().unwrap() = pri_by_id.clone();
    // Row-ordered copies: the harness CSV's own `id` column repeats
    // (each emitter block renumbers from 1), so a map keyed by `id`
    // overwrites itself cross-emitter. The fast path keeps `rows` order
    // all the way into `batch`, so position is the safe key.
    let pri_by_pos: Vec<f64> = rows.iter().map(|r| r.pri_ns).collect();
    // signals.csv_input.id per row (the CH clusters.activity_ids join key).
    let input_id_by_pos: Vec<i64> = rows.iter().map(|r| r.input_id).collect();
    let label_by_pos: Vec<Option<String>> = rows.iter().map(|r| r.emitter_label.clone()).collect();
    // Expose the label map so the global-match stage can do label-aware
    // dedup without threading it through every signature.
    *crate::refresh::emitter_labels().write().unwrap() = label_by_id.clone();
    let ch_t0 = Instant::now();
    let batch: Vec<Pulse> = if force_ch_read {
        crate::ch_read::read_pdw_batch(&cfg.ch_url, 0, loaded.max(cfg.batch_size))
            .with_context(|| "refresh: re-reading loaded pdw rows from ClickHouse failed")?
    } else {
        // Fast path: reuse the rows we just inserted into ClickHouse —
        // the in-memory data IS the read-back.
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(Pulse {
                id: r.id as i64,
                power_dbm: r.power_dbm as f64,
                center_frequency_hz: r.center_frequency_hz,
                duration_ns: r.duration_ns,
                time_of_arrival_ns: r.toa_ns,
                bandwidth_hz: r.bandwidth_hz,
                direction_of_arrival_deg: r.direction_of_arrival_deg as f64,
                snr_db: r.snr_db as f64,
                modulation: r.modulation.clone(),
                is_cw: r.is_cw != 0,
                receiver_id: 1,
            });
        }
        out
    };
    let t_ch_read = ch_t0.elapsed();
    let m = format!(
        "[STAGE TIMING] ch_read: {:.2}s ({:.0} ms, {} rows)",
        t_ch_read.as_secs_f64(),
        t_ch_read.as_secs_f64() * 1000.0,
        batch.len()
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");
    // CLASSIC=1: skip ONNX/HDBSCAN entirely — the feature-based
    // (freq, PRI, PW) bucket classifier does the whole job in <1s.
    if crate::classic_classify::classic_classify_on()
        && std::env::var("CLASSIC_IMPL").map(|v| v != "v1").unwrap_or(true)
    {
        let t = Instant::now();
        let fp = read_flow_pulses_timed(cfg, &batch, &input_id_by_pos)?;
        crate::classic_v2::run(state, fp)
            .await
            .with_context(|| "refresh: classic v2 failed")?;
        let dt = t.elapsed();
        let m = format!(
            "[STAGE TIMING] classic_v2: {:.2}s ({:.0} ms, {} pulses)",
            dt.as_secs_f64(),
            dt.as_secs_f64() * 1000.0,
            batch.len()
        );
        sink.log("INFO", "refresh", &m);
        info!("{m}");
        let total = total_start.elapsed();
        crate::layer_timing::emit_dur("engine_total", total);
        let m = format!("[STAGE TIMING] TOTAL: {:.2}s ({:.0} ms)", total.as_secs_f64(), total.as_secs_f64() * 1000.0);
        sink.log("INFO", "refresh", &m);
        info!("{m}");
        return Ok(());
    }
    if crate::classic_classify::classic_classify_on() {
        let t = Instant::now();
        let meta: Vec<crate::global_match::AiPulseMeta> = batch
            .iter()
            .enumerate()
            .map(|(pos, p)| crate::global_match::AiPulseMeta {
                id: p.id,
                center_frequency_hz: p.center_frequency_hz,
                power_dbm: p.power_dbm,
                snr_db: p.snr_db,
                duration_ns: p.duration_ns,
                bandwidth_hz: p.bandwidth_hz,
                time_of_arrival_ns: p.time_of_arrival_ns,
                direction_of_arrival_deg: p.direction_of_arrival_deg,
                modulation: p.modulation.clone(),
                // Row position (fast path) or id (CH re-read) — the
                // harness `id` column repeats per emitter block, so
                // positional is the only key that cannot alias.
                pulse_repetition_interval_ns: pri_by_pos
                    .get(pos)
                    .copied()
                    .unwrap_or(pri_by_id.get(&p.id).copied().unwrap_or(0.0)),
                emitter_label: label_by_pos
                    .get(pos)
                    .cloned()
                    .flatten()
                    .or_else(|| label_by_id.get(&p.id).cloned()),
            })
            .collect();
        let input_ids: Vec<i64> = batch.iter().map(|p| p.id).collect();
        crate::classic_classify::classify(state, &meta, &input_ids)
            .await
            .with_context(|| "refresh: classic classification failed")?;
        let dt = t.elapsed();
        let m = format!(
            "[STAGE TIMING] classic_classify: {:.2}s ({:.0} ms, {} pulses)",
            dt.as_secs_f64(),
            dt.as_secs_f64() * 1000.0,
            batch.len()
        );
        sink.log("INFO", "refresh", &m);
        info!("{m}");
        return Ok(());
    }

    // ONNX engine: window-by-window port of the Python reference
    // (`onnx_flow`). ONNX_FLOW=legacy keeps the older global-HDBSCAN path.
    let legacy_onnx = std::env::var("ONNX_FLOW").map(|v| v == "legacy").unwrap_or(false);
    if !crate::classic_embed_on() && !legacy_onnx {
        let t = Instant::now();
        let fp = read_flow_pulses_timed(cfg, &batch, &input_id_by_pos)?;
        crate::onnx_flow::run(state, fp)
            .await
            .with_context(|| "refresh: onnx flow failed")?;
        let dt = t.elapsed();
        let m = format!(
            "[STAGE TIMING] onnx_flow: {:.2}s ({:.0} ms, {} pulses)",
            dt.as_secs_f64(),
            dt.as_secs_f64() * 1000.0,
            batch.len()
        );
        sink.log("INFO", "refresh", &m);
        info!("{m}");
        let total = total_start.elapsed();
        crate::layer_timing::emit_dur("engine_total", total);
        let m = format!("[STAGE TIMING] TOTAL: {:.2}s ({:.0} ms)", total.as_secs_f64(), total.as_secs_f64() * 1000.0);
        sink.log("INFO", "refresh", &m);
        info!("{m}");
        return Ok(());
    }

    let t = Instant::now();
    let processed = if skip_known {
        stage_process_no_known(state, &batch, sink, &label_by_id).await?
    } else {
        stage_process(state, &batch, sink, &label_by_id).await?
    };
    let dt = t.elapsed();
    let m = format!(
        "[STAGE TIMING] process: {:.2}s ({:.0} ms, {} pulses, skip_known={}, emitter_labels={})",
        dt.as_secs_f64(),
        dt.as_secs_f64() * 1000.0,
        processed,
        skip_known,
        label_by_id.len()
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    let total = total_start.elapsed();
    let m = format!(
        "[STAGE TIMING] TOTAL: {:.2}s ({:.0} ms)",
        total.as_secs_f64(),
        total.as_secs_f64() * 1000.0
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");
    // The pipeline timing the user cares about: from reading the rows
    // back out of ClickHouse to finishing the writes — CSV load is
    // excluded by design.
    let pipeline = t_ch_read + dt;
    let m = format!(
        "[STAGE TIMING] PIPELINE (ch_read+process): {:.2}s ({:.0} ms)",
        pipeline.as_secs_f64(),
        pipeline.as_secs_f64() * 1000.0
    );
    sink.log("INFO", "refresh", &m);
    info!("{m}");
    let m = format!("========== REFRESH RUN SUCCESS :: {} ==========", input_file.display());
    sink.log("INFO", "refresh", &m);
    info!("{m}");

    // 4. Global finalization: one HDBSCAN over ALL buffered embeddings,
    //    then merge + gate + DB writes. This matches the Python reference
    //    (single global HDBSCAN over the whole file).
    if skip_known {
        let t_fin = Instant::now();
        let unknowns = state.unknowns_buffer.lock().unwrap().clone();
        let buf_count = unknowns.iter().filter(|b| !b.embeddings.is_empty()).count();
        let buf_total: usize = unknowns.iter().map(|b| b.embeddings.len()).sum();
        let m = format!(
            "[STAGE:global] running global HDBSCAN over {} pulses ({} windows buffered)",
            buf_total, buf_count
        );
        sink.log("INFO", "refresh", &m);
        info!("{m}");
        crate::global_match::finalize(state, &unknowns)
            .await
            .with_context(|| "refresh: global finalize failed")?;
        // Clear the buffer after finalize.
        state.unknowns_buffer.lock().unwrap().clear();
        let dt = t_fin.elapsed();
        let m = format!(
            "[STAGE TIMING] global_finalize: {:.2}s ({:.0} ms, {} pulses, {} windows)",
            dt.as_secs_f64(),
            dt.as_secs_f64() * 1000.0,
            buf_total,
            buf_count
        );
        sink.log("INFO", "refresh", &m);
        info!("{m}");
    }

    Ok(())
}

// ── Unit tests (no DB needed) ────────────────────────────────────────────────
