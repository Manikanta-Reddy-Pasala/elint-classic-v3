//! Standalone binary — two modes:
//!
//! 1. **Polling** (default): polls ClickHouse `signals.pdw` and runs the
//!    AI stage via the `ai_processor` library forever.
//!
//! 2. **One-shot** (`--csv <path>`): wipe → load CSV into ClickHouse →
//!    process → global finalize → exit. This is the mode the Python test
//!    harness uses as a drop-in replacement for `python app.py`
//!    (one run of `run_ai_model(csv_file_path)` per test).
//!
//! Run:
//!   cargo run --bin ai_processor -- --csv path/to/pdw.csv

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{info, warn};

use ai_processor::{
    ch_read::read_pdw_batch, config::Config, refresh, refresh_logger::RotatingSink, AiPulse,
    AiState,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let csv = args
        .iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1))
        .cloned();
    if args.iter().any(|a| a == "--load-only") {
        // wipe + load the CSV into ClickHouse, no processing
        std::env::set_var("LOAD_ONLY", "1");
    }
    if args.iter().any(|a| a == "--from-db") {
        return run_from_db().await;
    }
    match csv {
        Some(path) => run_one_shot(&path).await,
        None => run_poller().await,
    }
}

/// `--from-db`: process the pulses already in ClickHouse `signals.pdw`.
async fn run_from_db() -> Result<()> {
    let cfg = Config::from_env();
    let state = AiState::new(cfg.clone()).await?;
    let t0 = Instant::now();
    refresh::run_from_db(&cfg, &state.pg, &state.ch, &state).await?;
    info!(target: "ai_processor", "ai_processor: from-db run finished in {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// One-shot run: wipe → load CSV → process → finalize, then exit.
/// Returns Ok(()) only when the full run succeeded, so the caller
/// (test harness) can rely on the exit code.
async fn run_one_shot(csv_path: &str) -> Result<()> {
    let log_dir = std::env::var("ELINT_LOG_DIR").unwrap_or_else(|_| "logs".into());
    let sink = RotatingSink::new(&log_dir)
        .with_context(|| format!("ai_processor: cannot init rotating log in {log_dir}"))?;
    info!(target: "ai_processor", "ai_processor: one-shot mode, logging to {log_dir}");

    let cfg = Config::from_env();
    info!(target: "ai_processor", "ai_processor: loading ONNX model: {}", cfg.onnx_path);
    let state = AiState::new(cfg.clone()).await?;

    let t0 = Instant::now();
    let result =
        refresh::run_refresh(&cfg, &state.pg, &state.ch, &state, &sink, csv_path).await;

    match &result {
        Ok(()) => info!(
            target: "ai_processor",
            "ai_processor: one-shot run finished in {:.2}s",
            t0.elapsed().as_secs_f64()
        ),
        Err(e) => {
            let msg = format!("ai_processor: one-shot run failed: {e:#}");
            sink.log("ERROR", "ai_processor", &msg);
            warn!("{msg}");
        }
    }
    result.map_err(|e| e.context("ai_processor: one-shot refresh failed"))
}

/// Polling loop: read new rows from ClickHouse and run the AI stage.
async fn run_poller() -> Result<()> {
    let cfg = Config::from_env();
    let state = AiState::new(cfg.clone()).await?;

    let (ch_host, ch_port) = ai_processor::ch_read::host_port(&cfg.ch_url);
    let scheme = if cfg.ch_url.starts_with("https:") { "https" } else { "http" };
    let ch_http = format!("{scheme}://{ch_host}:{ch_port}");

    let mut cursor: i64 = 0;
    let mut idle = 0u64;
    loop {
        let batch = match read_pdw_batch(&ch_http, cursor, cfg.batch_size) {
            Ok(b) => b,
            Err(e) => {
                warn!("ai-processor: pdw read failed: {e}; backing off 500ms");
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        };
        if batch.is_empty() {
            idle += 1;
            if idle % 500 == 1 {
                info!("ai-processor: idle (cycle {idle}, waiting for PDW > id {cursor})");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        }
        idle = 0;
        cursor = batch.iter().map(|p| p.id).max().unwrap_or(cursor);

        let ai_pulses: Vec<AiPulse> = batch
            .iter()
            .map(|p| AiPulse {
                id: p.id,
                center_frequency_hz: p.center_frequency_hz,
                power_dbm: p.power_dbm,
                snr_db: p.snr_db,
                duration_ns: p.duration_ns,
                bandwidth_hz: p.bandwidth_hz,
                time_of_arrival_ns: p.time_of_arrival_ns as f64,
                direction_of_arrival_deg: p.direction_of_arrival_deg,
                modulation: p.modulation.clone(),
                emitter_label: None,
            })
            .collect();
        ai_processor::run_batch(&state, &ai_pulses).await?;
    }
}
