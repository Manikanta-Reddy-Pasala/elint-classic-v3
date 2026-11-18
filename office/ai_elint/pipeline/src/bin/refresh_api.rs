//! Refresh API server (Rust, no Python, no external HTTP crate).
//!
//! Endpoints (raw HTTP/1.1 over TCP):
//!
//!   POST /api/refresh-data   {"csv_filename": "test/pdw.csv"}
//!       → {"status":"success"}  or  {"status":"failed","error":"..."}
//!   GET  /health             → {"status":"ok","ready":true}
//!
//! Run:
//!   cargo run --bin refresh_api

use std::net::TcpListener;
use std::sync::Arc;

use anyhow::Result;
use tracing::info;

use ai_processor::config::Config;
use ai_processor::refresh_api_core::{ApiState, handle_conn};
use ai_processor::{refresh_logger, AiState};

#[tokio::main]
async fn main() -> Result<()> {
    let log_dir = std::env::var("ELINT_LOG_DIR").unwrap_or_else(|_| "logs".into());
    let sink = refresh_logger::init(&log_dir)
        .unwrap_or_else(|| panic!("refresh_api: cannot init rotating log in {log_dir}"));
    info!(target: "refresh_api", "refresh_api: logging to console + {} (rotating)", log_dir);

    let port: u16 = std::env::var("REFRESH_API_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8788);

    let cfg = Config::from_env();
    info!(target: "refresh_api", "[API] Loading ONNX model: {}", cfg.onnx_path);
    let state = AiState::new(cfg.clone()).await?;
    let api = Arc::new(ApiState {
        cfg,
        state: Arc::new(state),
        sink,
    });
    info!(target: "refresh_api", "[API] Startup complete: model + ClickHouse + Postgres ready.");

    let listener = TcpListener::bind(("0.0.0.0", port))?;
    info!(target: "refresh_api", "refresh_api: listening on 0.0.0.0:{port}");

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let api = api.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, api) {
                tracing::warn!(target: "refresh_api", "refresh_api: conn error: {e}");
            }
        });
    }
    Ok(())
}
