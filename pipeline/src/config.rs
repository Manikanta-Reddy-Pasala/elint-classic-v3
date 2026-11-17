use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub onnx_path: String,
    pub onnx_session_count: usize,
    pub onnx_threads: usize,
    pub known_threshold: f32,
    pub angle_gate: f32,
    pub ema_alpha: f32,
    pub batch_size: usize,
    pub min_cluster_size: usize,
    pub max_samples: usize,
    /// Pulses per ONNX inference sub-batch (default 512).
    pub onnx_batch_size: usize,
    /// Pulses buffered before one HDBSCAN clustering pass (default 1024).
    pub cluster_batch_size: usize,
    pub pg_url: String,
    pub ch_url: String,
}

impl Config {
    pub fn from_env() -> Self {
        Config {
            onnx_path: env::var("ONNX_PATH")
                .unwrap_or_else(|_| "/models/transformer_model_known_v16.onnx".into()),
            onnx_session_count: env::var("ONNX_SESSION_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            onnx_threads: env::var("ONNX_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6),
            known_threshold: env::var("KNOWN_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.80),
            angle_gate: env::var("ANGLE_GATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(40.0),
            ema_alpha: env::var("EMA_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.1),
            batch_size: env::var("BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2000),
            min_cluster_size: env::var("MIN_CLUSTER_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            max_samples: env::var("MAX_SAMPLES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0), // 0 = use all rows (matching Python HDBSCAN default)
            onnx_batch_size: env::var("ONNX_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            cluster_batch_size: env::var("CLUSTER_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024),
            pg_url: env::var("PG_URL")
                .unwrap_or_else(|_| "postgresql://elint:elint@127.0.0.1:5432/sigint".into()),
            ch_url: env::var("CH_URL").unwrap_or_else(|_| "http://127.0.0.1:8123".into()),
        }
    }
}

// ── Unit tests (no DB / model needed) ────────────────────────────────────────
