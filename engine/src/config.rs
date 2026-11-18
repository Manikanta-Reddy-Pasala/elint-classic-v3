//! Engine configuration, all env-driven.

use std::env;

use crate::band::{specs_default, specs_from_env};
use crate::ch_read::fetch_limit;

#[derive(Debug, Clone)]
pub struct Config {
    /// ClickHouse base URL, e.g. `http://127.0.0.1:8123`.
    pub ch_url: String,
    /// ClickHouse database name.
    pub ch_db: String,
    /// Postgres DSN.
    pub pg_url: String,
    /// ONNX model path.
    pub model_path: String,

    /// Number of parallel ClickHouse read tasks.
    pub read_threads: usize,
    /// Number of worker threads that do band/AI/clustering.
    pub process_workers: usize,
    /// Rows per CDC read.
    pub fetch_limit: u32,

    /// Band specs (from ELINT_BANDS env, default S/L/X).
    pub bands: Vec<crate::band::BandSpec>,
    /// Known-emitter confidence threshold.
    pub known_threshold: f32,
}

/// ClickHouse URL — accept both `CLICKHOUSE_URL` and `CH_HTTP_URL`.
fn ch_url_from_env() -> String {
    if let Ok(v) = env::var("CLICKHOUSE_URL") {
        return v;
    }
    env::var("CH_HTTP_URL").unwrap_or_else(|_| "http://127.0.0.1:8123".into())
}

impl Config {
    pub fn from_env() -> Self {
        Config {
            ch_url: ch_url_from_env(),
            ch_db: env::var("CLICKHOUSE_DB").unwrap_or_else(|_| "signals".into()),
            pg_url: env::var("PG_URL")
                .or_else(|_| env::var("POSTGRES_URL"))
                .unwrap_or_else(|_| "postgresql://elint:elint@127.0.0.1:5432/sigint".into()),
            model_path: env::var("MODEL_PATH")
                .unwrap_or_else(|_| "/models/transformer_model_known_v16.onnx".into()),
            read_threads: env::var("READ_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            process_workers: env::var("PROCESS_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            fetch_limit: fetch_limit(),
            bands: specs_from_env(),
            known_threshold: env::var("KNOWN_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.8),
        }
    }

/// Postgres DSN — accept both `postgres://` and `postgresql://`.
    pub fn pg_dsn(&self) -> String {
        let u = self.pg_url.clone();
        if u.starts_with("postgres://") {
            u
        } else {
            format!("postgres://{}", u.trim_start_matches("postgresql://"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            ch_url: "http://127.0.0.1:8123".into(),
            ch_db: "signals".into(),
            pg_url: "postgres://elint:elint@127.0.0.1:5432/sigint".into(),
            model_path: "/models/model.onnx".into(),
            read_threads: 4,
            process_workers: 4,
            fetch_limit: 1000,
            bands: specs_default(),
            known_threshold: 0.8,
        }
    }

    #[test]
    fn pg_dsn_keeps_postgres_scheme() {
        let c = test_config();
        assert_eq!(c.pg_dsn(), "postgres://elint:elint@127.0.0.1:5432/sigint");
    }

    #[test]
    fn pg_dsn_converts_postgresql_scheme() {
        let mut c = test_config();
        c.pg_url = "postgresql://elint:elint@127.0.0.1:5432/sigint".into();
        assert_eq!(c.pg_dsn(), "postgres://elint:elint@127.0.0.1:5432/sigint");
    }

    #[test]
    fn config_fields_are_set() {
        let c = test_config();
        assert_eq!(c.ch_url, "http://127.0.0.1:8123");
        assert_eq!(c.ch_db, "signals");
        assert_eq!(c.read_threads, 4);
        assert_eq!(c.process_workers, 4);
        assert_eq!(c.fetch_limit, 1000);
        assert_eq!(c.known_threshold, 0.8);
        assert_eq!(c.bands.len(), 3);
    }
}
