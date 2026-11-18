//! Environment-driven configuration for the CDC poller.

use std::env;

pub const TOPIC_PDW: &str = "elint.pdw";
pub const TOPIC_DONE: &str = "elint.pdw.done";

/// Read mode for the poller.
/// - `Cdc`: read rows from ClickHouse `signals.pdw` in batches (default).
/// - `Generate`: synthesize pulses (legacy mode, CH_CDC=0).
/// - `Live`: data simulator — insert pulses into ClickHouse AND publish to
///   Kafka in the same run, simulating a continuous capture feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    Cdc,
    Generate,
    Live,
}

#[derive(Clone)]
pub struct Config {
    pub brokers: String,
    pub pulse_count: u64,
    pub read_mode: ReadMode,
    pub ch_url: String,
    /// Rows per ClickHouse SELECT batch.
    pub fetch_limit: u64,
    /// Number of parallel insert tasks in live-simulator mode.
    pub num_sim_tasks: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            brokers: "127.0.0.1:9092".into(),
            pulse_count: 50_000,
            read_mode: ReadMode::Cdc,
            ch_url: "http://127.0.0.1:8123".into(),
            fetch_limit: 2_000,
            num_sim_tasks: 4,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let read_mode = match env::var("CH_CDC").as_deref().unwrap_or("1") {
            "0" | "false" | "generate" => ReadMode::Generate,
            "live" | "sim" => ReadMode::Live,
            _ => ReadMode::Cdc,
        };
        Self {
            brokers: env::var("KAFKA_BROKERS")
                .unwrap_or_else(|_| "127.0.0.1:9092".into()),
            pulse_count: env::var("PULSE_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000),
            read_mode,
            ch_url: env::var("CLICKHOUSE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8123".into()),
            fetch_limit: env::var("FETCH_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_000),
            num_sim_tasks: env::var("SIM_TASKS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
        }
    }
}

/// A lock that serializes tests that mutate the process environment.
/// `cargo test` runs tests on parallel threads; env changes from one
/// test would otherwise leak into another.
pub struct EnvLock(std::sync::Mutex<()>);

impl EnvLock {
    pub const fn new() -> Self {
        EnvLock(std::sync::Mutex::new(()))
    }
    pub fn guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Serialize env-dependent tests (see `EnvLock`).
pub static ENV_LOCK: EnvLock = EnvLock::new();
