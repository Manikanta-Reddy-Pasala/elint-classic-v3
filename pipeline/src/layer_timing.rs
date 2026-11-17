//! Per-layer timing lines for reports: `LAYER|<stage>|<value>` on stdout.
//! Values are milliseconds unless the stage name says otherwise
//! (`windows`, `pulses`). One line per stage per run; the report builder
//! (`ai_elint_tests/scripts/report/build_report.py`) parses them.

use std::sync::atomic::{AtomicU64, Ordering};

static DB_NS: AtomicU64 = AtomicU64::new(0);

pub fn emit(stage: &str, value: f64) {
    println!("LAYER|{stage}|{value:.2}");
}

pub fn emit_dur(stage: &str, d: std::time::Duration) {
    emit(stage, d.as_secs_f64() * 1e3);
}

/// Time spent in PG/CH writes by the ONNX / classic-v2 writers.
pub fn add_db(d: std::time::Duration) {
    DB_NS.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}

pub fn take_db_ms() -> f64 {
    DB_NS.swap(0, Ordering::Relaxed) as f64 / 1e6
}

static PG_CLU_NS: AtomicU64 = AtomicU64::new(0);
static CH_CLU_NS: AtomicU64 = AtomicU64::new(0);

/// Split of the DB-writes layer: PG `clusters` insert vs CH `signals.clusters` insert.
pub fn add_pg_clusters(d: std::time::Duration) {
    PG_CLU_NS.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}
pub fn add_ch_clusters(d: std::time::Duration) {
    CH_CLU_NS.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
}
/// Emit `<prefix>.db_writes` plus its PG/CH cluster-insert split.
pub fn emit_db(prefix: &str) {
    emit(&format!("{prefix}.db_writes"), take_db_ms());
    emit(&format!("{prefix}.db_pg_clusters"), PG_CLU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6);
    emit(&format!("{prefix}.db_ch_clusters"), CH_CLU_NS.swap(0, Ordering::Relaxed) as f64 / 1e6);
}
