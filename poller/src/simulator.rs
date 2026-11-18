//! Data simulator — continuously inserts pulses into ClickHouse `signals.pdw`
//! via the HTTP interface, simulating a live capture feed.
//!
//! Used with `CH_CDC=live`: each worker batch is inserted into ClickHouse
//! AND published to Kafka in the same step, so the CDC poller can read the
//! rows back from the table while the feed is still being written.

use std::time::{Duration, Instant};

use anyhow::Result;

use std::io::Write;

use crate::config::Config;
use crate::pulse::{Modulation, Pulse};

/// Insert a batch of pulses into `signals.pdw` via ClickHouse HTTP.
///
/// Uses the same hand-rolled TCP HTTP POST as `ch_read::http_get` to avoid
/// pulling in `reqwest` (which needs edition-2024 crates on the 1.75 toolchain).
pub fn insert_batch_sync(cfg: &Config, pulses: &[Pulse]) -> Result<usize> {
    if pulses.is_empty() {
        return Ok(0);
    }

    let sql = build_insert_sql(pulses);
    let (host, port) = host_port(cfg);
    let url = "/?database=signals&insert=1".to_string();

    let start = Instant::now();
    let mut stream = std::net::TcpStream::connect((host.as_str(), port))
        .map_err(|e| anyhow::anyhow!("sim: tcp connect: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(Duration::from_secs(60)))?;

    let req = format!(
        "POST {url} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        sql.len(),
        sql
    );
    stream.write_all(req.as_bytes())
        .map_err(|e| anyhow::anyhow!("sim: write: {e}"))?;

    let mut resp = Vec::new();
    use std::io::Read;
    stream.read_to_end(&mut resp)?;
    let text = String::from_utf8_lossy(&resp).to_string();

    if !text.starts_with("HTTP/1.1 200") && !text.starts_with("HTTP/1.1 204") {
        anyhow::bail!("sim: CH insert failed: {}", &text[..text.len().min(300)]);
    }

    let elapsed = start.elapsed();
    if elapsed > Duration::from_millis(200) {
        log::debug!("sim: insert {} rows took {elapsed:?}", pulses.len());
    }
    Ok(pulses.len())
}

/// Build a multi-row INSERT SQL string.
pub fn build_insert_sql(pulses: &[Pulse]) -> String {
    let mut sql = String::with_capacity(pulses.len() * 96);
    sql.push_str(
        "INSERT INTO signals.pdw \
         (id, power_dbm, center_frequency_hz, time_of_arrival_ns, duration_ns, \
          bandwidth_hz, direction_of_arrival_deg, snr_db, modulation, is_cw, receiver_id) \
         VALUES ",
    );
    for (i, p) in pulses.iter().enumerate() {
        if i > 0 {
            sql.push(',');
        }
        sql.push_str(&format!(
            "({}, {:.2}, {:.1}, {}, {:.1}, {:.1}, {:.2}, {:.2}, '{}', {}, {})",
            p.id,
            p.power_dbm,
            p.center_frequency_hz,
            p.time_of_arrival_ns,
            p.duration_ns,
            p.bandwidth_hz,
            p.direction_of_arrival_deg,
            p.snr_db,
            p.modulation,
            p.is_cw,
            p.receiver_id,
        ));
    }
    sql
}

pub fn host_port(cfg: &Config) -> (String, u16) {
    let u = cfg
        .ch_url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let (host, port) = u.rsplit_once(':').unwrap_or((u, "8123"));
    let port = port.split('/').next().unwrap_or("8123");
    (host.to_string(), port.parse().unwrap_or(8123))
}


/// Simple shared counter for the live simulator.
pub struct SimCounters {
    inserted: std::sync::atomic::AtomicU64,
    published: std::sync::atomic::AtomicU64,
}

impl SimCounters {
    pub fn new() -> Self {
        Self { inserted: std::sync::atomic::AtomicU64::new(0), published: std::sync::atomic::AtomicU64::new(0) }
    }
    pub fn add_inserted(&self, n: u64) { self.inserted.fetch_add(n, std::sync::atomic::Ordering::Relaxed); }
    pub fn add_published(&self, n: u64) { self.published.fetch_add(n, std::sync::atomic::Ordering::Relaxed); }
    pub fn summary(&self) -> (u64, u64) {
        (self.inserted.load(std::sync::atomic::Ordering::Relaxed), self.published.load(std::sync::atomic::Ordering::Relaxed))
    }
}
