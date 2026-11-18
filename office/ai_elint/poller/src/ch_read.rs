//! ClickHouse CDC reader: SELECT batches from signals.pdw, ordered by id,
//! starting from the last published id (resume support). No re-scan of old rows.
//!
//! Talks to the ClickHouse HTTP interface (port 8123) with plain std TCP +
//! a hand-rolled HTTP/1.1 GET — no reqwest, no rustls, no h2 (those pull
//! transitive deps that need edition-2024 crates on the 1.75 toolchain).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use log::{info, warn};

use crate::config::Config;
use crate::pulse::{Modulation, Pulse};

pub const QUERY: &str = "SELECT id, power_dbm, center_frequency_hz, time_of_arrival_ns, \
                     duration_ns, bandwidth_hz, direction_of_arrival_deg, snr_db, \
                     modulation, is_cw, receiver_id, source \
                     FROM signals.pdw WHERE id > {start} ORDER BY id LIMIT {limit}";

pub fn host_port(cfg: &Config) -> (String, u16) {
    let u = if cfg.ch_url.contains("://") {
        cfg.ch_url.trim_start_matches("http://").trim_start_matches("https://")
    } else {
        cfg.ch_url.as_str()
    };
    let (host, port) = u.rsplit_once(':').unwrap_or((u, "8123"));
    let port = port.split('/').next().unwrap_or("8123");
    (host.to_string(), port.parse().unwrap_or(8123))
}

/// Minimal HTTP/1.1 GET. Returns the decoded response body.
/// `format` is the ClickHouse output format (sent via the
/// X-ClickHouse-Format header, never the query string — Code 115).
pub fn http_get(cfg: &Config, query: &str, format: &str) -> Result<String> {
    let (host, port) = host_port(cfg);
    let url = format!("/?query={}", urlencode(query));
    let mut stream = TcpStream::connect((host.as_str(), port)).context("ch: tcp connect")?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    // CH rejects `format=` in the query string (Code 115); the
    // X-ClickHouse-Format header is the supported way to pick the output
    // format.
    let req = format!(
        "GET {url} HTTP/1.1\r\nHost: {host}:{port}\r\nX-ClickHouse-Format: {format}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).context("ch: send request")?;

    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .context("ch: read response")?;
    let text = String::from_utf8_lossy(&buf).to_string();
    // split headers / body on first blank line
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b).context("ch: bad http response")?;
    if text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.1 204") {
        // The body is chunked-transfer-encoded (Transfer-Encoding: chunked).
        // Decode the chunks before handing rows to the caller.
        Ok(decode_chunked(body).unwrap_or_else(|| body.to_string()))
    } else {
        bail!("ch: non-200 response: {}", &text[..text.len().min(300)])
    }
}

/// Decode an HTTP chunked body (the part of the response after the header
/// block). Returns None if the body is not a well-formed chunked stream, in
/// which case the caller falls back to the raw text.
pub fn decode_chunked(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut rest = raw;
    loop {
        let (size_line, r2) = rest.split_once('\n')?;
        let size = usize::from_str_radix(size_line.trim(), 16).ok()?;
        rest = r2;
        if size == 0 {
            break;
        }
        let chunk = &rest[..size];
        out.push_str(chunk);
        rest = rest[size..].trim_start_matches("\r\n");
    }
    Some(out)
}

/// URL-encode for the query string (only what CH query strings need).
pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push_str("%20"),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// Map a ClickHouse modulation string onto the compact enum.
pub fn parse_modulation(s: &str) -> Modulation {
    match s.to_ascii_lowercase().as_str() {
        "psk" => Modulation::Psk,
        "fsk" => Modulation::Fsk,
        "am" => Modulation::Am,
        "pm" => Modulation::Pm,
        "cw" => Modulation::Cw,
        "qam" => Modulation::Qam,
        _ => Modulation::Unknown,
    }
}

/// Pull up to `limit` PDW rows with `id > start_id` from ClickHouse.
/// Returns (rows, max_id_seen) so the caller can advance the cursor.
/// ClickHouse returns rows as tab-separated text (TabSeparatedWithNamesAndTypes).
pub fn read_batch(cfg: &Config, start_id: u64) -> Result<(Vec<Pulse>, u64)> {
    let sql = QUERY
        .replace("{start}", &start_id.to_string())
        .replace("{limit}", &cfg.fetch_limit.to_string());
    info!("ch: reading batch after id={start_id}, limit={}", cfg.fetch_limit);

    let body = http_get(cfg, &sql, "TabSeparatedWithNamesAndTypes")
        .context("clickhouse: query failed")?;
    let mut lines = body.lines();
    // header: names line, then types line — skip both
    let _names = lines.next().context("ch: empty response")?;
    let _types = lines.next().context("ch: missing types line")?;

    let mut rows = Vec::with_capacity(cfg.fetch_limit as usize);
    let mut max_id = start_id;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 12 {
            warn!("ch: skipping malformed row ({} cols): {line}", f.len());
            continue;
        }
        let id: u64 = f[0].parse().context("ch: bad id")?;
        max_id = max_id.max(id);
        rows.push(Pulse {
            id,
            power_dbm: f[1].parse().unwrap_or(0.0),
            center_frequency_hz: f[2].parse().unwrap_or(0.0),
            time_of_arrival_ns: f[3].parse().unwrap_or(0),
            duration_ns: f[4].parse().unwrap_or(0.0),
            bandwidth_hz: f[5].parse().unwrap_or(0.0),
            direction_of_arrival_deg: f[6].parse().unwrap_or(0.0),
            snr_db: f[7].parse().unwrap_or(0.0),
            modulation: parse_modulation(f[8]),
            is_cw: f[9].parse().unwrap_or(0),
            receiver_id: f[10].parse().unwrap_or(1),
        });
    }

    if rows.is_empty() {
        bail!("ch: no rows after id={start_id} — is signals.pdw populated?");
    }
    info!("ch: read {} rows, max_id={max_id}", rows.len());
    Ok((rows, max_id))
}

#[allow(dead_code)]
pub fn table_count(cfg: &Config) -> Option<u64> {
    let body = http_get(cfg, "SELECT count() FROM signals.pdw", "TabSeparated").ok()?;
    body.trim().parse().ok()
}
