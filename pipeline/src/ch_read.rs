/// CDC read — direct read from ClickHouse `signals.pdw` via HTTP.
///
/// Hand-rolled TCP + HTTP/1.1, no reqwest.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use tracing::warn;

/// One PDW pulse row from ClickHouse.
#[derive(Debug, Clone)]
pub struct Pulse {
    pub id: i64,
    pub power_dbm: f64,
    pub center_frequency_hz: f64,
    /// Time-of-arrival in nanoseconds (the CH pdw table has no TOA
    /// column; the refresh path fills this from the CSV timestamp).
    pub time_of_arrival_ns: f64,
    pub duration_ns: f64,
    pub bandwidth_hz: f64,
    pub direction_of_arrival_deg: f64,
    pub snr_db: f64,
    pub modulation: String,
    pub is_cw: bool,
}

pub const QUERY: &str = "SELECT id, power_dbm, center_frequency_hz, time_of_arrival_ns, \
                     duration_ns, bandwidth_hz, direction_of_arrival_deg, snr_db, \
                     modulation, is_cw \
                     FROM signals.pdw WHERE id > {start} ORDER BY id LIMIT {limit}";

pub fn host_port(ch_url: &str) -> (String, u16) {
    let u = ch_url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/');
    match u.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(8123)),
        None => (u.to_string(), 8123),
    }
}

// ── Unit tests (no DB needed) ────────────────────────────────────────────────
pub fn http_get(host: &str, port: u16, query: &str, format: &str) -> Result<String> {
    let url = format!("/?{}query={}", auth_query(None), urlencode(query));
    let mut stream = TcpStream::connect((host, port)).context("ch: tcp connect")?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let req = format!(
        "GET {url} HTTP/1.1\r\nHost: {host}:{port}\r\nX-ClickHouse-Format: {format}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).context("ch: send request")?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).context("ch: read response")?;
    let text = String::from_utf8_lossy(&buf).to_string();
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .context("ch: bad http response")?;
    if text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.1 204") {
        Ok(decode_chunked(body).unwrap_or_else(|| body.to_string()))
    } else {
        bail!("ch: non-200 response: {}", &text[..text.len().min(300)])
    }
}

pub fn ch_auth() -> (String, String) {
    (
        std::env::var("CLICKHOUSE_USER")
            .or_else(|_| std::env::var("CLICKHOUSE_DB_USER"))
            .unwrap_or_default(),
        std::env::var("CLICKHOUSE_PASSWORD")
            .or_else(|_| std::env::var("CLICKHOUSE_DB_PASS"))
            .unwrap_or_default(),
    )
}

pub fn auth_query(database: Option<&str>) -> String {
    let (u, p) = ch_auth();
    let auth = match (u.is_empty(), p.is_empty()) {
        (true, true) => String::new(),
        _ => format!("user={}&password={}", urlencode(&u), urlencode(&p)),
    };
    match (auth.is_empty(), database.map(|d| !d.is_empty()).unwrap_or(false)) {
        (true, false) => String::new(),
        (true, true) => format!("database={}", urlencode(database.unwrap())),
        (false, false) => format!("{auth}&"),
        (false, true) => format!("{auth}&database={}", urlencode(database.unwrap())),
    }
}

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

/// PDW table to read (`PDW_TABLE`, default `signals.pdw`) — lets benchmarks
/// point the poller at a fixed-size copy without touching the live table.
pub fn pdw_table() -> String {
    std::env::var("PDW_TABLE").unwrap_or_else(|_| "signals.pdw".into())
}

/// Pull up to `limit` PDW rows with `id > start_id` from ClickHouse.
pub fn read_pdw_batch(ch_url: &str, start_id: i64, limit: usize) -> Result<Vec<Pulse>> {
    let (host, port) = host_port(ch_url);
    let table = pdw_table();
    let sql = QUERY
        .replace("{start}", &start_id.to_string())
        .replace("{limit}", &limit.to_string())
        .replace("signals.pdw", &table);
    let body = http_get(&host, port, &sql, "TabSeparatedWithNamesAndTypes")
        .context("clickhouse: query failed")?;
    let mut lines = body.lines();
    let _names = lines.next().context("ch: empty response")?;
    let _types = lines.next().context("ch: missing types line")?;

    let mut rows = Vec::with_capacity(limit);
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 10 {
            warn!("ch: skipping malformed row ({} cols)", f.len());
            continue;
        }
        rows.push(Pulse {
            id: f[0].parse().context("ch: bad id")?,
            power_dbm: f[1].parse().unwrap_or(0.0),
            center_frequency_hz: f[2].parse().unwrap_or(0.0),
            time_of_arrival_ns: f[3].parse().unwrap_or(0.0),
            duration_ns: f[4].parse().unwrap_or(0.0),
            bandwidth_hz: f[5].parse().unwrap_or(0.0),
            direction_of_arrival_deg: f[6].parse().unwrap_or(0.0),
            snr_db: f[7].parse().unwrap_or(0.0),
            modulation: f[8].to_string(),
            is_cw: f[9] == "1",
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_port_with_scheme() {
        let (h, p) = host_port("http://10.0.0.5:9999");
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 9999);
    }

    #[test]
    fn host_port_default_port() {
        let (h, p) = host_port("http://10.0.0.5");
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 8123);
    }

    #[test]
    fn host_port_no_scheme() {
        let (h, p) = host_port("10.0.0.7:8124");
        assert_eq!(h, "10.0.0.7");
        assert_eq!(p, 8124);
    }

    #[test]
    fn host_port_trailing_slash() {
        let (h, p) = host_port("http://10.0.0.5/");
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 8123);
    }

    #[test]
    fn decode_chunked_single() {
        assert_eq!(decode_chunked("5\r\nhello\r\n0\r\n\r\n"), Some("hello".into()));
    }

    #[test]
    fn decode_chunked_multi() {
        assert_eq!(decode_chunked("4\r\nabcd\r\n0\r\n\r\n"), Some("abcd".into()));
    }

    #[test]
    fn decode_chunked_not_chunked() {
        assert_eq!(decode_chunked("plain text"), None);
    }

    #[test]
    fn urlencode_safe_chars() {
        assert_eq!(urlencode("abc-DEF_0.~"), "abc-DEF_0.~");
    }

    #[test]
    fn urlencode_space() {
        assert_eq!(urlencode("a b"), "a%20b");
    }

    #[test]
    fn urlencode_special() {
        assert_eq!(urlencode("a:b?c=d"), "a%3Ab%3Fc%3Dd");
    }

    #[test]
    fn pdw_table_default() {
        std::env::remove_var("PDW_TABLE");
        assert_eq!(pdw_table(), "signals.pdw");
    }

    #[test]
    fn query_template_shape() {
        let sql = QUERY.replace("{start}", "42").replace("{limit}", "100");
        assert!(sql.contains("WHERE id > 42"));
        assert!(sql.contains("LIMIT 100"));
        assert!(sql.contains("ORDER BY id"));
    }
}

/// Read every pulse of `signals.pdw` in id (= arrival/file) order for the
/// ONNX / classic-v2 engines. TSV body parsed in parallel.
pub fn read_flow_pulses(ch_url: &str) -> Result<Vec<crate::onnx_flow::FlowPulse>> {
    use rayon::prelude::*;
    let (host, port) = host_port(ch_url);
    let sql = format!(
        "SELECT id, power_dbm, center_frequency_hz, time_of_arrival_ns, duration_ns, bandwidth_hz, \
         direction_of_arrival_deg, snr_db, modulation FROM {} ORDER BY id",
        pdw_table()
    );
    let body = http_get(&host, port, &sql, "TabSeparated").context("clickhouse: pdw read failed")?;
    let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
    lines
        .par_iter()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() < 9 {
                bail!("pdw row has {} columns: {l:?}", c.len());
            }
            let num = |i: usize| c[i].parse::<f64>().unwrap_or(0.0);
            // Float32 columns: parse as f32 then widen, so the value equals the
            // f32 the loader wrote (parsing the printed text as f64 gives a
            // slightly different number and flips near-tie decisions).
            let num32 = |i: usize| c[i].parse::<f32>().unwrap_or(0.0) as f64;
            Ok(crate::onnx_flow::FlowPulse {
                input_id: c[0].parse().unwrap_or(0),
                power_dbm: num32(1),
                freq_hz: num(2),
                toa_ns: crate::refresh::time_of_arrival_to_ns(c[3]).unwrap_or(0.0),
                duration_ns: num(4),
                bandwidth_hz: num(5),
                doa_deg: num32(6),
                snr_db: num32(7),
                modulation: c[8].to_string(),
            })
        })
        .collect()
}
