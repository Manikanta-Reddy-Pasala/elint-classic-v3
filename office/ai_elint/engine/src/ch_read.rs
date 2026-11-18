//! CDC read — direct read from ClickHouse `signals.pdw` via HTTP.
//!
//! Persistent keep-alive TCP connection per reader (no per-call connect).
//! `read_batch` sends the query and reads the full response in one call;
//! the socket stays open so the next read skips the TCP handshake.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::Duration;
use anyhow::{anyhow, bail, Context, Result};
use tracing::warn;
/// One PDW pulse row from ClickHouse.
#[derive(Debug, Clone)]
pub struct Pulse {
    pub id: u64,
    pub power_dbm: f32,
    pub center_frequency_hz: f64,
    pub time_of_arrival_ns: u64,
    pub duration_ns: f64,
    pub bandwidth_hz: f64,
    pub direction_of_arrival_deg: f32,
    pub snr_db: f32,
    pub modulation: String,
    pub is_cw: bool,
    pub receiver_id: u8,
}
impl Pulse {
    pub fn emitter_id(&self) -> String {
        format!("em-{}", self.id)
    }
}
/// Persistent ClickHouse HTTP connection (one per reader task).
///
/// The TCP socket is opened once and reused. Each `read_batch` call
/// sends a new GET request over the same socket and reads the response.
/// `Connection: keep-alive` keeps the socket open between calls.
pub struct ChConn {
    host: String,
    port: u16,
    stream: Mutex<TcpStream>,
}
impl ChConn {
    /// Open the TCP connection once.
    pub fn connect(ch_url: &str) -> Result<Self> {
        let (host, port) = host_port(ch_url);
        let stream = TcpStream::connect((host.as_str(), port)).context("ch: tcp connect")?;
        stream.set_nodelay(true)?;
        Ok(Self {
            host,
            port,
            stream: Mutex::new(stream),
        })
    }
    /// Send a query and read the full response body. Reuses the socket.
    ///
    /// ClickHouse always responds with `Transfer-Encoding: chunked` for
    /// SELECT queries (GET or POST, keep-alive or close — verified on
    /// the VM).  We therefore decode the chunked body properly:
    /// `<hex-size>\r\n<data>\r\n ... 0\r\n\r\n`.
    pub fn http_roundtrip(&self, query: &str, format: &str) -> Result<String> {
        let url = format!("/?{}query={}", auth_query(), urlencode(query));
        let req = format!(
            "GET {url} HTTP/1.1\r\nHost: {}:{}\r\nX-ClickHouse-Format: {}\r\nConnection: keep-alive\r\n\r\n",
            self.host,
            self.port,
            format
        );
        let mut stream = self.stream.lock().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        stream.write_all(req.as_bytes()).context("ch: send request")?;

        // Phase 1: read the headers (up to \r\n\r\n).
        let mut buf: Vec<u8> = Vec::with_capacity(65536);
        let mut tmp = [0u8; 65536];
        let header_end = loop {
            if let Some(pos) = find_header_end(&buf) {
                break pos;
            }
            match stream.read(&mut tmp) {
                Ok(0) => return Err(anyhow!("ch: connection closed before headers")),
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                    return Err(anyhow!("ch: read headers: timeout"))
                }
                Err(e) => return Err(anyhow::anyhow!("ch: read headers: {e}")),
            }
        };
        let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();

        // Status check.
        if !header_text.starts_with("HTTP/1.1 2") {
            let body_str = String::from_utf8_lossy(&buf[header_end..]);
            bail!(
                "ch: non-200: {} | {}",
                header_text.lines().next().unwrap_or(""),
                &body_str[..body_str.len().min(300)]
            );
        }

        // Bytes that arrived together with the headers, after \r\n\r\n.
        let mut body: Vec<u8> = buf[header_end..].to_vec();
        let mut out: Vec<u8> = Vec::new();

        let is_chunked = header_text.to_lowercase().contains("transfer-encoding: chunked");
        let content_length: usize = header_text
            .lines()
            .find_map(|l| {
                let l = l.to_lowercase();
                l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))
            })
            .unwrap_or(0);

        if is_chunked {
            // Decode: <hex-size>\r\n <data>\r\n ... 0\r\n\r\n
            // `body` holds the bytes after the header block; we drain
            // framing bytes out of it and append the chunk DATA to `out`.
            loop {
                // Read until we have a full size line (terminated by \r\n).
                while !body_contains_crlf(&body) {
                    match stream.read(&mut tmp) {
                        Ok(0) => return Err(anyhow!("ch: EOF mid-chunk-size")),
                        Ok(n) => body.extend_from_slice(&tmp[..n]),
                        Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                            return Err(anyhow!("ch: read chunk size: timeout"))
                        }
                        Err(e) => return Err(anyhow!("ch: read chunk size: {e}")),
                    }
                }
                // Size line is the first line of `body`.
                let crlf_pos = find_crlf(&body).unwrap();
                let size_line =
                    String::from_utf8_lossy(&body[..crlf_pos]).trim().to_string();
                let chunk_size: usize = usize::from_str_radix(&size_line, 16)
                    .map_err(|_| anyhow!("ch: bad chunk size: {size_line}"))?;
                // Drop the size line (includes the \r\n).
                body.drain(..crlf_pos + 2);

                if chunk_size == 0 {
                    // Consume the final trailing \r\n after the 0 chunk.
                    while body.len() < 2 {
                        match stream.read(&mut tmp) {
                            Ok(0) => break,
                            Ok(n) => body.extend_from_slice(&tmp[..n]),
                            Err(_) => break,
                        }
                    }
                    if body.len() >= 2 {
                        body.drain(..2);
                    }
                    break;
                }
                // Ensure we have chunk data + trailing \r\n.
                while body.len() < chunk_size + 2 {
                    match stream.read(&mut tmp) {
                        Ok(0) => return Err(anyhow!("ch: EOF mid-chunk-data")),
                        Ok(n) => body.extend_from_slice(&tmp[..n]),
                        Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                            return Err(anyhow!("ch: read chunk data: timeout"))
                        }
                        Err(e) => return Err(anyhow!("ch: read chunk data: {e}")),
                    }
                }
                // Keep the data, drop it plus its trailing \r\n.
                out.extend_from_slice(&body[..chunk_size]);
                body.drain(..chunk_size + 2);
            }
        } else {
            // Plain Content-Length framing.
            while out.len() < content_length {
                if !body.is_empty() {
                    let take = body.len().min(content_length - out.len());
                    out.extend_from_slice(&body[..take]);
                    body.drain(..take);
                }
                if out.len() >= content_length {
                    break;
                }
                match stream.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => {
                        let take = n.min(content_length - out.len());
                        out.extend_from_slice(&tmp[..take]);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                    Err(e) => return Err(anyhow!("ch: read body: {e}")),
                }
            }
        }

        Ok(String::from_utf8_lossy(&out).to_string())
    }
    /// Pull up to `limit` PDW rows with `id > start_id`.
    pub fn read_batch(&self, start_id: u64, limit: u32) -> Result<Vec<Pulse>> {
        let sql = format!(
            "SELECT id, power_dbm, center_frequency_hz, toString(time_of_arrival_ns), \
             duration_ns, bandwidth_hz, direction_of_arrival_deg, snr_db, \
             modulation, is_cw, receiver_id \
             FROM signals.pdw WHERE id > {} ORDER BY id LIMIT {}",
            start_id, limit
        );
        let body = self.http_roundtrip(&sql, "TabSeparatedWithNamesAndTypes")?;
        Ok(parse_rows(&body))
    }
    /// Pull up to `limit` PDW rows starting at row-offset `offset`
    /// (0-based). Uses `LIMIT offset, n` which walks the table in
    /// primary-key order regardless of id values.
    pub fn read_batch_offset(&self, offset: u64, limit: u32) -> Result<Vec<Pulse>> {
        let sql = format!(
            "SELECT id, power_dbm, center_frequency_hz, toString(time_of_arrival_ns), \
             duration_ns, bandwidth_hz, direction_of_arrival_deg, snr_db, \
             modulation, is_cw, receiver_id \
             FROM signals.pdw ORDER BY id LIMIT {} OFFSET {}",
            limit, offset
        );
        let body = self.http_roundtrip(&sql, "TabSeparatedWithNamesAndTypes")?;
        Ok(parse_rows(&body))
    }
    /// Run a simple scalar query (e.g. `SELECT count()`).
    pub fn query_scalar(&self, sql: &str) -> Result<String> {
        self.http_roundtrip(sql, "TabSeparated")
    }
    /// Reconnect: close the old socket and open a fresh one.
    pub fn reconnect(&self) -> Result<()> {
        let mut stream = self.stream.lock().unwrap();
        let _ = stream.shutdown(std::net::Shutdown::Both);
        *stream = TcpStream::connect((self.host.as_str(), self.port))
            .context("ch: reconnect")?;
        stream.set_nodelay(true)?;
        Ok(())
    }
}
/// Parse TSV body: skip names + types lines, parse data rows.
pub fn parse_rows(body: &str) -> Vec<Pulse> {
    let mut lines = body.lines();
    let _names = lines.next();
    let _types = lines.next();
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 11 {
            warn!("ch: skipping malformed row ({} cols)", f.len());
            continue;
        }
        rows.push(Pulse {
            id: f[0].parse().unwrap_or(0),
            power_dbm: f[1].parse().unwrap_or(0.0),
            center_frequency_hz: f[2].parse().unwrap_or(0.0),
            time_of_arrival_ns: f[3].parse::<u64>().unwrap_or_else(|_| {
                f[3].parse::<f64>().map(|v| v as u64).unwrap_or(0)
            }),
            duration_ns: f[4].parse().unwrap_or(0.0),
            bandwidth_hz: f[5].parse().unwrap_or(0.0),
            direction_of_arrival_deg: f[6].parse().unwrap_or(0.0),
            snr_db: f[7].parse().unwrap_or(0.0),
            modulation: f[8].to_string(),
            is_cw: f[9] == "1",
            receiver_id: f[10].parse().unwrap_or(1),
        });
    }
    rows
}
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
pub fn ch_auth() -> (String, String) {
    // The e2e harness passes CLICKHOUSE_DB_USER/PASS as shell vars for the
    // CSV loader, but the engine container gets its env via `docker run -e`.
    // If the env vars are not set in the container, fall back to the
    // docker-compose defaults (user=elint, password=elint).
    let user = std::env::var("CLICKHOUSE_USER")
        .or_else(|_| std::env::var("CLICKHOUSE_DB_USER"))
        .unwrap_or_else(|_| "elint".into());
    let pass = std::env::var("CLICKHOUSE_PASSWORD")
        .or_else(|_| std::env::var("CLICKHOUSE_DB_PASS"))
        .unwrap_or_else(|_| "elint".into());
    (user, pass)
}
pub fn auth_query() -> String {
    let (u, p) = ch_auth();
    match (u.is_empty(), p.is_empty()) {
        (true, true) => String::new(),
        _ => format!("user={}&password={}&", urlencode(&u), urlencode(&p)),
    }
}
pub fn body_contains_crlf(buf: &[u8]) -> bool {
    if buf.len() < 2 {
        return false;
    }
    for i in 0..(buf.len() - 1) {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' {
            return true;
        }
    }
    false
}

pub fn find_crlf(buf: &[u8]) -> Option<usize> {
    if buf.len() < 2 {
        return None;
    }
    for i in 0..(buf.len() - 1) {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' {
            return Some(i);
        }
    }
    None
}

pub fn find_header_end(buf: &[u8]) -> Option<usize> {
    // Find the CRLF CRLF that ends the HTTP header block.
    let pat = [b'\r', b'\n', b'\r', b'\n'];
    if buf.len() < 4 {
        return None;
    }
    for i in 0..(buf.len() - 3) {
        if buf[i] == pat[0] && buf[i + 1] == pat[1] && buf[i + 2] == pat[2] && buf[i + 3] == pat[3] {
            return Some(i + 4);
        }
    }
    None
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
/// `CH_ROWS` — rows per ClickHouse fetch (default 20000).
pub fn fetch_limit() -> u32 {
    std::env::var("CH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000)
}
/// One-shot read (convenience for callers that don't keep a conn).
pub fn read_batch(ch_url: &str, start_id: u64, limit: u32) -> Result<Vec<Pulse>> {
    let conn = ChConn::connect(ch_url)?;
    conn.read_batch(start_id, limit)
}
// ── Unit tests (no DB needed) ────────────────────────────────────────────────
