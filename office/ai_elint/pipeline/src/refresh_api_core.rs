//! Core HTTP handler functions for the refresh API, exposed as a library
//! module so integration tests can reach them.

use std::io::{Read, Write};
use std::sync::Arc;

use anyhow::Result;
use tracing::info;

use crate::config::Config;
use crate::refresh::run_refresh;
use crate::refresh_logger::RotatingSink;
use crate::AiState;

/// Parse a raw HTTP/1.1 request (method, path, body) from the first read.
pub fn parse_request(buf: &[u8]) -> (String, String, String) {
    let head = String::from_utf8_lossy(buf).to_string();
    // Support both \r\n and \n line endings
    let normalized = head.replace("\r\n", "\n");
    let mut lines = normalized.split('\n');
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let content_length = normalized
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.trim().eq_ignore_ascii_case("content-length") {
                v.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    let head_end_norm = normalized.find("\n\n").map(|i| i + 2).unwrap_or(normalized.len());
    let body_start = if head.contains("\r\n") {
        head.find("\r\n\r\n").map(|i| i + 4).unwrap_or(head.len())
    } else {
        head_end_norm
    };
    let body = if content_length > 0 && buf.len() >= body_start + content_length {
        String::from_utf8_lossy(&buf[body_start..body_start + content_length]).to_string()
    } else {
        String::new()
    };
    (method, path, body)
}

/// Pull `key=value` out of a `?key=value` query string in the path.
pub fn extract_query_param(path: &str, key: &str) -> Option<String> {
    let q = path.split('?').nth(1)?;
    for pair in q.split('&') {
        if let Some(val) = pair.strip_prefix(&format!("{key}=")) {
            return Some(val.to_string());
        }
    }
    None
}

/// Pull `"csv_filename": "..."` out of the JSON body (no JSON crate needed).
pub fn extract_csv_filename(body: &str) -> Option<String> {
    let key = "\"csv_filename\"";
    let i = body.find(key)?;
    let rest = body[i + key.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Write a JSON HTTP response back over the TCP stream.
pub fn write_json(stream: &mut std::net::TcpStream, code: u16, body: &str) -> Result<()> {
    let status_text = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let resp = format!(
        "HTTP/1.1 {code} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}

/// Shared state for the refresh API handler.
pub struct ApiState {
    pub cfg: Config,
    pub state: Arc<AiState>,
    pub sink: RotatingSink,
}

/// Handle a single TCP connection: read the request, route it, write the reply.
pub fn handle_conn(mut stream: std::net::TcpStream, api: Arc<ApiState>) -> Result<()> {
    let mut buf = Vec::with_capacity(65536);
    let mut tmp = [0u8; 8192];
    loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > 65536 {
            break;
        }
        if let Some(i) = String::from_utf8_lossy(&buf).find("\r\n\r\n") {
            let head_len = i + 4;
            let len = std::str::from_utf8(&buf).ok()
                .and_then(|s| s.lines().find_map(|l| l.split_once(':').and_then(
                    |(k, v)| if k.trim().eq_ignore_ascii_case("content-length") {
                        v.trim().parse::<usize>().ok()
                    } else { None })))
                .unwrap_or(0);
            if buf.len() >= head_len + len {
                break;
            }
        }
    }
    let (method, path, body) = parse_request(&buf);
    info!(target: "refresh_api", "[API] {} {} buf_len={} body_len={}", method, path, buf.len(), body.len());

    let route_path = path.split('?').next().unwrap_or(&path);

    match (method.as_str(), route_path) {
        ("GET", "/health") => {
            write_json(&mut stream, 200, r#"{"status":"ok","ready":true}"#)
        }
        ("POST", "/api/refresh-data") => {
            let csv_filename = extract_csv_filename(&body)
                .or_else(|| extract_query_param(&path, "file"))
                .unwrap_or_default();
            if csv_filename.is_empty() {
                return write_json(
                    &mut stream,
                    200,
                    r#"{"status":"failed","error":"csv_filename is required (JSON body or ?file= param)"}"#,
                );
            }
            let st = api.state.clone();
            let cfg = api.cfg.clone();
            let sink = api.sink.clone();
            let resp = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(run_refresh(&cfg, &st.pg, &st.ch, &st, &sink, &csv_filename));
            match resp {
                Ok(()) => write_json(&mut stream, 200, r#"{"status":"success"}"#),
                Err(e) => {
                    let msg = format!(
                        r#"{{"status":"failed","error":"{}"}}"#,
                        e.to_string().replace('"', "'")
                    );
                    write_json(&mut stream, 200, &msg)
                }
            }
        }
        _ => write_json(&mut stream, 404, r#"{"status":"failed","error":"not found"}"#),
    }
}
