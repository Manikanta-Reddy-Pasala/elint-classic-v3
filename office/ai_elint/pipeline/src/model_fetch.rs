//! Call a remote ONNX model service over HTTP.
//!
//! The ONNX engine no longer loads or runs the model inside this process:
//! the model lives on a different machine, and this service simply POSTs
//! the feature rows to the model's URL and reads the embeddings back.
//! The model service is any HTTP endpoint that speaks:
//!
//!   POST <ONNX_URL>
//!   Content-Type: application/octet-stream
//!   body:   n * 458 f32, little-endian, row-major   (the raw feature rows)
//!
//!   200 OK
//!   Content-Type: application/octet-stream
//!   body:   n * 1024 f32, little-endian, row-major  (the embeddings)
//!
//! This mirrors the repo's existing hand-rolled TCP + HTTP/1.1 style
//! (`ch_read.rs`) — no reqwest/ureq added to the dependency tree.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// One connect/read timeout for a model call.
const TIMEOUT: Duration = Duration::from_secs(120);

/// POST the raw feature rows to `url` and return the raw embedding bytes
/// (`n * out_dim * 4` bytes of little-endian f32). `n` is the row count
/// (used to sanity-check the response length); `dim` is the per-row input
/// width (documented on the wire, not re-checked on the response).
///
/// Returns `Ok(vec)` on success, or `Err` (connect/read/HTTP failure).
/// The caller decides whether to retry or fall back.
pub fn call_model(
    url: &str,
    input: &[f32],
    n: usize,
    dim: usize,
    out_dim: usize,
) -> anyhow::Result<Vec<u8>> {
    return call_model_with_predictions(url, input, n, dim, out_dim)
        .map(|v| v.0);
}

/// ONNX /predict returns a little-endian header: embedding u64, label u64,
/// similarity u64. The /embed compatibility endpoint returns embeddings only.
pub fn call_model_with_predictions(
    url: &str,
    input: &[f32],
    n: usize,
    dim: usize,
    out_dim: usize,
) -> anyhow::Result<(Vec<u8>, Vec<i64>, Vec<f32>)> {
    // The body is a whole number of rows (checked against `n * dim * 4`);
    // the caller passes `dim` only to keep the wire contract documented
    // at the call site.
    let _ = dim;
    let (host, port, path) =
        parse_url(url).ok_or_else(|| anyhow::anyhow!("model_fetch: cannot parse URL {url}"))?;

    let mut stream = TcpStream::connect((host.as_str(), port))
        .map_err(|e| anyhow::anyhow!("model_fetch: connect {host}:{port} failed: {e}"))?;
    let _ = stream.set_read_timeout(Some(TIMEOUT));
    let _ = stream.set_write_timeout(Some(TIMEOUT));

    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: ai_processor\r\n\
         Content-Type: application/octet-stream\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        input.len() * 4
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| anyhow::anyhow!("model_fetch: write request head failed: {e}"))?;
    stream
        .write_all(&input_to_bytes(input))
        .map_err(|e| anyhow::anyhow!("model_fetch: write request body failed: {e}"))?;

    // Status line + headers (skip Content-Length; rely on Connection: close).
    let mut reader = std::io::BufReader::new(stream);
    let status = read_line(&mut reader)?;
    let ok = status
        .split_whitespace()
        .nth(1)
        .map(|s| s.starts_with('2'))
        .unwrap_or(false);
    if !ok {
        anyhow::bail!("model_fetch: model service returned non-2xx: {status:?}");
    }
    loop {
        let line = read_line(&mut reader)?;
        if line.is_empty() {
            break;
        }
    }

    // Stream the whole body (connection closes at the end).
    let mut body = Vec::new();
    reader
        .read_to_end(&mut body)
        .map_err(|e| anyhow::anyhow!("model_fetch: read response body failed: {e}"))?;

    let expected = n * out_dim * 4;
    if path.trim_end_matches('/') == "/predict" {
        if body.len() < 24 {
            anyhow::bail!("model_fetch: /predict must return a 24-byte length header");
        }
        let ne = u64::from_le_bytes(body[0..8].try_into()?) as usize;
        let nl = u64::from_le_bytes(body[8..16].try_into()?) as usize;
        let ns = u64::from_le_bytes(body[16..24].try_into()?) as usize;
        if ne != expected || nl != n * 8 || ns != n * 4 {
            anyhow::bail!("model_fetch: incorrect /predict section lengths");
        }
        let mut result = vec![0; ne + nl + ns];
        let _ = body.drain(0..24);
        reader.read_exact(&mut result)?;
        let embeddings = result[..ne].to_vec();
        let labels = result[ne..ne + nl]
            .chunks_exact(8)
            .map(|b| Ok(i64::from_le_bytes(b.try_into()?)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let similarities = bytes_to_f32(&result[ne + nl..ne + nl + ns]);
        if labels.len() != n || similarities.len() != n {
            anyhow::bail!("model_fetch: incomplete /predict response");
        }
        return Ok((embeddings, labels, similarities));
    }
    if body.len() != expected {
        anyhow::bail!(
            "model_fetch: expected {expected} bytes of embeddings, got {}",
            body.len()
        );
    }
    Ok((body, vec![-1; n], vec![0.0; n]))
}

/// f32 slice → little-endian bytes (native on x86-64).
pub fn input_to_bytes(input: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 4);
    for v in input {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// little-endian f32 bytes → f32 slice.
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for c in bytes.chunks_exact(4) {
        let arr: [u8; 4] = [c[0], c[1], c[2], c[3]];
        out.push(f32::from_le_bytes(arr));
    }
    out
}

/// Parse `host`, `port`, `path` out of a http(s) URL.
pub(crate) fn parse_url(url: &str) -> Option<(String, u16, String)> {
    let without_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))?;
    let (hostport, path) = match without_scheme.find('/') {
        Some(i) => (&without_scheme[..i], &without_scheme[i..]),
        None => (without_scheme, "/"),
    };
    let (host, port) = match hostport.rfind(':') {
        Some(i) => (hostport[..i].to_string(), hostport[i + 1..].parse().ok()?),
        None => (hostport.to_string(), 80),
    };
    Some((host, port, path.to_string()))
}

/// Read one `\n`-terminated line (stripping `\r\n`); `Ok("")` at EOF.
fn read_line<R: Read>(reader: &mut std::io::BufReader<R>) -> anyhow::Result<String> {
    let mut buf = [0u8; 1];
    let mut line = String::new();
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| anyhow::anyhow!("model_fetch: read line failed: {e}"))?;
        if n == 0 {
            return Ok(line);
        }
        if buf[0] == b'\n' {
            break;
        }
        line.push(buf[0] as char);
    }
    if line.ends_with('\r') {
        line.pop();
    }
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Start a tiny in-process model server on 127.0.0.1:0. It reads one
    /// HTTP request, returns a fixed 200 with `n*1024` f32 bytes of value
    /// `fill`, and exits. Returns the URL to call.
    fn start_model_server(n: usize, fill: f32) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
            let mut reader = std::io::BufReader::new(&mut stream);
            let _ = read_line(&mut reader); // request line
            let mut content_len: Option<usize> = None;
            loop {
                let line = read_line(&mut reader).unwrap_or_default();
                if line.is_empty() {
                    break;
                }
                if let Some(v) = line
                    .strip_prefix("Content-Length:")
                    .or_else(|| line.strip_prefix("content-length:"))
                {
                    content_len = v.trim().parse().ok();
                }
            }
            if let Some(len) = content_len {
                let mut body = vec![0u8; len];
                let _ = std::io::Read::read_exact(&mut reader, &mut body);
            }
            let mut out = Vec::with_capacity(n * 1024 * 4);
            let le = fill.to_le_bytes();
            for _ in 0..(n * 1024) {
                out.extend_from_slice(&le);
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                out.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.write_all(&out);
        });
        format!("http://127.0.0.1:{}/embed", addr.port())
    }

    #[test]
    fn parse_url_basic() {
        assert_eq!(
            parse_url("http://h:8080/a/b.onnx").unwrap(),
            ("h".to_string(), 8080, "/a/b.onnx".to_string())
        );
        assert_eq!(
            parse_url("https://h/x.onnx").unwrap(),
            ("h".to_string(), 80, "/x.onnx".to_string())
        );
        assert!(parse_url("not-a-url").is_none());
    }

    #[test]
    fn f32_roundtrip_le_bytes() {
        let v: Vec<f32> = vec![1.5, -2.25, 0.0, 3.75];
        let bytes = input_to_bytes(&v);
        assert_eq!(bytes.len(), 16);
        assert_eq!(bytes_to_f32(&bytes), v);
    }

    #[test]
    fn call_model_posts_and_parses_embeddings() {
        let n = 2;
        let dim = crate::inference::MODEL_INPUT_DIM;
        let url = start_model_server(n, 0.125f32);
        let input = vec![0.5f32; n * dim];
        let body = call_model(&url, &input, n, dim, 1024).unwrap();
        let emb = bytes_to_f32(&body);
        assert_eq!(emb.len(), n * 1024);
        assert!(emb.iter().all(|&v| (v - 0.125).abs() < 1e-6));
    }

    #[test]
    fn call_model_wrong_body_length_errors() {
        // Server answers with n=1 worth of embeddings, but the client
        // asked for n=2 → the length check must reject it.
        let url = start_model_server(1, 0.5f32);
        let dim = crate::inference::MODEL_INPUT_DIM;
        let input = vec![0.5f32; 2 * dim];
        let err = call_model(&url, &input, 2, dim, 1024).unwrap_err();
        assert!(err.to_string().contains("expected"), "got: {err}");
    }

    #[test]
    fn call_model_unreachable_errors() {
        let dim: usize = crate::inference::MODEL_INPUT_DIM;
        let input: Vec<f32> = vec![0.1f32; dim];
        let err = call_model("http://127.0.0.1:1/embed", &input, 1, dim, 1024);
        assert!(err.is_err());
    }
}
