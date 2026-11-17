//! Unit tests for the swappable embedding-engine abstraction.
//!
//! `build_engine_from_env` is pure env-reading — no network, no model
//! load — so these tests are safe to run in parallel as long as each one
//! cleans its own env keys first. The classic embed test uses the
//! in-house ClassicEngine directly (no ONNX runtime, no network needed).
//! The remote-ONNX embed test uses the in-process model server from
//! `model_fetch`'s tests (also no real model).

use ai_processor::engine::{build_engine_from_env, model_input_dim, resolve_onnx_url, EmbeddingEngine};
use ai_processor::classic_embed_on;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::time::Duration;

/// Serialises the env-mutating assertions (tests share one process).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn clean_env() {
    std::env::remove_var("CLASSIC");
    std::env::remove_var("EMBED");
    std::env::remove_var("ONNX_URL");
    std::env::remove_var("ONNX_SESSION_COUNT");
    std::env::remove_var("ONNX_THREADS");
}

#[test]
fn onnx_is_default_engine() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    // No CLASSIC / EMBED / ONNX_URL env → remote-ONNX engine (default),
    // pointing at the documented default model-service URL.
    let eng = build_engine_from_env("/nonexistent/model.onnx", 1, 1);
    assert_eq!(eng.name(), "onnx");
    assert_eq!(eng.out_dim(), ai_processor::inference::MODEL_OUTPUT_DIM);
    // No explicit session-count and threads=1 → 1 concurrent call.
    assert_eq!(eng.parallelism(), 1);
    // The default URL has no path segment → the engine appends /embed.
    let eng = ai_processor::engine::OnnxEngine::new("http://127.0.0.1:8000", 2);
    assert_eq!(eng.call_url(), "http://127.0.0.1:8000/embed");
    // A URL that already carries a path is used verbatim.
    let eng = ai_processor::engine::OnnxEngine::new("http://127.0.0.1:8000/embed", 2);
    assert_eq!(eng.call_url(), "http://127.0.0.1:8000/embed");
    // A file-like URL (old ONNX_PATH values) is also used verbatim.
    let eng = ai_processor::engine::OnnxEngine::new("http://10.0.0.5/m.onnx", 1);
    assert_eq!(eng.call_url(), "http://10.0.0.5/m.onnx");
}

#[test]
fn onnx_engine_threads_carry_over_as_parallelism() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    std::env::remove_var("ONNX_SESSION_COUNT");
    // No explicit ONNX_SESSION_COUNT → ONNX_THREADS becomes the
    // HTTP fan-out width (the Python config's thread meaning).
    let eng = build_engine_from_env("/nonexistent/model.onnx", 1, 6);
    assert_eq!(eng.name(), "onnx");
    assert_eq!(eng.parallelism(), 6);
    // An explicit session-count wins over threads.
    std::env::set_var("ONNX_SESSION_COUNT", "3");
    let eng = build_engine_from_env("/nonexistent/model.onnx", 3, 6);
    assert_eq!(eng.parallelism(), 3);
    clean_env();
    std::env::remove_var("ONNX_SESSION_COUNT");
}

#[test]
fn onnx_engine_uses_onnx_url_env() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    std::env::set_var("ONNX_URL", "http://model.host:8000/embed");
    std::env::set_var("ONNX_SESSION_COUNT", "3");
    let eng = build_engine_from_env("/whatever/model.onnx", 3, 6);
    assert_eq!(eng.name(), "onnx");
    assert_eq!(eng.parallelism(), 3);
    // The URL the engine uses is exactly the env var.
    assert_eq!(resolve_onnx_url("/whatever/model.onnx"), "http://model.host:8000/embed");
    clean_env();
}

#[test]
fn onnx_engine_takes_url_from_config_path_when_http() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    // ONNX_PATH (the config field) holding an http(s) URL is used directly
    // even when ONNX_URL is not set.
    let url = resolve_onnx_url("http://10.0.0.5:8000/embed");
    assert_eq!(url, "http://10.0.0.5:8000/embed");
    // A plain local path (classic-era config) falls back to the default
    // local model-service URL rather than trying to load a file.
    let url = resolve_onnx_url("/models/transformer_model_known_v16.onnx");
    assert!(url.starts_with("http://"), "got {url}");
}

#[test]
fn classic_engine_selected_by_classic() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    std::env::set_var("CLASSIC", "1");
    let eng = build_engine_from_env("/nonexistent/model.onnx", 1, 1);
    assert_eq!(eng.name(), "classic");
    assert_eq!(eng.out_dim(), ai_processor::classic_embed::OUT_DIM);
    assert_eq!(eng.parallelism(), 1);
    clean_env();
}

#[test]
fn classic_engine_selected_by_embed() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    std::env::set_var("EMBED", "classic");
    let eng = build_engine_from_env("/nonexistent/model.onnx", 1, 1);
    assert_eq!(eng.name(), "classic");
    clean_env();
}

#[test]
fn model_input_dim_is_458() {
    // The v16 transformer model's fixed input width.
    assert_eq!(model_input_dim(), ai_processor::inference::MODEL_INPUT_DIM);
    assert_eq!(model_input_dim(), 458);
}

/// Start a tiny in-process model server on 127.0.0.1:0 that answers one
/// request with `n` rows of `fill` f32 embeddings, then exits.
fn start_model_server(n: usize, fill: f32) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
        let mut reader = std::io::BufReader::new(&mut stream);
        let _ = read_line(&mut reader);
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

fn read_line<R: Read>(reader: &mut std::io::BufReader<R>) -> std::io::Result<String> {
    let mut buf = [0u8; 1];
    let mut line = String::new();
    loop {
        let n = reader.read(&mut buf)?;
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

#[test]
fn onnx_engine_calls_remote_service_and_returns_embeddings() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    let dim = model_input_dim();
    let n = 4;
    let url = start_model_server(n, 0.25f32);
    // parallel=1: the test server accepts a single connection, so drive
    // this case with one call at a time (no fan-out to a 2nd request).
    let eng = build_engine_from_env(&url, 1, 1);
    assert_eq!(eng.name(), "onnx");
    let input: Vec<f32> = (0..(n * dim)).map(|i| (i % 7) as f32 / 7.0).collect();
    let out = eng.embed(&input, n, dim).unwrap();
    // 4 rows of 1024-dim embeddings, all equal to the server's fill value.
    assert_eq!(out.len(), n * 1024);
    assert!(out.iter().all(|&v| (v - 0.25).abs() < 1e-6));
}

#[test]
fn onnx_engine_empty_input_returns_empty() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    let dim = model_input_dim();
    let url = start_model_server(1, 0.5f32);
    let eng = build_engine_from_env(&url, 1, 1);
    let out = eng.embed(&[], 0, dim).unwrap();
    assert!(out.is_empty());
}

#[test]
fn classic_engine_embed_shape_and_norm() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    let eng = ai_processor::engine::ClassicEngine::new();
    // Classic engine: 8 feature rows of 458 → 8 × (ceil(458/8)*8) f32.
    let n = 8;
    let dim = model_input_dim();
    let expected_dim = (dim + 7) / 8 * 8;
    let mut input = vec![0.0f32; n * dim];
    for (i, slot) in input.iter_mut().enumerate() {
        *slot = ((i * 7) % 13) as f32 / 13.0;
    }
    let out = eng.embed(&input, n, dim).unwrap();
    assert_eq!(out.len(), n * expected_dim);
    // Classic embeddings are deterministic: same input → same output.
    let again = eng.embed(&input, n, dim).unwrap();
    assert_eq!(out, again);
    // Each row is L2-normalised to unit length.
    let row: &[f32] = &out[0..expected_dim];
    let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "row norm = {norm}");
}

#[test]
fn classic_embed_on_flag_tracks_env() {
    let _g = ENV_LOCK.lock().unwrap();
    clean_env();
    // Default: ONNX is the engine — classic flag is off.
    assert!(!classic_embed_on(), "ONNX default: classic_embed_on must be false");
    // EMBED=classic selects the in-house engine.
    std::env::set_var("EMBED", "classic");
    assert!(classic_embed_on(), "EMBED=classic: classic_embed_on must be true");
    std::env::remove_var("EMBED");
    // CLASSIC=1 selects the in-house engine too.
    std::env::set_var("CLASSIC", "1");
    assert!(classic_embed_on(), "CLASSIC=1: classic_embed_on must be true");
    clean_env();
    assert!(!classic_embed_on());
}
