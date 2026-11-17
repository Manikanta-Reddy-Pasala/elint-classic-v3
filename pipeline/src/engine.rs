//! Swappable embedding-engine abstraction.
//!
//! The rest of the pipeline (features → clustering → merge → DB sinks)
//! never talks to a concrete model: it gets an [`EmbeddingEngine`] from
//! [`build_engine_from_env`] and calls [`EmbeddingEngine::embed`].
//!
//! Engines today:
//!   * `onnx`    — remote ONNX model, called over HTTP (default)
//!   * `classic` — in-house feature-based bag-of-windows embedding
//!
//! Adding a new engine = implement [`EmbeddingEngine`] + add a match arm
//! in [`build_engine_from_env`]. Multiple machines can run different
//! engines (or point the ONNX engine at different model services) from
//! the same binary by setting the `EMBED` / `ONNX_URL` env vars — no
//! code change.
//!
//! The ONNX engine does NOT load or run the model in-process: it POSTs
//! the feature rows to the model's HTTP endpoint (see
//! [`crate::model_fetch::call_model`]) and reads the embeddings back.
//! The model itself runs wherever that endpoint lives.

use std::sync::Arc;

use anyhow::Result;

use crate::inference::{MODEL_INPUT_DIM, MODEL_OUTPUT_DIM};

/// One embedding engine: turns `n` flat feature rows (dim `dim`) into
/// `n * out_dim` f32 embedding values, in row order.
pub struct EmbeddingOutput {
    pub embeddings: Vec<f32>,
    /// ONNX model center_predictions and similarities, both pulse order.
    /// Classic never populates these: this is an ONNX-only routing signal.
    pub labels: Vec<i64>,
    pub similarities: Vec<f32>,
}

pub trait EmbeddingEngine: Send + Sync {
    /// Engine name for logs and reporting (`"onnx"`, `"classic"`).
    fn name(&self) -> &str;
    /// Dimension of one embedding produced for one pulse.
    fn out_dim(&self) -> usize;
    /// `input` holds `n * dim` f32; returns `n * self.out_dim()` f32.
    fn embed(&self, input: &[f32], n: usize, dim: usize) -> Result<Vec<f32>>;
    fn embedding_output(&self, input: &[f32], n: usize, dim: usize) -> Result<EmbeddingOutput> {
        let _ = (input, n, dim);
        anyhow::bail!("{}: prediction output is not available", self.name())
    }
    /// Parallelism hint (ONNX: concurrent HTTP calls; 1 for classic).
    fn parallelism(&self) -> usize;
}

/// Remote-ONNX engine: POSTs feature rows to the model service's URL
/// and reads the embeddings back. No model file, no onnxruntime, no
/// session pool in this process — the model runs on a different machine
/// (or container) that the user runs themselves.
///
/// The wire protocol is `crate::model_fetch::call_model`:
///   POST <url>  body = n*dim f32 LE  →  200  body = n*1024 f32 LE
pub struct OnnxEngine {
    /// Base URL of the model service. If the URL has no path suffix the
    /// engine appends `/embed` (the same shape the Python repo calls).
    url: String,
    /// Number of concurrent in-flight model calls (the ONNX
    /// "session-count" equivalent, now meaning HTTP concurrency).
    parallel: usize,
}

impl OnnxEngine {
    /// `model_url` is the full URL to POST to, e.g.
    /// `http://model-host:8000/embed`.
    pub fn new(model_url: &str, parallel: usize) -> Self {
        Self {
            url: model_url.to_string(),
            parallel: parallel.max(1),
        }
    }

    /// The full URL this engine POSTs to. The configured URL is used
    /// verbatim when it already carries a path (or a file-like last
    /// segment); a bare `host[:port]` gets `/embed` appended (matches
    /// the Python repo's model-service endpoint shape).
    pub fn call_url(&self) -> String {
        let base = self.url.trim_end_matches('/');
        let after_scheme = base.find("://").map(|i| &base[i + 3..]).unwrap_or("");
        // A path exists when the authority (host[:port]) itself carries a
        // slash, or a '/' follows the authority: host[:port] alone has no
        // slash (default port 80), host:port/path and host/path both do.
        let has_path = after_scheme
            .split_once('/')
            .map(|(host, _)| !host.contains('/'))
            .unwrap_or(false);
        if has_path {
            self.url.clone()
        } else {
            format!("{base}/embed")
        }
    }
}

impl EmbeddingEngine for OnnxEngine {
    fn name(&self) -> &str {
        "onnx"
    }
    fn out_dim(&self) -> usize {
        MODEL_OUTPUT_DIM
    }
    fn embed(&self, input: &[f32], n: usize, dim: usize) -> Result<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        // Fan out the batch across `self.parallel` concurrent HTTP calls.
        // Split into near-equal contiguous chunks (like the old session
        // pool) and call the model once per chunk on its own thread.
        let url = self.call_url();
        let par = self.parallel.min(n);
        let base = n / par;
        let rem = n % par;
        let urls: Vec<String> = vec![url; par];
        let inputs: Vec<Vec<f32>> = (0..par)
            .map(|i| {
                let len = base + if i < rem { 1 } else { 0 };
                let start = i * base + i.min(rem);
                input[start * dim..(start + len) * dim].to_vec()
            })
            .collect();

        let handles: Vec<std::thread::JoinHandle<Result<Vec<f32>>>> =
            inputs.into_iter().zip(urls.into_iter()).map(|(rows, u)| {
                std::thread::spawn(move || {
                    let n_chunk = rows.len() / dim;
                    crate::model_fetch::call_model(&u, &rows, n_chunk, dim, MODEL_OUTPUT_DIM)
                        .map(|b| crate::model_fetch::bytes_to_f32(&b))
                })
            }).collect();

        let mut out: Vec<f32> = Vec::with_capacity(n * MODEL_OUTPUT_DIM);
        for h in handles {
            let res = h
                .join()
                .map_err(|_| anyhow::anyhow!("onnx: worker thread panicked"))?;
            out.extend(res?);
        }
        Ok(out)
    }
    fn embedding_output(&self, input: &[f32], n: usize, dim: usize) -> Result<EmbeddingOutput> {
        if n == 0 { return Ok(EmbeddingOutput { embeddings: vec![], labels: vec![], similarities: vec![] }); }
        let url = self.call_url().trim_end_matches('/').strip_suffix("/embed").unwrap_or(self.call_url().trim_end_matches('/')).to_string() + "/predict";
        let par = self.parallel.min(n);
        let base = n / par;
        let rem = n % par;
        let mut jobs = Vec::new();
        for i in 0..par {
            let len = base + if i < rem { 1 } else { 0 };
            let start = i * base + i.min(rem);
            let rows = input[start * dim..(start + len) * dim].to_vec();
            let u = url.clone();
            jobs.push(std::thread::spawn(move || {
                crate::model_fetch::call_model_with_predictions(&u, &rows, len, dim, MODEL_OUTPUT_DIM)
            }));
        }
        let mut out = EmbeddingOutput { embeddings: Vec::with_capacity(n * MODEL_OUTPUT_DIM), labels: Vec::with_capacity(n), similarities: Vec::with_capacity(n) };
        for h in jobs {
            let (bytes, labels, similarities) = h.join().map_err(|_| anyhow::anyhow!("onnx: worker thread panicked"))??;
            out.embeddings.extend(crate::model_fetch::bytes_to_f32(&bytes));
            out.labels.extend(labels);
            out.similarities.extend(similarities);
        }
        Ok(out)
    }
    fn parallelism(&self) -> usize {
        self.parallel
    }
}

/// Adapter: the in-house classic bag-of-windows embedding
/// (`classic_embed::embed`) as an engine of the same shape.
pub struct ClassicEngine;

impl ClassicEngine {
    pub fn new() -> Self {
        Self
    }
}

impl EmbeddingEngine for ClassicEngine {
    fn name(&self) -> &str {
        "classic"
    }
    fn out_dim(&self) -> usize {
        crate::classic_embed::OUT_DIM
    }
    fn embed(&self, input: &[f32], n: usize, dim: usize) -> Result<Vec<f32>> {
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let d = dim.min(crate::features::FEATURE_DIM);
                input[i * dim..i * dim + d].iter().map(|&v| v as f64).collect()
            })
            .collect();
        let emb = crate::classic_embed::embed(&rows, crate::features::WINDOW_SIZE);
        Ok(emb.into_iter().flat_map(|r| r).collect())
    }
    fn parallelism(&self) -> usize {
        1
    }
}

/// Resolve the ONNX model service URL at engine-selection time.
///
///   * `ONNX_URL` set → that URL is used (the model runs over there).
///   * otherwise `ONNX_PATH`/config value starting with http(s):// →
///     used as the URL directly.
///   * otherwise the default URL (so a deployment that never sets any of
///     these still has a documented default to point at).
pub fn resolve_onnx_url(onnx_path: &str) -> String {
    if let Ok(v) = std::env::var("ONNX_URL") {
        if !v.trim().is_empty() {
            return v;
        }
    }
    if onnx_path.starts_with("http://") || onnx_path.starts_with("https://") {
        return onnx_path.to_string();
    }
    // Default: a model service on the local host, same port the Python
    // repo's model server uses. Overridden by any of the two env vars.
    "http://127.0.0.1:8000".to_string()
}

/// Build the engine selected by env:
///   `CLASSIC=1` or `EMBED=classic` → classic engine
///   otherwise                      → remote-ONNX engine (default)
///
/// `onnx_path` is the configured model location; it is treated as a URL
/// when it looks like one (see [`resolve_onnx_url`]), so the same config
/// field can hold either a local path (classic-era configs) or the model
/// service URL.
pub fn build_engine_from_env(
    onnx_path: &str,
    onnx_session_count: usize,
    onnx_threads: usize,
) -> Arc<dyn EmbeddingEngine> {
    if crate::classic_embed_on() {
        tracing::info!("engine: in-house classic feature engine (EMBED=classic or CLASSIC=1)");
        return Arc::new(ClassicEngine::new());
    }
    let url = resolve_onnx_url(onnx_path);
    // ONNX threads carry over as the HTTP fan-out width when no explicit
    // ONNX_SESSION_COUNT was set (the Python config's ONNX_THREADS
    // meaning: how many in-flight calls per batch).
    let parallel = if std::env::var("ONNX_SESSION_COUNT").is_err() {
        onnx_threads.max(1)
    } else {
        onnx_session_count.max(1)
    };
    tracing::info!(
        "engine: remote-ONNX engine (model service url={url}, {} concurrent calls)",
        parallel
    );
    Arc::new(OnnxEngine::new(&url, parallel))
}

/// Model input width callers pad feature rows to (458 for the v16 model).
pub fn model_input_dim() -> usize {
    MODEL_INPUT_DIM
}
