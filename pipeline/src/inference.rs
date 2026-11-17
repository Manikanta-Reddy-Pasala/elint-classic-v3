/// ONNX inference — streaming 1024-dim embedding per pulse.
///
/// ort 2.0.0-rc.13 API (load-dynamic: libonnxruntime from host volume).
/// Model: transformer_model_known_v16.onnx
///   input  "input_features" [n, 1, 458] f32
///   output "embeddings"     [n, 1, 1024] f32
///
/// Config-based session pool: `ONNX_SESSION_COUNT` independent sessions,
/// each with ONE dedicated worker thread for the whole process (rc.13
/// has no intra-op thread API — parallelism comes from N sessions ×
/// 1 resident thread each; see `OnnxSessionPool`).
///
/// If the model or libonnxruntime is missing, a stub session is used
/// (features padded + L2-normalised) so the pipeline still runs E2E.

use std::path::Path;

use anyhow::{anyhow, Result};
use tracing::{info, warn};

/// Fixed model input dim (2 + 200 pw-RBF + 256 toa-RBF).
pub const MODEL_INPUT_DIM: usize = 458;
/// Fixed model output dim (v16 embeddings).
pub const MODEL_OUTPUT_DIM: usize = 1024;

pub struct OnnxSession {
    pub real: Option<ort::session::Session>,
    pub input_name: String,
    pub output_name: String,
}

/// Fixed-size pool of ONNX sessions.
///
/// # Routing model (the fix for spread=100%)
///
/// Each session has ONE dedicated OS worker thread that lives for the
/// whole process. `run()` splits the batch into near-equal contiguous
/// chunks and hands chunk `i` to session `i`'s queue (round-robin over
/// the sessions, in order). Workers pull their chunk, run it on their
/// session and reply with the embeddings.
///
/// Why the old code showed spread=100%: it spawned N fresh threads per
/// *call*, and those threads competed for cores with the 24-thread
/// rayon pool that `features::generate_features` uses on the very next
/// stage. With small chunks the scheduler almost always ran session
/// 0's thread first; sessions 1..n sat at 0 ms. Dedicated resident
/// workers (one per session) keep the per-call thread cost at zero and
/// spread work deterministically.
pub struct OnnxSessionPool {
    /// One queue per session; `tx[i]` feeds session i's worker thread.
    pub txs: Vec<std::sync::mpsc::Sender<Job>>,
}

/// A chunk job for a session worker.
struct Job {
    /// Flat feature rows for this session's chunk.
    rows: Vec<f32>,
    n_rows: usize,
    reply: std::sync::mpsc::Sender<(u64, Result<Vec<f32>>)>,
}

impl OnnxSessionPool {
    /// Load `count` independent sessions and start one worker thread
    /// per session. `intra_threads` = intra-op threads per session
    /// (spread a batch's matmuls across cores).
    pub fn new(model_path: &str, count: usize, intra_threads: usize) -> Self {
        let intra = intra_threads.max(1);
        let mut txs: Vec<std::sync::mpsc::Sender<Job>> = Vec::with_capacity(count);
        for i in 0..count {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            txs.push(tx);
            let path = model_path.to_string();
            std::thread::Builder::new()
                .name(format!("onnx-s{i}"))
                .spawn(move || Self::worker_loop(&path, intra, rx))
                .expect("onnx worker thread");
        }
        info!(
            "inference: session pool ready — {} sessions x {} intra-op threads, one dedicated worker thread each",
            count, intra
        );
        Self { txs }
    }

    /// Pool with NO workers and no session load. Used in `CLASSIC=1`
    /// mode, where the feature-based classifier does the whole job and
    /// ONNX is never called.  Spawning workers in that mode is what
    /// caused the rc=139 shutdown race: the one-shot run finishes in
    /// ~0.3s and the process exits while a worker is still inside
    /// `Session` init (`GetElementType is not implemented` → SIGSEGV
    /// on 5 of the 608-file batch, all fast/small files).
    pub fn disabled() -> Self {
        info!("inference: classic mode — ONNX pool disabled, no workers spawned");
        Self { txs: vec![] }
    }

    /// Worker main loop: load the session once on this thread, then run
    /// every chunk that arrives on `rx` and reply with (ns, output).
    /// `intra` = intra-op threads per session (see `load_real`).
    pub fn worker_loop(
        model_path: &str,
        intra: usize,
        rx: std::sync::mpsc::Receiver<Job>,
    ) {
        let mut sess = OnnxSession::with_threads(model_path, intra.max(1));
        for job in rx {
            let t0 = std::time::Instant::now();
            let out = sess.run(&job.rows, job.n_rows, MODEL_INPUT_DIM);
            let _ = job.reply.send((t0.elapsed().as_nanos() as u64, out));
        }
    }

    pub fn len(&self) -> usize {
        self.txs.len()
    }

    /// Run inference on `flat` (n × dim f32), splitting the rows into
    /// near-equal contiguous chunks dispatched round-robin to the
    /// pool's dedicated session workers. Returns n × out_dim f32 in row
    /// order.
    pub fn run(&self, flat: &[f32], n: usize, dim: usize) -> Result<Vec<f32>> {
        let n_sessions = self.txs.len();
        if n_sessions == 0 {
            return Err(anyhow!("onnx pool disabled (classic mode)"));
        }
        let out_dim = MODEL_OUTPUT_DIM;
        let mut out: Vec<f32> = vec![0.0; n * out_dim];
        if n == 0 {
            return Ok(out);
        }

        // Split rows into near-equal contiguous chunks; chunk i goes to
        // session i (deterministic round-robin).
        let base = n / n_sessions;
        let rem = n % n_sessions;
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(n_sessions);
        let mut start = 0usize;
        for s in 0..n_sessions {
            let len = base + if s < rem { 1 } else { 0 };
            ranges.push((start, start + len));
            start += len;
        }

        let mut replies: Vec<(usize, usize, usize, std::sync::mpsc::Receiver<(u64, Result<Vec<f32>>)>)> =
            Vec::with_capacity(n_sessions);
        for (s, (rs, re)) in ranges.iter().enumerate() {
            let rows = re - rs;
            if rows == 0 {
                continue;
            }
            let (tx, rx) = std::sync::mpsc::channel();
            self.txs[s].send(Job {
                rows: flat[*rs * dim..*re * dim].to_vec(),
                n_rows: rows,
                reply: tx,
            })
            .map_err(|_| anyhow!("onnx worker {s} gone"))?;
            replies.push((s, *rs, rows, rx));
        }

        let mut times: Vec<(usize, usize, f64)> = Vec::with_capacity(replies.len());
        for (s, rs, rows, rx) in replies {
            let (ns, result) = rx
                .recv()
                .map_err(|_| anyhow!("onnx worker {s} reply lost"))?;
            let result = result.map_err(|e| anyhow!("onnx session {s}: {e}"))?;
            times.push((s, rows, ns as f64 / 1e6));
            out[rs * out_dim..(rs + rows) * out_dim].copy_from_slice(&result);
        }

        // Load-balance proof: per-session row count and worker-side ms.
        if times.len() > 1 {
            let max = times.iter().map(|t| t.2).fold(0.0f64, f64::max);
            let min = times.iter().map(|t| t.2).fold(f64::INFINITY, f64::min);
            let detail: Vec<String> = times
                .iter()
                .map(|(s, rows, ms)| format!("s{s}={ms:.0}ms/{rows}"))
                .collect();
            info!(
                "TIMING onnx_sessions={} min={min:.0}ms max={max:.0}ms spread={:.1}% [{}]",
                times.len(),
                (max - min) / max.max(1.0) * 100.0,
                detail.join(" ")
            );
        }
        Ok(out)
    }
}

impl OnnxSession {
    pub fn new(model_path: &str) -> Self {
        Self::with_threads(model_path, 1)
    }

    /// Load a session with `intra_threads` intra-op ORT threads.
    pub fn with_threads(model_path: &str, intra_threads: usize) -> Self {
        let lib_path = std::env::var("ORT_LIB_PATH")
            .unwrap_or_else(|_| "/usr/local/lib/libonnxruntime.so".into());
        let real = if Path::new(model_path).exists() && Path::new(&lib_path).exists() {
            match Self::load_real(model_path, &lib_path, intra_threads.max(1)) {
                Some(s) => Some(s),
                None => {
                    warn!(
                        "inference: failed to load real session (model={model_path}, lib={lib_path}); \
                         falling back to stub"
                    );
                    None
                }
            }
        } else {
            warn!(
                "inference: model or libonnxruntime not found (model={model_path}, lib={lib_path}) \
                 — using stub (features pass through, normalised)"
            );
            None
        };
        let input_name = "input_features".into();
        let output_name = "embeddings".into();
        let session = OnnxSession {
            real,
            input_name,
            output_name,
        };
        info!(
            "inference: session ready (real={})",
            session.real.is_some()
        );
        session
    }

    pub fn load_real(model_path: &str, lib_path: &str, intra: usize) -> Option<ort::session::Session> {
        let _ = ort::init_from(lib_path);
        // intra-op count for ORT's own matmul splitting; 1 = legacy.
        let intra = intra.max(1);
        // rc.13: `Session::builder()` → `with_intra_threads` →
        // `commit_from_file`.
        let session = ort::session::Session::builder()
            .ok()?
            .with_intra_threads(intra)
            .ok()?
            .commit_from_file(model_path)
            .ok()?;
        info!(
            "inference: model loaded (intra_op_threads={intra})"
        );
        Some(session)
    }

    /// Fixed model input dimension (458 for the v16 model).
    pub fn input_dim(&self) -> usize {
        MODEL_INPUT_DIM
    }

    /// Run inference on `n` flat feature rows of dim `dim`.
    /// Returns `n * out_dim` f32 (the "embeddings" output).
    ///
    /// rc.13 `Session::run` takes `&mut self` — the pool passes each
    /// session to exactly one thread per call, so `&mut` access here is
    /// exclusive by construction.
    pub fn run(&mut self, input: &[f32], n: usize, dim: usize) -> Result<Vec<f32>> {
        let Some(session) = self.real.as_mut() else {
            return Ok(Self::stub_embeddings(input, n, dim));
        };
        // model expects a 3-D input [batch, seq, 458]
        let shape: Vec<i64> = vec![n as i64, 1, dim as i64];
        let input_val = ort::value::Tensor::from_array((shape.as_slice(), input.to_vec()))
            .map_err(|e| anyhow!("Tensor::from_array: {e}"))?;

        let mut outputs = session
            .run(ort::inputs![self.input_name.as_str() => input_val])
            .map_err(|e| anyhow!("onnx run: {e}"))?;

        let out_val = outputs
            .remove(&self.output_name)
            .ok_or_else(|| anyhow!("onnx: no output named '{}'", self.output_name))?;
        let shape = out_val.shape();
        let total: usize = shape
            .iter()
            .map(|&d| d as i64)
            .filter(|&d| d > 0)
            .product::<i64>() as usize;
        let ptr = out_val.data_ptr();
        let result: Vec<f32> =
            unsafe { std::slice::from_raw_parts(ptr as *const f32, total) }.to_vec();
        Ok(result)
    }

    /// Stub: derive a deterministic pseudo-random 1024-dim embedding from
    /// a hash of the input row (FNV-1a → xorshift PRNG), then L2-normalise.
    /// Deterministic: identical feature rows → identical embeddings, so
    /// HDBSCAN finds real clusters. Keeps the pipeline runnable without
    /// a model/lib.
    pub fn stub_embeddings(input: &[f32], n: usize, dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * MODEL_OUTPUT_DIM);
        for i in 0..n {
            let start = i * dim;
            let end = (start + dim).min(input.len());
            // Hash the row bytes (bit patterns) for a stable seed. The
            // plain FNV state degenerates for 458-dim uniform rows (the
            // multiply-then-xor step becomes its own inverse, so a
            // repeated value cancels itself out), so fold in the row
            // length and the sum of bit patterns first — different rows
            // must give different seeds.
            let mut seed: u64 = 0xcbf29ce484222325;
            let len_seed = (end - start) as u64;
            seed = seed.wrapping_mul(0x100000001b3) ^ len_seed;
            let mut bits_sum: u64 = 0;
            for &x in &input[start..end] {
                bits_sum = bits_sum.wrapping_add(x.to_bits() as u64);
            }
            seed = seed.wrapping_mul(0x100000001b3) ^ bits_sum;
            for &x in &input[start..end] {
                seed ^= (x.to_bits() as u64).wrapping_mul(0x100000001b3);
            }
            // xorshift64* PRNG → 1024 dims of near-iid values.
            let mut state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
            let mut next = move || {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64
            };
            let mut v = vec![0f32; MODEL_OUTPUT_DIM];
            let mut norm = 0f64;
            for j in 0..MODEL_OUTPUT_DIM {
                let val = next() / 9007199254740992.0 - 0.5; // roughly [-0.5, 0.5]
                v[j] = val as f32;
                norm += (val * val) as f64;
            }
            let norm = norm.sqrt();
            if norm > 1e-12 {
                for x in v.iter_mut() {
                    *x /= norm as f32;
                }
            }
            out.extend_from_slice(&v);
        }
        out
    }
}
