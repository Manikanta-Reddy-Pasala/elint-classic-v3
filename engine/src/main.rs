//! Engine — the single ELINT pipeline binary.
#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
//! Engine — the single ELINT pipeline binary.//!
//! Reads pulses directly from ClickHouse `signals.pdw` (CDC poller in
//! process), routes each pulse to its RF band (S/L/X) and runs the full
//! AI stage in-process via the `ai_processor` library: 458-dim features
//! → ONNX embedding → known/unknown match → HDBSCAN + agglomerative
//! merge → direct writes to Postgres + ClickHouse.
//!
//! No Kafka/Redpanda, no Go — the whole pipeline is one Rust binary.
//!
//! Thread layout (all tunable via env):
//!   - READ_THREADS    (default 10): parallel ClickHouse read tasks
//!   - PROCESS_WORKERS (default 10): worker threads

mod band;
mod ch_read;
mod config;



use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::ch_read::{fetch_limit, host_port, Pulse};
use crate::config::Config;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    // Pin the rayon pool (feature extraction, HDBSCAN k-NN, merge matrix)
    // to every core.
    rayon::ThreadPoolBuilder::new()
        .num_threads(cores)
        .build_global()
        .expect("rayon pool");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(
            std::env::var("TOKIO_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(cores)
        )
        .enable_all()
        .build()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let cfg = Config::from_env();
    info!(
        "engine starting: read_threads={} process_workers={}",
        cfg.read_threads, cfg.process_workers
    );

    // ── shared row-offset cursor (rows already delivered) ───────────────
    // Each reader advances this by the number of rows it just read, so
    // together they walk through the table in row-offset order.
    // This is robust regardless of how `id` is assigned.
    let cursor = Arc::new(Mutex::new(0u64));

    // ── Global-mode (E2E_MATCH=1): collect all embeddings/unknowns, then
    //    run ONE global HDBSCAN at the end — matches the old Python
    //    pipeline's per-file clustering output exactly. ─────────────────
    let global_mode = std::env::var("E2E_MATCH").is_ok();
    if global_mode {
        info!("engine: GLOBAL-MATCH mode — single global HDBSCAN at the end");
    }

    // ── shared AI state (ONNX session, PG tracks, cache) ──────────────
    let t_startup = Instant::now();
    let ai_cfg = ai_processor::config::Config {
        onnx_path: std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "/models/transformer_model_known_v16.onnx".into()),
        onnx_session_count: std::env::var("ONNX_SESSION_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2),
        onnx_threads: std::env::var("ONNX_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6),
        known_threshold: cfg.known_threshold,
        angle_gate: 40.0,
        ema_alpha: 0.1,
        batch_size: std::env::var("BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000),
        min_cluster_size: std::env::var("MIN_CLUSTER_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5),
        // Python processes in 512-pulse windows (PDW_CHUNK_SIZE=512) and
        // clusters ALL unknowns of each window. In windowed mode the
        // worker runs the AI stage per 512-window, so each HDBSCAN sees
        // a full window (<=512) — max_samples=0 means "no cap".
        max_samples: std::env::var("MAX_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        pg_url: cfg.pg_url.clone(),
        ch_url: cfg.ch_url.clone(),
        onnx_batch_size: std::env::var("ONNX_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024),
        cluster_batch_size: std::env::var("CLUSTER_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024),
    };
    // Verify PG connection before starting the engine.
    tracing::info!("engine: verifying PG connection to {}", cfg.pg_url);
    let pg_ok = ai_processor::db::test_pg(&cfg.pg_url).await;
    if !pg_ok {
        anyhow::bail!("PG not reachable at {} — cannot start", cfg.pg_url);
    }
    let ai_state = Arc::new(ai_processor::AiState::new(ai_cfg.clone()).await?);
    let t_startup_ms = t_startup.elapsed().as_secs_f64() * 1000.0;
    let rayon_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    info!(
        "TIMING engine_startup={:.0}ms (onnx_sessions={} onnx_threads={} rayon_threads={} pg_conns={})",
        t_startup_ms,
        ai_cfg.onnx_session_count,
        ai_cfg.onnx_threads,
        rayon_threads,
        std::env::var("PG_CONNS").unwrap_or_else(|_| "4".into())
    );

    // ── E2E mode: exit after E2E_MAX_PULSES rows ───────────────────────
    let seen_counter = Arc::new(AtomicU64::new(0));
    let e2e_max: Option<u64> = std::env::var("E2E_MAX_PULSES").ok().and_then(|v| v.parse().ok());
    // In global-match mode the readers drain the whole table once (no
    // new rows ever appear), so the worker simply finishes and the
    // readers exit on empty batches — no premature exit needed.
    if let Some(max) = e2e_max {
        if !global_mode {
            info!("engine: E2E mode — will exit after {max} pulses");
            let counter = seen_counter.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    if counter.load(std::sync::atomic::Ordering::SeqCst) >= max {
                        info!("engine: E2E max reached, shutting down");
                        std::process::exit(0);
                    }
                }
            });
        } else {
            info!("engine: E2E global-match mode — max {max} informational; runs to end of table");
        }
    }

    // ── input channel: readers push batches, workers pull ──────────────
    let (tx, rx) = mpsc::unbounded_channel::<Vec<Pulse>>();
    let mut handles = Vec::new();

    // ── reader tasks ───────────────────────────────────────────────────
    // In global-match mode there is one consumer, so only reader 0
    // pulls data; the others stay parked to avoid N× duplication.
    {
        let cfg = cfg.clone();
        let cursor = cursor.clone();
        let tx = tx.clone();
        let gm = global_mode;
        handles.push(tokio::spawn(async move {
            if let Err(e) = run_readers(cfg, cursor, tx, gm).await {
                warn!("readers stopped: {e}");
            }
        }));
    }

    // ── worker task (single consumer drains the channel) ───────────────
    // In global-match mode the worker does a long, CPU-heavy finalize
    // (HDBSCAN + medoids + merge + DB writes).  If we spawn it as a
    // tokio task on the multi-threaded runtime, its blocking
    // `std::thread::join()` for the finalize thread starves every
    // other runtime thread — including the reader task that needs to
    // close the channel so the worker's `rx.recv()` returns None.
    // Instead, run the worker on its own dedicated std::thread and
    // block on it here; the reader task (tokio) runs independently.
    let (host, port) = host_port(&cfg.ch_url);
    let url = format!("http://{host}:{port}");
    let bands = crate::band::specs_from_env();

    if global_mode {
        // Global-match: worker on its own std::thread, readers on tokio.
        //
        // The worker blocks on `rx.recv()` (async, on its own
        // current-thread runtime).  The reader task holds `tx`; when
        // it drains the table and exits it drops `tx`, so
        // `rx.recv()` returns None and the worker proceeds to
        // finalize.  We must drop main's `tx` clone first so there
        // are no extra senders keeping the channel open.
        //
        // We do NOT await the reader task before joining the worker
        // thread: `worker_handle.join()` blocks this thread, which
        // would starve the tokio runtime (including the reader task)
        // on the main thread.  Instead we use a oneshot channel to
        // signal when the worker thread is done, letting the tokio
        // runtime keep running the reader task meanwhile.
        let bands_g = bands.clone();
        let seen_g = seen_counter.clone();
        let ai_state_g = ai_state.clone();
        let url_g = url.clone();
        let rx_g = rx;
        drop(tx); // drop main's clone so only the reader's tx remains

        let (worker_done_tx, worker_done_rx) = tokio::sync::oneshot::channel::<Result<()>>();
        let wdt: std::sync::Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<Result<()>>>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Some(worker_done_tx)));
        let wdt2 = wdt.clone();

        let worker_handle = std::thread::spawn(move || -> Result<()> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("worker rt");
            let result = rt.block_on(run_worker_global(0, rx_g, ai_state_g, bands_g, seen_g, url_g));
            // Signal the main tokio runtime that the worker is done.
            if let Some(tx) = wdt2.lock().unwrap().take() {
                let _ = tx.send(result);
            }
            Ok(())
        });

        // Await the worker-done signal.  The reader task (tokio) runs
        // on the multi-threaded runtime in parallel; it will finish
        // (and drop its tx) on its own, which lets the worker's
        // rx.recv() return None and the worker proceed to finalize.
        match worker_done_rx.await {
            Ok(Ok(())) => info!("engine: global-match worker done"),
            Ok(Err(e)) => warn!("worker 0: {e}"),
            Err(_) => warn!("worker 0: oneshot dropped"),
        }
        // Make sure the worker thread has actually exited.
        let _ = worker_handle.join();
        // Reap the reader task (it has already finished by now).
        let reader_handle = handles.pop().expect("reader handle");
        let _ = reader_handle.await;
        info!("engine: global-match mode complete — shutting down");
    } else {
        // Normal mode: worker as a tokio task.
        let ai_state_w = ai_state.clone();
        let bands_w = bands.clone();
        let seen_w = seen_counter.clone();
        handles.push(tokio::spawn(async move {
            if let Err(e) = run_worker(0, rx, ai_state_w, bands_w, seen_w, url).await {
                warn!("worker 0: {e}");
            }
        }));
        drop(tx);
    }

    // ── wait for shutdown (normal mode only; global mode already waited above) ──
    if !global_mode {
        signal::ctrl_c().await?;
        info!("engine: ctrl-c received, shutting down");
        for h in handles.iter_mut() {
            let _ = h.await;
        }
    }
    Ok(())
}

/// Parallel ClickHouse readers pulling batches of FETCH_LIMIT rows.
///
/// Each reader owns ONE persistent `ChConn` (keep-alive TCP socket) so
/// every batch reuses the same connection — no per-call handshake.
/// In normal mode all readers race ahead (double-buffer).  In
/// global-match mode only reader 0 pulls data (one consumer).
async fn run_readers(
    cfg: Config,
    cursor: Arc<Mutex<u64>>,
    tx: mpsc::UnboundedSender<Vec<Pulse>>,
    global_mode: bool,
) -> Result<()> {
    let (host, port) = host_port(&cfg.ch_url);
    let url = format!("http://{host}:{port}");

    // One shared persistent connection for the table count + pre-check.
    let probe_url = url.clone();
    let probe = std::thread::spawn(move || {
        let conn = crate::ch_read::ChConn::connect(&probe_url);
        match conn {
            Ok(c) => {
                let total = c.query_scalar("SELECT count() FROM signals.pdw");
                let pre = c.read_batch_offset(0, 3);
                (c, total, pre)
            }
            Err(e) => {
                eprintln!("engine: CH connect failed: {e}");
                std::process::exit(1);
            }
        }
    })
    .join()
    .unwrap();

    // Move the probe conn into reader 0's slot (reader 0 reuses it).
    // Other readers get their own connections.
    let mut total: u64 = match &probe.1 {
        Ok(s) => s.parse().unwrap_or(0),
        Err(e) => {
            warn!("engine: pdw count failed: {e}");
            0
        }
    };
    // The CH count query can silently fail (auth 403, transient) while
    // the data read works.  If the pre-check read returned rows, trust
    // that the table is non-empty and let the readers drain it — the
    // empty-batch check stops them anyway.
    if total == 0 {
        if let Ok(p) = &probe.2 {
            if !p.is_empty() {
                info!(
                    "engine: count said 0 but pre-check read {} rows — count query unreliable, continuing",
                    p.len()
                );
                total = u64::MAX;
            }
        }
    }
    info!("engine: pdw table has {total} rows");
    match probe.2 {
        Ok(p) => info!("engine: CH pre-check ok, {} rows", p.len()),
        Err(e) => warn!("engine: pre-check CH read failed: {e}; readers will retry"),
    }

    // Reader 0 reuses the probe connection; the rest connect fresh.
    let probe_conn = std::sync::Arc::new(std::sync::Mutex::new(probe.0));

    let mut handles = Vec::new();
    for r in 0..cfg.read_threads {
        let url = url.clone();
        let tx = tx.clone();
        let cursor = cursor.clone();
        let fetch_limit = fetch_limit();
        let conn_init: std::sync::Arc<std::sync::Mutex<crate::ch_read::ChConn>> = if r == 0 {
            probe_conn.clone()
        } else {
            std::sync::Arc::new(std::sync::Mutex::new(
                crate::ch_read::ChConn::connect(&url)
                    .unwrap_or_else(|e| {
                        eprintln!("reader {r}: connect failed: {e}");
                        std::process::exit(1);
                    }),
            ))
        };

        handles.push(tokio::spawn(async move {
            let mut offset = 0u64;
            loop {
                // Use the shared row-offset cursor as our starting point.
                let shared = *cursor.lock().unwrap();
                if shared > offset {
                    offset = shared;
                }
                // In global-match (E2E_MATCH) mode there is only ONE
                // consumer, so the other readers stay parked.  Skip
                // their reads entirely — reader 0 drains the table.
                if r != 0 {
                    // In global-match mode the other readers do nothing.
                    // Exit immediately (dropping their tx clone) so the
                    // outer run_readers can finish and close the channel.
                    info!("reader {r}: parked (global mode), exiting");
                    break;
                }
                // Whole table already delivered (or empty) — stop without
                // another pointless query.
                if offset >= total {
                    info!("reader {r}: cursor {offset} >= total {total}, stopping");
                    break;
                }
                let t_read = Instant::now();
                let batch: Result<Vec<Pulse>> = tokio::task::spawn_blocking({
                    let conn = conn_init.clone();
                    let off = offset;
                    move || {
                        let c = conn.lock().unwrap();
                        c.read_batch_offset(off, fetch_limit)
                    }
                })
                .await
                .unwrap_or_else(|_| Err(anyhow!("reader {r}: task panicked")));

                // ClickHouse pads the trailing partial page to the
                // page size — trim any phantom rows past the true count.
                let e2e_max = std::env::var("E2E_MAX_PULSES")
                    .ok()
                    .and_then(|v| v.parse().ok());
                let batch = match batch {
                    Ok(mut b) => {
                        let _ = e2e_max.and_then(|max| {
                            if (offset + b.len() as u64) > max {
                                let keep = (max - offset) as usize;
                                if keep < b.len() {
                                    info!(
                                        "reader {r}: trimming {} phantom rows (table has {max})",
                                        b.len() - keep
                                    );
                                    b.truncate(keep);
                                }
                            }
                            None::<()>
                        });
                        b
                    }
                    Err(e) => {
                        warn!("reader {r}: read failed: {e}; backing off 500ms, reconnecting");
                        if let Ok(c) = conn_init.lock().unwrap().reconnect() {
                            let _ = c;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        continue;
                    }
                };
                if batch.is_empty() {
                    // Static E2E table: once we have read the whole
                    // table (an empty batch proves the cursor is past
                    // the end), stop polling — this reader is done.
                    info!("reader {r}: table drained ({offset} rows), stopping");
                    break;
                }
                let n = batch.len() as u64;
                let read_ms = t_read.elapsed().as_secs_f64() * 1000.0;
                info!(
                    "TIMING reader {r}: ch_read {:.1}ms, {} rows (offset={})",
                    read_ms,
                    batch.len(),
                    offset + n
                );
                offset += n;
                *cursor.lock().unwrap() = offset;
                if tx.send(batch).is_err() {
                    info!("reader {r}: channel closed, stopping");
                    break;
                }
            }
            Ok::<_, anyhow::Error>(())
        }));
    }
    for h in handles {
        let _ = h.await;
    }
    Ok(())
}

/// Global-match worker (E2E_MATCH=1): runs the AI stage WITHOUT the
/// per-batch HDBSCAN, collects the unknown embeddings, and at the end
/// of the file runs ONE global HDBSCAN + merge + DB writes so the
/// cluster output matches the old Python pipeline.
async fn run_worker_global(
    id: usize,
    mut rx: mpsc::UnboundedReceiver<Vec<Pulse>>,
    ai_state: Arc<ai_processor::AiState>,
    bands: Vec<crate::band::BandSpec>,
    seen_counter: Arc<AtomicU64>,
    _url: String,
) -> Result<()> {
    let mut seen = 0u64;
    // (embeddings, pulse_ids, input_ids) per batch — ids are needed at
    // finalize time so cluster rows in CH can carry the real pulse set.
    let mut unknown_batches: Vec<ai_processor::global_match::BufferedBatch> = Vec::new();
    let t_engine_start = Instant::now();
    let mut batch_count = 0u32;
    loop {
        let batch = match rx.recv().await {
            Some(b) => b,
            None => break,
        };
        seen += batch.len() as u64;
        seen_counter.store(seen, std::sync::atomic::Ordering::SeqCst);
        batch_count += 1;

        let band_counts: std::collections::HashMap<String, usize> = batch
            .iter()
            .filter_map(|p| crate::band::for_freq(&bands, p.center_frequency_hz))
            .fold(std::collections::HashMap::new(), |mut m, b| {
                *m.entry(b.as_str().to_string()).or_insert(0) += 1;
                m
            });
        let band_str: Vec<String> = band_counts
            .iter()
            .map(|(b, n)| format!("{b}={n}"))
            .collect();
        info!("worker {id}: bands: {}", band_str.join(", "));

        let ai_pulses: Vec<ai_processor::AiPulse> = batch
            .iter()
            .map(|p| ai_processor::AiPulse {
                id: p.id as i64,
                center_frequency_hz: p.center_frequency_hz,
                power_dbm: p.power_dbm as f64,
                snr_db: p.snr_db as f64,
                duration_ns: p.duration_ns,
                bandwidth_hz: p.bandwidth_hz as f64,
                time_of_arrival_ns: p.time_of_arrival_ns as f64,
                direction_of_arrival_deg: p.direction_of_arrival_deg as f64,
                modulation: p.modulation.clone(),
                emitter_label: None,
            })
            .collect();
        let t_ai = Instant::now();
        match ai_processor::run_batch_no_cluster(&ai_state, &ai_pulses).await {
            Ok(unk) => {
                if !unk.is_empty() {
                    let meta: Vec<ai_processor::global_match::AiPulseMeta> = ai_pulses
                        .iter()
                        .map(|p| ai_processor::global_match::AiPulseMeta {
                            id: p.id,
                            center_frequency_hz: p.center_frequency_hz,
                            power_dbm: p.power_dbm,
                            snr_db: p.snr_db,
                            duration_ns: p.duration_ns,
                            bandwidth_hz: p.bandwidth_hz,
                            time_of_arrival_ns: p.time_of_arrival_ns,
                            direction_of_arrival_deg: p.direction_of_arrival_deg,
                            modulation: p.modulation.clone(),
                            pulse_repetition_interval_ns: 0.0,
                            emitter_label: None,
                        })
                        .collect();
                    let input_ids: Vec<i64> = ai_pulses.iter().map(|p| p.id).collect();
                    unknown_batches.push(ai_processor::global_match::BufferedBatch {
                        embeddings: unk,
                        meta,
                        input_ids,
                        predictions: None,
                    });
                }
            }
            Err(e) => warn!("worker {id}: ai batch failed: {e}"),
        }
        info!(
            "worker {id}: ai_batch_no_cluster {:.1}ms",
            t_ai.elapsed().as_secs_f64() * 1000.0
        );

        if seen % 10_000 < batch.len() as u64 {
            info!("engine worker {id}: {seen} pulses processed");
        }
    }

    // ── End of file: ONE global HDBSCAN over all unknowns ─────────────
    info!(
        "worker {id}: file done, {seen} pulses total — running global HDBSCAN over {} unknown batches",
        unknown_batches.len()
    );
    // Run finalize directly on THIS current-thread runtime.
    // The runtime's blocking pool is available (we are not inside a
    // spawn_blocking call here), so finalize's inner spawn_blocking
    // for PG/CH writes works correctly.
    info!("worker {id}: calling global_match::finalize ...");
    let t_global = Instant::now();
    match ai_processor::global_match::finalize(&ai_state, &unknown_batches).await {
        Ok(()) => info!("worker {id}: finalize done in {:.1}s", t_global.elapsed().as_secs_f64()),
        Err(e) => warn!("worker {id}: global match failed: {e}"),
    }
    info!(
        "worker {id}: all done, {seen} pulses total, {} batches, engine_wall={:.0}ms",
        batch_count,
        t_engine_start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// One worker: band-route the batch, then run the AI stage on a
/// blocking thread for each 512-pulse window — matching the Python
/// reference (`PDW_CHUNK_SIZE=512`) which processes 512 pulses per
/// ONNX → known/unknown → cluster cycle.
///
/// The 512 window keeps the per-batch HDBSCAN input small (O(n²) k-NN
/// on 512 rows instead of O(10,000)²) and makes the unknown-pulse
/// clustering match the Python pipeline's per-window clustering.
async fn run_worker(
    id: usize,
    mut rx: mpsc::UnboundedReceiver<Vec<Pulse>>,
    ai_state: Arc<ai_processor::AiState>,
    bands: Vec<crate::band::BandSpec>,
    seen_counter: Arc<AtomicU64>,
    url: String,
) -> Result<()> {
    let mut seen = 0u64;
    let window: usize = std::env::var("WINDOW_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512); // Python's PDW_CHUNK_SIZE
    if window != 512 {
        info!("worker {id}: WINDOW_SIZE={window} (non-default)");
    }
    loop {
        let batch = match rx.recv().await {
            Some(b) => b,
            None => break,
        };
        seen += batch.len() as u64;
        seen_counter.store(seen, std::sync::atomic::Ordering::SeqCst);

        // Band split (S/L/X) — the routing step of the architecture.
        let band_counts: std::collections::HashMap<String, usize> = batch
            .iter()
            .filter_map(|p| crate::band::for_freq(&bands, p.center_frequency_hz))
            .fold(std::collections::HashMap::new(), |mut m, b| {
                *m.entry(b.as_str().to_string()).or_insert(0) += 1;
                m
            });
        let band_str: Vec<String> = band_counts
            .iter()
            .map(|(b, n)| format!("{b}={n}"))
            .collect();
        info!("worker {id}: bands: {}", band_str.join(", "));

        // Split the read batch into 512-pulse windows (Python's PDW_CHUNK_SIZE)
        // and run SEVERAL windows concurrently — each on its own blocking
        // thread with its own current-thread runtime. Windows touch
        // independent state (ONNX pool round-robins sessions, PG writes
        // round-robin connections, HDBSCAN uses the shared rayon pool),
        // so they run in parallel. Concurrency is capped by AI_CONCURRENCY
        // (default 4) so the 24-core box is not oversubscribed.
        let ai_concurrency: usize = std::env::var("AI_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
            .max(1);
        let mut total_ai_ms = 0.0f64;
        let windows: Vec<(usize, &[Pulse])> =
            batch.chunks(window).enumerate().collect();
        for wgroup in windows.chunks(ai_concurrency) {
            let t_group = Instant::now();
            let mut handles = Vec::with_capacity(wgroup.len());
            for (w, window) in wgroup {
                let ai_pulses: Vec<ai_processor::AiPulse> = window
                    .iter()
                    .map(|p| ai_processor::AiPulse {
                        id: p.id as i64,
                        center_frequency_hz: p.center_frequency_hz,
                        power_dbm: p.power_dbm as f64,
                        snr_db: p.snr_db as f64,
                        duration_ns: p.duration_ns,
                        bandwidth_hz: p.bandwidth_hz as f64,
                        time_of_arrival_ns: p.time_of_arrival_ns as f64,
                        direction_of_arrival_deg: p.direction_of_arrival_deg as f64,
                        modulation: p.modulation.clone(),
                        emitter_label: None,
                    })
                    .collect();
                let ai_state2 = ai_state.clone();
                let w = *w;
                handles.push(std::thread::spawn(move || -> Result<(), (usize, String)> {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("tokio rt");
                    rt.block_on(ai_processor::run_batch(&ai_state2, &ai_pulses))
                        .map_err(|e| (w, e.to_string()))
                }));
            }
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err((w, e))) => warn!("worker {id}: ai window {w} failed: {e}"),
                    Err(_) => warn!("worker {id}: ai window thread panicked"),
                }
            }
            total_ai_ms += t_group.elapsed().as_secs_f64() * 1000.0;
        }
        info!(
            "worker {id}: ai_batch {:.1}ms ({} windows of {})",
            total_ai_ms,
            (batch.len() + window - 1) / window,
            window
        );

        if seen % 10_000 < batch.len() as u64 {
            info!("engine worker {id}: {seen} pulses processed");
        }
    }
    info!("worker {id}: all done, {seen} pulses total");
    Ok(())
}
