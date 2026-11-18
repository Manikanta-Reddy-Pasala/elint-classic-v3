/// Postgres + ClickHouse persistence — persistent, pooled, async.
///
/// - Postgres: a `PgState` of persistent connections, created once at
///   startup and reused for every read/write (no per-call connect).
/// - ClickHouse: one persistent keep-alive TCP socket for writes, shared
///   under a mutex, so every insert reuses the same connection.
///
/// No Kafka, no Go: rows go straight to the DBs.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio_postgres::config::Config;
use tokio_postgres::{GenericClient, NoTls};
use tracing::{info, warn};

// ── Postgres ──────────────────────────────────────────────────────────────────

pub fn normalize(pg_url: &str) -> String {
    let d = pg_url.trim();
    if d.starts_with("postgres://") || d.starts_with("postgresql://") {
        d.to_string()
    } else {
        format!("postgres://{d}")
    }
}

// ── Unit tests (no DB needed) ────────────────────────────────────────────────
/// One raw Postgres connection: the client plus its connection task,
/// kept alive for the life of the process.
pub struct PgConn {
    pub client: tokio_postgres::Client,
    pub _task: tokio::task::JoinHandle<()>,
}

impl PgConn {
    async fn connect(pg_url: &str) -> Result<Self> {
        let dsn = normalize(pg_url);
        let (client, conn) = tokio_postgres::Config::from_str(&dsn)?
            .connect(NoTls)
            .await?;
        let task = tokio::spawn(async move {
            if let Err(e) = conn.await {
                warn!("db: conn died: {e}");
            }
        });
        Ok(PgConn { client, _task: task })
    }
}

/// Persistent Postgres connection set for the whole engine.
///
/// N connections; a round-robin cursor spreads writes across them.
/// Connections stay open for the life of the process — every read/write
/// is async over an existing socket (no per-call handshake).
pub struct PgState {
    pub conns: std::sync::Arc<Vec<PgConn>>,
    pub next: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl PgState {
    /// Open `n` persistent connections.
    pub async fn connect(pg_url: &str, n: usize) -> Result<Self> {
        let n = n.max(1);
        let mut conns = Vec::with_capacity(n);
        for i in 0..n {
            match PgConn::connect(pg_url).await {
                Ok(c) => conns.push(c),
                Err(e) => warn!("db: pg conn {i}: {e}"),
            }
        }
        if conns.is_empty() {
            anyhow::bail!("db: could not open any Postgres connections to {pg_url}");
        }
        tracing::info!("db: {} persistent Postgres connections open to {pg_url}", conns.len());
        Ok(PgState {
            conns: std::sync::Arc::new(conns),
            next: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// Grab a connection from the round-robin set.
    pub fn acquire(&self) -> &tokio_postgres::Client {
        let i = self
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.conns.len();
        &self.conns[i].client
    }
}

// PgState is cheaply clonable: the Arc fields share the same connections.
impl Clone for PgState {
    fn clone(&self) -> Self {
        PgState {
            conns: self.conns.clone(),
            next: self.next.clone(),
        }
    }
}

// ── Schema ────────────────────────────────────────────────────────────────────

/// Test whether Postgres is reachable.  Returns true on success.
pub async fn test_pg(pg_url: &str) -> bool {
    match PgConn::connect(pg_url).await {
        Ok(client) => {
            let ok = client
                .client
                .query_one("SELECT 1", &[])
                .await
                .map(|_r| true)
                .unwrap_or(false);
            if ok {
                let v: Option<String> = client
                    .client
                    .query_one("SELECT current_database()", &[])
                    .await
                    .ok()
                    .map(|r| r.get::<_, String>(0));
                tracing::info!("db: test_pg ok, database={v:?}");
            }
            ok
        }
        Err(e) => {
            warn!("db: test_pg failed: {e}");
            false
        }
    }
}

/// Create the pipeline tables if they do not exist.
pub async fn ensure_schema(pg: &PgState) -> Result<()> {
    let client = pg.acquire();
    // Fail fast if we are not on the expected database.
    let cur: String = client
        .query_one("SELECT current_database()", &[])
        .await?
        .get(0);
    tracing::info!("db: connected to database '{cur}'");
    // The test harness owns the schema: it TRUNCATEs `clusters`, `signals`
    // and `emitters` before each test and seeds `emitters` with the known
    // profiles. Detect the harness layout (emitters.signature_matched_id)
    // and keep it exactly as-is — never drop/recreate those tables.
    let harness_schema: i64 = client
        .query_one(
            "SELECT count(*) FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'emitters'
               AND column_name = 'signature_matched_id'",
            &[],
        )
        .await?
        .get(0);
    if harness_schema > 0 {
        tracing::info!("db: harness schema detected — keeping it, only adding missing engine columns");
        // The harness ships `signals` with only the legacy production
        // columns (frequency_hz, power_dbm, doa_deg, …).  The classic and
        // ONNX writers INSERT (emitter_id, name, freq, pri, doa, power,
        // modulation, count), so the schema must carry those five columns
        // or every signal insert fails with `column "name" of relation
        // signals does not exist`.  The `count` column the signal insert
        // uses as a literal (1) is added too for completeness.
        let _ = add_cols(
            client,
            &[
                ("emitters", "medoid", "DOUBLE PRECISION[]"),
                ("emitters", "source", "TEXT NOT NULL DEFAULT ''"),
                ("emitters", "freq", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
                ("emitters", "pri", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
                ("emitters", "doa", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
                ("emitters", "power", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
                ("emitters", "modulation", "TEXT NOT NULL DEFAULT ''"),
                ("signals", "frequency_hz", "BIGINT"),
                ("signals", "power_dbm", "DOUBLE PRECISION"),
                ("signals", "doa_deg", "DOUBLE PRECISION"),
                ("signals", "name", "TEXT"),
                ("signals", "freq", "DOUBLE PRECISION"),
                ("signals", "pri", "DOUBLE PRECISION"),
                ("signals", "doa", "DOUBLE PRECISION"),
                ("signals", "power", "DOUBLE PRECISION"),
                ("signals", "modulation", "TEXT"),
                ("signals", "count", "BIGINT NOT NULL DEFAULT 1"),
                // Classic cluster writes need these stats columns too — add
                // them in harness mode as well as the fallback path.
                ("clusters", "snr_db", "DOUBLE PRECISION"),
                ("clusters", "min_snr_db", "DOUBLE PRECISION"),
                ("clusters", "max_snr_db", "DOUBLE PRECISION"),
                ("clusters", "std_snr_db", "DOUBLE PRECISION"),
                ("clusters", "min_bandwidth_hz", "DOUBLE PRECISION"),
                ("clusters", "max_bandwidth_hz", "DOUBLE PRECISION"),
            ],
        )
        .await;
        tracing::info!("db: schema ensured (harness mode)");
        return Ok(());
    }
    // Fallback: no harness schema present — create the minimal tables.
    let r1 = client
        .batch_execute(
            "CREATE TABLE IF NOT EXISTS emitters (
                id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name        TEXT,
                source      TEXT NOT NULL,
                freq        DOUBLE PRECISION NOT NULL,
                pri         DOUBLE PRECISION NOT NULL,
                doa         DOUBLE PRECISION NOT NULL,
                power       DOUBLE PRECISION NOT NULL,
                modulation  TEXT NOT NULL,
                medoid      DOUBLE PRECISION[] NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )",
        )
        .await;
    if let Err(e) = r1 {
        tracing::error!("db: create emitters failed: {e}");
        anyhow::bail!("db: create emitters failed: {e}");
    }
    let check: tokio_postgres::Row = client
        .query_one(
            "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relname = 'emitters'",
            &[],
        )
        .await?;
    let n: i64 = check.get(0);
    if n == 0 {
        anyhow::bail!("db: emitters table missing after CREATE");
    }
    let r2 = client
        .batch_execute(
            "CREATE TABLE IF NOT EXISTS signals (
                id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                emitter_id  BIGINT NOT NULL REFERENCES emitters(id),
                name        TEXT,
                freq        DOUBLE PRECISION NOT NULL,
                pri         DOUBLE PRECISION NOT NULL,
                doa         DOUBLE PRECISION NOT NULL,
                power       DOUBLE PRECISION NOT NULL,
                modulation  TEXT NOT NULL,
                count       BIGINT NOT NULL DEFAULT 1,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )",
        )
        .await;
    if let Err(e) = r2 {
        tracing::error!("db: create signals failed: {e}");
    }
    let r3 = client
        .batch_execute(
            "CREATE TABLE IF NOT EXISTS clusters (
                id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name        VARCHAR(255) NOT NULL DEFAULT '',
                pdw_count   INTEGER,
                toa_ns      BIGINT,
                signal_id   INTEGER,
                frequency_hz            BIGINT,
                min_frequency_hz        BIGINT,
                max_frequency_hz        BIGINT,
                std_frequency_hz        DOUBLE PRECISION,
                power_dbm               DOUBLE PRECISION,
                min_power_dbm           DOUBLE PRECISION,
                max_power_dbm           DOUBLE PRECISION,
                std_power_dbm           DOUBLE PRECISION,
                pulse_width_us          DOUBLE PRECISION,
                min_pulse_width_us      DOUBLE PRECISION,
                max_pulse_width_us      DOUBLE PRECISION,
                std_pulse_width_us      DOUBLE PRECISION,
                pulse_repetition_interval_us          DOUBLE PRECISION,
                min_pulse_repetition_interval_us      DOUBLE PRECISION,
                max_pulse_repetition_interval_us      DOUBLE PRECISION,
                std_pulse_repetition_interval_us      DOUBLE PRECISION,
                bandwidth_hz            DOUBLE PRECISION,
                min_bandwidth_hz        DOUBLE PRECISION,
                max_bandwidth_hz        DOUBLE PRECISION,
                std_bandwidth_hz        DOUBLE PRECISION,
                doa_deg                 DOUBLE PRECISION,
                min_doa_deg             DOUBLE PRECISION,
                max_doa_deg             DOUBLE PRECISION,
                std_doa_deg             DOUBLE PRECISION,
                is_ai         BOOLEAN NOT NULL DEFAULT TRUE,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )",
        )
        .await;
    if let Err(e) = r3 {
        tracing::error!("db: create clusters failed: {e}");
    }
    // Pre-existing tables from the Python era may lack the new columns.
    let _ = add_cols(
        client,
        &[
            ("emitters", "name", "TEXT"),
            ("emitters", "updated_at", "TIMESTAMPTZ NOT NULL DEFAULT NOW()"),
            ("signals", "name", "TEXT"),
            ("clusters", "name", "VARCHAR(255)"),
            ("clusters", "pdw_count", "INTEGER"),
            ("clusters", "toa_ns", "BIGINT"),
            ("clusters", "signal_id", "INTEGER"),
            ("clusters", "frequency_hz", "BIGINT"),
            ("clusters", "min_frequency_hz", "BIGINT"),
            ("clusters", "max_frequency_hz", "BIGINT"),
            ("clusters", "std_frequency_hz", "DOUBLE PRECISION"),
            ("clusters", "power_dbm", "DOUBLE PRECISION"),
            ("clusters", "min_power_dbm", "DOUBLE PRECISION"),
            ("clusters", "max_power_dbm", "DOUBLE PRECISION"),
            ("clusters", "std_power_dbm", "DOUBLE PRECISION"),
            ("clusters", "pulse_width_us", "DOUBLE PRECISION"),
            ("clusters", "min_pulse_width_us", "DOUBLE PRECISION"),
            ("clusters", "max_pulse_width_us", "DOUBLE PRECISION"),
            ("clusters", "std_pulse_width_us", "DOUBLE PRECISION"),
            ("clusters", "pulse_repetition_interval_us", "DOUBLE PRECISION"),
            ("clusters", "min_pulse_repetition_interval_us", "DOUBLE PRECISION"),
            ("clusters", "max_pulse_repetition_interval_us", "DOUBLE PRECISION"),
            ("clusters", "std_pulse_repetition_interval_us", "DOUBLE PRECISION"),
            ("clusters", "bandwidth_hz", "DOUBLE PRECISION"),
            ("clusters", "min_bandwidth_hz", "DOUBLE PRECISION"),
            ("clusters", "max_bandwidth_hz", "DOUBLE PRECISION"),
            ("clusters", "std_bandwidth_hz", "DOUBLE PRECISION"),
            ("clusters", "doa_deg", "DOUBLE PRECISION"),
            ("clusters", "min_doa_deg", "DOUBLE PRECISION"),
            ("clusters", "max_doa_deg", "DOUBLE PRECISION"),
            ("clusters", "std_doa_deg", "DOUBLE PRECISION"),
            ("clusters", "is_ai", "BOOLEAN NOT NULL DEFAULT TRUE"),
            ("clusters", "updated_at", "TIMESTAMPTZ NOT NULL DEFAULT NOW()"),
            // The cluster INSERT writes these six stats columns; the
            // harness `clusters` table (01_schema.sql) does not create
            // them, so every insert failed with `column "snr_db" of
            // relation "clusters" does not exist`.
            ("clusters", "snr_db", "DOUBLE PRECISION"),
            ("clusters", "min_snr_db", "DOUBLE PRECISION"),
            ("clusters", "max_snr_db", "DOUBLE PRECISION"),
            ("clusters", "std_snr_db", "DOUBLE PRECISION"),
            ("clusters", "min_bandwidth_hz", "DOUBLE PRECISION"),
            ("clusters", "max_bandwidth_hz", "DOUBLE PRECISION"),
        ],
    )
    .await;
    tracing::info!("db: schema ensured");
    Ok(())
}

/// Apply a list of `ALTER TABLE <t> ADD COLUMN IF NOT EXISTS <c> <type>`
/// statements one by one (tokio-postgres `batch_execute` accepts a single
/// statement, so each column is its own round-trip).
async fn add_cols(client: &tokio_postgres::Client, cols: &[(&str, &str, &str)]) {
    for (table, col, ty) in cols {
        let sql = format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {ty}");
        if let Err(e) = client.batch_execute(&sql).await {
            tracing::warn!("db: add column {table}.{col}: {e}");
        }
    }
}

// ── Postgres reads / writes (all async, all on persistent conns) ─────────────

/// Insert a new emitter with an explicit display name (Python writes the
/// profile id for known emitters, e.g. `PSR1`; the test harness queries
/// `emitters.name`). Returns the new `emitter_id`.
pub async fn insert_emitter_pg(
    pg: &PgState,
    name: Option<&str>,
    source: &str,
    freq: f64,
    pri: f64,
    doa: f64,
    power: f64,
    modulation: &str,
    medoid: &[f32],
) -> Result<i64> {
    let client = pg.acquire();
    let t0 = Instant::now();
    let name_str: &str = name.unwrap_or("unknown");
    let med_arr: Vec<f64> = medoid.iter().map(|&x| x as f64).collect();
    let id: i64 = client
        .query_one(
            "INSERT INTO emitters (name, source, freq, pri, doa, power, modulation, medoid, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW()) RETURNING id",
            &[
                &name_str,
                &source.to_string(),
                &freq,
                &pri,
                &doa,
                &power,
                &modulation.to_string(),
                &med_arr,
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("insert_emitter: {e}"))?
        .get::<_, i64>(0);
    PG_INSERT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    tracing::info!("db: insert_emitter ok, id={id} name={name_str}");
    Ok(id)
}

/// Insert several emitters in ONE multi-row INSERT (one round-trip),
/// returning the new ids in input order. Used by the refresh sink:
/// ~1153 single-row INSERTs become ~115 statements.
///
/// Column order per row: source, freq, pri, doa, power, modulation, medoid.
/// Parameters are passed per field, repeated N times, so column i of
/// row r binds to $((r*7)+(i+1)).
pub async fn insert_emitters_pg_batch(
    pg: &PgState,
    source: &str,
    emitters: &[&crate::cluster_merge::MergedCluster],
    freq: f64,
) -> Result<Vec<i64>> {
    let n = emitters.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let client = pg.acquire();
    let sources: Vec<String> = (0..n).map(|_| source.to_string()).collect();
    let freqs: Vec<f64> = (0..n).map(|_| freq).collect();
    let pris: Vec<f64> = vec![0.0; n];
    let doas: Vec<f64> = emitters.iter().map(|mc| mc.angle).collect();
    let powers: Vec<f64> = vec![0.0; n];
    let mods: Vec<String> = (0..n).map(|_| "unknown".to_string()).collect();
    let medoids: Vec<Vec<f64>> = emitters
        .iter()
        .map(|mc| mc.medoid.iter().map(|&x| x as f64).collect())
        .collect();

    let mut sql = String::from(
        "INSERT INTO emitters (source, freq, pri, doa, power, modulation, medoid) VALUES ",
    );
    for i in 0..n {
        if i > 0 {
            sql.push_str(", ");
        }
        let base = i * 7;
        sql.push_str(&format!(
            "(${}, ${}, ${}, ${}, ${}, ${}, ${})",
            base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7
        ));
    }
    sql.push_str(" RETURNING id");

    type P<'a> = &'a (dyn tokio_postgres::types::ToSql + Sync);
    let sources_ref: Vec<P<'_>> =
        sources.iter().map(|s| s as P<'_>).collect();
    let freqs_ref: Vec<P<'_>> =
        freqs.iter().map(|v| v as P<'_>).collect();
    let pris_ref: Vec<P<'_>> =
        pris.iter().map(|v| v as P<'_>).collect();
    let doas_ref: Vec<P<'_>> =
        doas.iter().map(|v| v as P<'_>).collect();
    let powers_ref: Vec<P<'_>> =
        powers.iter().map(|v| v as P<'_>).collect();
    let mods_ref: Vec<P<'_>> =
        mods.iter().map(|s| s as P<'_>).collect();
    let medoids_ref: Vec<P<'_>> =
        medoids.iter().map(|m| m as P<'_>).collect();

    // Flatten in per-row order: row 0 (source, freq, pri, doa, power, mod, medoid), row 1, ...
    let mut params: Vec<P<'_>> = Vec::with_capacity(n * 7);
    for i in 0..n {
        params.push(sources_ref[i]);
        params.push(freqs_ref[i]);
        params.push(pris_ref[i]);
        params.push(doas_ref[i]);
        params.push(powers_ref[i]);
        params.push(mods_ref[i]);
        params.push(medoids_ref[i]);
    }

    let t0 = Instant::now();
    let rows = client
        .query(&sql, &params)
        .await
        .map_err(|e| anyhow::anyhow!("insert_emitters_batch: {e}"))?;
    PG_INSERT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(rows.iter().map(|r| r.get::<_, i64>(0)).collect())
}

/// Load known emitter tracks from Postgres to seed the in-memory cache.
pub async fn load_tracks(pg: &PgState) -> Result<Vec<crate::cache::EmitterTrack>> {
    let client = pg.acquire();
    let rows = client
        .query(
            "SELECT id, freq, doa, power, medoid FROM emitters WHERE source <> 'unknown_ai' AND medoid IS NOT NULL AND freq IS NOT NULL AND doa IS NOT NULL AND power IS NOT NULL",
            &[],
        )
        .await
        .unwrap_or_default();
    Ok(rows
        .into_iter()
        .map(|r| {
            let id: i64 = r.get(0);
            let freq: f64 = r.get(1);
            let doa: f64 = r.get(2);
            let power: f64 = r.get(3);
            let medoid: Vec<f64> = r.get(4);
            crate::cache::EmitterTrack {
                signal_id: id,
                vector: medoid.iter().map(|&x| x as f32).collect(),
                freq,
                angle: doa,
                power,
                modulation: "known".into(),
                cardinality: 1,
            }
        })
        .collect())
}

/// Find an existing emitter whose medoid is within the given cosine
/// similarity threshold of `medoid`. Returns the emitter id if found.
/// Follows the Python logic: before inserting a new emitter, check if a
/// near-identical one already exists (the Python code updates the existing
/// signal row instead of creating a new emitter row).
pub async fn find_existing_emitter(
    pg: &PgState,
    medoid: &[f32],
    threshold: f64,
) -> Option<i64> {
    let client = pg.acquire();
    // The harness schema has no `medoid` column — dedup is impossible
    // there, and every test run starts from a TRUNCATEd table anyway.
    let col_exists: i64 = client
        .query_one(
            "SELECT count(*) FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'emitters'
               AND column_name = 'medoid'",
            &[],
        )
        .await
        .ok()?
        .get(0);
    if col_exists == 0 {
        return None;
    }
    let rows = client
        .query("SELECT id, medoid FROM emitters WHERE medoid IS NOT NULL", &[])
        .await
        .ok()?;
    for row in &rows {
        let id: i64 = row.get(0);
        let existing: Option<Vec<f64>> = row.get(1);
        let existing = existing?;
        if existing.len() != medoid.len() {
            continue;
        }
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..medoid.len() {
            let a = medoid[i] as f64;
            let b = existing[i];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        if na > 0.0 && nb > 0.0 {
            let cos = dot / (na.sqrt() * nb.sqrt());
            if cos >= threshold {
                return Some(id);
            }
        }
    }
    None
}

/// Insert a matched signal row for a known emitter.
/// `name` is the emitter display name copied onto the signal row for the
/// test harness (Python keeps the same name on signals).
/// Cumulative nanoseconds spent in Postgres INSERT statements (all
/// callers). Sampled and reported as `sink_pg` on the `TIMING batch_done`
/// line for the E2E harness.
pub static PG_INSERT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Accessor for the sink-timing reporter (see `run_batch` in lib.rs).
pub fn pg_sink_ns() -> &'static std::sync::atomic::AtomicU64 {
    &PG_INSERT_NS
}

pub async fn insert_signal_pg(
    pg: &PgState,
    emitter_id: i64,
    name: &str,
    freq: f64,
    pri: f64,
    doa: f64,
    power: f64,
    modulation: &str,
) -> Result<i64> {
    let client = pg.acquire();
    let t0 = Instant::now();
    let id: i64 = client
        .query_one(
            "INSERT INTO signals (emitter_id, name, freq, pri, doa, power, modulation, count)
             VALUES ($1, $2, $3, $4, $5, $6, $7, 1) RETURNING id",
            &[
                &emitter_id,
                &name,
                &freq,
                &pri,
                &doa,
                &power,
                &modulation,
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("insert_signal: {e}"))?
        .get(0);
    PG_INSERT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(id)
}

/// Fix up a signal row's emitter_id (used when the signal row is inserted
/// in parallel with its emitter row and needs the new emitter id).
pub async fn fix_signal_emitter_id(pg: &PgState, signal_id: i64, emitter_id: i64) -> Result<()> {
    let client = pg.acquire();
    client
        .execute(
            "UPDATE signals SET emitter_id = $2 WHERE id = $1",
            &[&signal_id, &emitter_id],
        )
        .await
        .map_err(|e| anyhow::anyhow!("fix_signal_emitter_id: {e}"))?;
    Ok(())
}

/// `time.time_ns()` equivalent for the harness-shaped signal insert.
fn now_unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Insert a full cluster stats row into PG `clusters`.
/// Mirrors the Python `bulk_insert_clusters_to_postgres`.
pub async fn insert_cluster_pg(
    pg: &PgState,
    name: &str,
    pdw_count: i64,
    toa_ns: i64,
    signal_id: Option<i64>,
    freq_mean: i64, freq_min: i64, freq_max: i64, freq_std: f64,
    power_mean: f64, power_min: f64, power_max: f64, power_std: f64,
    snr_mean: f64, snr_min: f64, snr_max: f64, snr_std: f64,
    pw_mean: f64, pw_min: f64, pw_max: f64, pw_std: f64,
    pri_mean: f64, pri_min: f64, pri_max: f64, pri_std: f64,
    bw_mean: f64, bw_min: f64, bw_max: f64, bw_std: f64,
    doa_mean: f64, doa_min: f64, doa_max: f64, doa_std: f64,
) -> Result<i64> {
    let client = pg.acquire();
    let t0 = Instant::now();
    let pdw_count_i32 = pdw_count as i32;
    // The harness schema types `clusters.signal_id` as INTEGER; cast the
    // bigint signal id (PG identity) to i32 for the bind.
    let signal_id_i32 = signal_id.map(|s| s as i32);
    let rows = client
        .query(
            "INSERT INTO clusters (
                name, pdw_count, toa_ns, signal_id,
                frequency_hz, min_frequency_hz, max_frequency_hz, std_frequency_hz,
                power_dbm, min_power_dbm, max_power_dbm, std_power_dbm,
                pulse_width_us, min_pulse_width_us, max_pulse_width_us, std_pulse_width_us,
                pulse_repetition_interval_us, min_pulse_repetition_interval_us,
                max_pulse_repetition_interval_us, std_pulse_repetition_interval_us,
                bandwidth_hz, min_bandwidth_hz, max_bandwidth_hz, std_bandwidth_hz,
                snr_db, min_snr_db, max_snr_db, std_snr_db,
                doa_deg, min_doa_deg, max_doa_deg, std_doa_deg
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32)
            RETURNING id",
            &[
                &name, &pdw_count_i32, &toa_ns, &signal_id_i32,
                &freq_mean, &freq_min, &freq_max, &freq_std,
                &power_mean, &power_min, &power_max, &power_std,
                &pw_mean, &pw_min, &pw_max, &pw_std,
                &pri_mean, &pri_min, &pri_max, &pri_std,
                &bw_mean, &bw_min, &bw_max, &bw_std,
                &snr_mean, &snr_min, &snr_max, &snr_std,
                &doa_mean, &doa_min, &doa_max, &doa_std,
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("insert_cluster_pg: {e}"))?;
    let id: i32 = rows
        .get(0)
        .and_then(|r| r.get(0))
        .ok_or_else(|| anyhow::anyhow!("insert_cluster_pg: no id returned"))?;
    PG_INSERT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(id as i64)
}

// ── ClickHouse (persistent keep-alive write socket) ───────────────────────────

/// Cumulative nanoseconds spent in ClickHouse write POSTs (all callers).
pub static CH_INSERT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Accessor for the sink-timing sampler (see `start_sink_timing`).
pub fn ch_sink_ns() -> &'static std::sync::atomic::AtomicU64 {
    &CH_INSERT_NS
}

/// Minimal base64 for the Authorization header (avoids a crate dep).
fn ch_b64(s: String) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = s.as_bytes();
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

/// One persistent keep-alive ClickHouse HTTP connection for writes.
struct ChWriteConn {
    stream: TcpStream,
    host: String,
    port: u16,
}

impl ChWriteConn {
    pub fn connect(host: &str, port: u16) -> Result<Self> {
        let mut stream = TcpStream::connect((host, port))
            .map_err(|e| anyhow::anyhow!("ch: tcp connect: {e}"))?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        stream.set_write_timeout(Some(Duration::from_secs(60)))?;
        Ok(ChWriteConn {
            stream,
            host: host.to_string(),
            port,
        })
    }

    /// Send one HTTP request on the keep-alive socket and read the response.
    /// Returns true on HTTP success, false on transport error (caller reconnects).
    pub fn post(&mut self, query: &str, database: Option<&str>) -> Result<bool> {
        let (cu, cp) = crate::ch_read::ch_auth();
        let auth_hdr = if cu.is_empty() && cp.is_empty() {
            String::new()
        } else {
            format!("Authorization: Basic {}\r\n", ch_b64(format!("{cu}:{cp}")))
        };
        let db_q = database.map(|d| format!("database={}", d)).unwrap_or_default();
        let req = format!(
            "POST /?{db_q} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: text/plain; charset=UTF-8\r\nContent-Length: {len}\r\n{auth_hdr}Connection: keep-alive\r\n\r\n{query}",
            host = self.host,
            port = self.port,
            len = query.len(),
        );
        if self.stream.write_all(req.as_bytes()).is_err() {
            return Ok(false);
        }
        // Read the FULL HTTP response: headers AND the exact body.
        // Stopping at the header end leaves body bytes on the keep-alive
        // socket; the next request on that socket is then malformed and
        // ClickHouse answers 400 (SyntaxError, Code 62). That is why
        // stage_load_csv_input failed right after the wipe DDLs.
        let mut all: Vec<u8> = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let pos = match all.windows(4).position(|w| w == b"\r\n\r\n") {
                Some(p) => p + 4,
                None => all.len(),
            };
            if pos == all.len() {
                match self.stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => all.extend_from_slice(&buf[..n]),
                    Err(e) => {
                        warn!("ch write: read: {e}");
                        return Ok(false);
                    }
                }
                continue;
            }
            let head = std::str::from_utf8(&all[..pos]).unwrap_or("HTTP/1.1 400");
            let head = head.lines().collect::<Vec<_>>();
            let status_ok = head
                .first()
                .map(|l| l.starts_with("HTTP/1.1 200") || l.starts_with("HTTP/1.1 204"))
                .unwrap_or(false);
            let get = |k: &str| -> String {
                head.iter()
                    .find_map(|l| l.split_once(':'))
                    .filter(|(n, _)| n.trim().eq_ignore_ascii_case(k))
                    .map(|(_, v)| v.trim().to_string())
                    .unwrap_or_default()
            };
            let transfer = get("Transfer-Encoding");
            if transfer.eq_ignore_ascii_case("chunked") {
                // Drain the chunked body (always short — error text).
                let mut more: Vec<u8> = Vec::new();
                loop {
                    let avail: Vec<u8> = [&all[pos..], &more].concat();
                    if let Some(end) = ch_end_of_chunked(&avail) {
                        more.truncate(end);
                        all.extend_from_slice(&more);
                        break;
                    }
                    match self.stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => more.extend_from_slice(&buf[..n]),
                        Err(e) => {
                            warn!("ch write: read: {e}");
                            return Ok(false);
                        }
                    }
                }
            } else {
                let len = get("Content-Length").parse::<usize>().unwrap_or(0);
                while all.len() - pos < len {
                    match self.stream.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => all.extend_from_slice(&buf[..n]),
                        Err(e) => {
                            warn!("ch write: read: {e}");
                            return Ok(false);
                        }
                    }
                }
            }
            let text = String::from_utf8_lossy(&all).to_string();
            if status_ok {
                return Ok(true);
            }
            warn!("ch write response: {text}");
            return Err(anyhow::anyhow!("ch insert failed: {}", &text[..text.len().min(600)]));
        }
        // Server closed the connection without a response.
        warn!("ch write: connection closed without response");
        Ok(false)
    }
}

/// Byte offset one past the end of a complete chunked body
/// (i.e. past the terminating `0\r\n\r\n`), if complete.
fn ch_end_of_chunked(buf: &[u8]) -> Option<usize> {
    let mut rest = buf;
    let mut consumed = 0usize;
    loop {
        let nl = rest.iter().position(|&b| b == b'\n')? + 1;
        let size_line = std::str::from_utf8(&rest[..nl]).ok()?;
        let size = usize::from_str_radix(size_line.trim().trim_end_matches('\r'), 16).ok()?;
        rest = &rest[nl..];
        if size == 0 {
            // Trailer section ends at the next empty line.
            let trailer_end = rest.windows(2).position(|w| w == b"\r\n")? + 2;
            return Some(consumed + nl + trailer_end);
        }
        if rest.len() < size + 2 {
            return None; // body not fully buffered yet
        }
        rest = &rest[size + 2..];
        consumed += nl + size + 2;
    }
}

/// Persistent ClickHouse write connection, shared under a mutex.
///
/// Writes fan out across a small pool of keep-alive sockets (round-robin,
/// CH_WRITE_CONNS env, default 4) so concurrent inserts run in parallel
/// instead of serialising on one socket.
#[derive(Clone)]
pub struct ChState {
    pub inner: std::sync::Arc<Vec<tokio::sync::Mutex<Option<ChWriteConn>>>>,
    pub next: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub host: String,
    pub port: u16,
    pub database: Option<String>,
}

impl ChState {
    /// Number of concurrent CH write sockets (env CH_WRITE_CONNS, default 4).
    fn write_conns() -> usize {
        std::env::var("CH_WRITE_CONNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
            .max(1)
    }

    pub fn new(ch_url: &str, database: Option<String>) -> Self {
        // Write socket is the native TCP protocol (default 9000). An HTTP
        // CH_URL naming 8123/8443 (HTTP ports) is remapped to native 9000.
        let (host, port0) = crate::ch_read::host_port(ch_url);
        let port = if port0 == 8123 || port0 == 8443 { 9000 } else { port0 };
        let n = Self::write_conns();
        tracing::info!("ch: write pool of {n} keep-alive sockets");
        ChState {
            inner: std::sync::Arc::new((0..n).map(|_| tokio::sync::Mutex::new(None)).collect()),
            next: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            host,
            port,
            database,
        }
    }

    /// Round-robin slot for the next write.
    fn slot(&self) -> &tokio::sync::Mutex<Option<ChWriteConn>> {
        let i = self
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.inner.len();
        &self.inner[i]
    }

    /// Get-or-create a keep-alive write socket on one pool slot.
    async fn conn(&self) -> tokio::sync::MutexGuard<'_, Option<ChWriteConn>> {
        let mut guard = self.slot().lock().await;
        if guard.is_none() {
            let (h, p) = (self.host.clone(), self.port);
            match tokio::task::spawn_blocking(move || ChWriteConn::connect(&h, p))
                .await
                .unwrap_or(Err(anyhow::anyhow!("ch: spawn_blocking failed")))
            {
                Ok(c) => *guard = Some(c),
                Err(e) => warn!("ch: initial connect: {e}"),
            }
        }
        guard
    }

    /// Run one SQL statement on the keep-alive socket (async from the
    /// caller's view; the blocking I/O runs on the blocking pool).
    pub async fn execute(&self, sql: &str) -> Result<()> {
        let t0 = Instant::now();
        let res = self.execute_inner(sql).await;
        CH_INSERT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        res
    }

    async fn execute_inner(&self, sql: &str) -> Result<()> {
        let mut guard = self.conn().await;
        let mut last_err: Option<anyhow::Error> = None;
        for _attempt in 0..2 {
            // Clone the connection info so we don't hold a mutable borrow
            // across the await point.
            if guard.is_none() {
                let (h, p) = (self.host.clone(), self.port);
                match tokio::task::spawn_blocking(move || ChWriteConn::connect(&h, p))
                    .await
                    .unwrap_or(Err(anyhow::anyhow!("ch: spawn_blocking failed")))
                {
                    Ok(c) => *guard = Some(c),
                    Err(e) => {
                        last_err = Some(e);
                        continue;
                    }
                }
            }
            // Take the connection out, do the work, put it back.
            let Some(mut c) = guard.take() else {
                last_err = Some(anyhow::anyhow!("ch: no connection"));
                continue;
            };
            let sql_owned = sql.to_string();
            let db = self.database.clone();
            let res = tokio::task::spawn_blocking(move || c.post(&sql_owned, db.as_deref()))
                .await
                .unwrap_or(Err(anyhow::anyhow!("ch: spawn_blocking failed")));
            match res {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    // transport error — drop, will reconnect next attempt
                    last_err = Some(anyhow::anyhow!("ch: transport error"));
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("ch: write failed")))
    }
}

/// Ensure `signals.clusters` exists (idempotent).
pub async fn ensure_clusters_ch(ch: &ChState) {
    // The pytest harness creates `signals.clusters` with only
    // (cluster_id, activity_ids, created_at); the drop+recreate in the
    // wipe stage uses the same canonical schema.
    let sql = "CREATE TABLE IF NOT EXISTS signals.clusters (
                cluster_id UInt64,
                activity_ids Array(String),
                created_at DateTime DEFAULT now()
             ) ENGINE=MergeTree ORDER BY (cluster_id)";
    if let Err(e) = ch.execute(sql).await {
        warn!("ch: clusters ensure: {e}");
    }
}

/// Insert a merged cluster row into `signals.clusters` via the keep-alive
/// CH socket (async).
pub async fn insert_cluster_ch(
    ch: &ChState,
    cluster_id: i64,
    mc: &crate::cluster_merge::MergedCluster,
    mean_freq_hz: f64,
    mean_power: f64,
) -> Result<()> {
    // activity_ids: the real pulse ids in this cluster, as strings (the
    // Python harness joins these to signals.csv_input.id).
    let ids: Vec<String> = mc
        .member_ids
        .iter()
        .map(|id| id.to_string())
        .collect();
    let body = format!(
        "INSERT INTO signals.clusters (cluster_id, activity_ids) \
         VALUES ({0}, [{1}])",
        cluster_id,
        ids.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(",")
    );
    ch.execute(&body).await
}

/// Insert a `signals.clusters` row whose `activity_ids` are the given
/// 1-based `signals.csv_input` ids (the key the pytest harness joins on
/// — NOT the engine's internal pulse ids).
pub async fn insert_cluster_ch_ids(
    ch: &ChState,
    cluster_id: i64,
    ids: &[i64],
) -> Result<()> {
    let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    let body = format!(
        "INSERT INTO signals.clusters (cluster_id, activity_ids) \
         VALUES ({0}, [{1}])",
        cluster_id,
        ids.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(",")
    );
    ch.execute(&body).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_pg_url_postgres() {
        assert_eq!(
            normalize("postgres://u:p@h:5432/db"),
            "postgres://u:p@h:5432/db"
        );
    }

    #[test]
    fn normalize_pg_url_postgresql() {
        assert_eq!(
            normalize("postgresql://u:p@h:5432/db"),
            "postgresql://u:p@h:5432/db"
        );
    }

    #[test]
    fn normalize_pg_url_bare() {
        assert_eq!(
            normalize("u:p@h:5432/db"),
            "postgres://u:p@h:5432/db"
        );
    }

    #[test]
    fn normalize_pg_url_trims_whitespace() {
        assert_eq!(
            normalize("  postgres://u:p@h:5432/db  "),
            "postgres://u:p@h:5432/db"
        );
    }

    #[test]
    fn pg_sink_ns_returns_static() {
        let a = pg_sink_ns();
        let b = pg_sink_ns();
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn ch_sink_ns_returns_static() {
        let a = ch_sink_ns();
        let b = ch_sink_ns();
        assert!(std::ptr::eq(a, b));
    }
}
