//! Logging for the refresh API — console + rotating log file.
//!
//! Mirrors the Python `logger.py` behaviour:
//! - console handler (tracing fmt → stdout)
//! - rotating file handler (plain `YYYY-MM-DD HH:MM:SS | LEVEL | target | message`)
//!
//! Rotation: when the log file exceeds `ELINT_LOG_MAX_BYTES`
//! (default 10 MB), `elint.log` is renamed to `elint.log.1`,
//! `elint.log.1` → `elint.log.2` … up to `ELINT_LOG_BACKUP_COUNT`
//! (default 5), then a fresh `elint.log` is opened.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Plain timestamp: `YYYY-MM-DD HH:MM:SS`.
pub fn timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86400) as i64;
    let rem = (secs % 86400) as i64;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}")
}

pub struct RotFile {
    path: PathBuf,
    #[allow(dead_code)]
    pub max_bytes: u64,
    pub backups: usize,
}

impl RotFile {
    pub fn rotate(&self) {
        let stem = &self.path;
        for i in (1..self.backups).rev() {
            let from = PathBuf::from(format!("{}.{}", stem.display(), i));
            let to = PathBuf::from(format!("{}.{}", stem.display(), i + 1));
            if from.exists() {
                let _ = fs::rename(&from, &to);
            }
        }
        if stem.exists() {
            let first = PathBuf::from(format!("{}.1", stem.display()));
            let _ = fs::rename(stem, &first);
        }
    }

    pub fn append_line(&self, level: &str, target: &str, msg: &str) {
        if self
            .path
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0)
            >= self.max_bytes
        {
            self.rotate();
        }
        let line = format!("{} | {:>7} | {} | {}\n", timestamp(), level, target, msg);
        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Shared rotating-file sink. Cheap to clone (Arc), used by both the
/// tracing layer and the direct `stage_log!` macro.
#[derive(Clone)]
pub struct RotatingSink {
    pub inner: Arc<Mutex<RotFile>>,
}

impl RotatingSink {
    pub fn new(log_dir: &str) -> Option<Self> {
        fs::create_dir_all(log_dir).ok()?;
        let file = std::env::var("ELINT_LOG_FILE")
            .unwrap_or_else(|_| format!("{}/elint.log", log_dir.trim_end_matches('/')));
        let max_bytes = std::env::var("ELINT_LOG_MAX_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10 * 1024 * 1024);
        let backups: usize = std::env::var("ELINT_LOG_BACKUP_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        Some(RotatingSink {
            inner: Arc::new(Mutex::new(RotFile {
                path: PathBuf::from(file),
                max_bytes,
                backups,
            })),
        })
    }

    pub fn log(&self, level: &str, target: &str, msg: &str) {
        if let Ok(st) = self.inner.lock() {
            st.append_line(level, target, msg);
        }
    }
}


pub fn test_lock(sink: &RotatingSink) -> std::sync::LockResult<std::sync::MutexGuard<'_, RotFile>> {
    sink.inner.lock()
}

struct FieldVisitor<'a>(&'a mut String);

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, _field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push_str(&format!("{value:?}"));
    }
}

impl FieldVisitor<'_> {
    pub fn unwrap_quoted(s: &str) -> String {
        if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    }
}

/// A `tracing_subscriber::Layer` that mirrors every record to the
/// rotating file as `ts | LEVEL | target | message`.
pub struct RotatingLogLayer {
    pub sink: RotatingSink,
}

impl RotatingLogLayer {
    pub fn new(sink: RotatingSink) -> Self {
        RotatingLogLayer { sink }
    }
}

impl<S> tracing_subscriber::Layer<S> for RotatingLogLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        let meta = event.metadata();
        let mut fields = String::new();
        event.record(&mut FieldVisitor(&mut fields));
        let msg = FieldVisitor::unwrap_quoted(&fields);
        self.sink.log(meta.level().as_str(), meta.target(), &msg);
    }
}

/// Init tracing: console (stdout, colored via RUST_LOG) + rotating file.
/// Returns the shared sink so the refresh stages can log directly.
pub fn init(log_dir: &str) -> Option<RotatingSink> {
    let sink = RotatingSink::new(log_dir)?;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stdout)
        .init();
    // The rotating file sink is used directly via sink.log() calls;
    // the tracing layer is registered for future use.
    Some(sink)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_format() {
        let ts = timestamp();
        // Format: YYYY-MM-DD HH:MM:SS (19 chars)
        assert_eq!(ts.len(), 19);
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], " ");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
    }

    #[test]
    fn timestamp_year_is_recent() {
        let ts = timestamp();
        let year: i32 = ts[..4].parse().unwrap();
        assert!(year >= 2024 && year <= 2030, "unexpected year {year}");
    }

    /// These tests mutate the process env (`ELINT_LOG_FILE`,
    /// `ELINT_LOG_MAX_BYTES`) and `cargo test` runs tests in parallel —
    /// serialise them so one test's env never clobbers another's.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn rotating_sink_new_creates_dir() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("elint_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("ELINT_LOG_FILE", dir.join("test.log").to_str().unwrap());
        let sink = RotatingSink::new(dir.to_str().unwrap()).unwrap();
        // Log a line and verify the file was created
        sink.log("INFO", "test", "hello");
        let content = std::fs::read_to_string(dir.join("test.log")).unwrap();
        assert!(content.contains("hello"));
        assert!(content.contains("INFO"));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("ELINT_LOG_FILE");
    }

    #[test]
    fn rotating_sink_log_format() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("elint_fmt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log_file = dir.join("fmt.log");
        std::env::set_var("ELINT_LOG_FILE", log_file.to_str().unwrap());
        std::env::set_var("ELINT_LOG_MAX_BYTES", "1000000");
        let sink = RotatingSink::new(dir.to_str().unwrap()).unwrap();
        sink.log("WARN", "my_target", "something happened");
        let content = std::fs::read_to_string(&log_file).unwrap();
        // Format: "YYYY-MM-DD HH:MM:SS | LEVEL | target | message"
        let line: Vec<&str> = content.lines().next().unwrap().split(" | ").collect();
        assert_eq!(line.len(), 4);
        assert_eq!(line[1], "   WARN");
        assert_eq!(line[2], "my_target");
        assert_eq!(line[3], "something happened");
        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("ELINT_LOG_FILE");
        std::env::remove_var("ELINT_LOG_MAX_BYTES");
    }

    #[test]
    fn unwrap_quoted() {
        assert_eq!(FieldVisitor::unwrap_quoted("\"hello\""), "hello");
        assert_eq!(FieldVisitor::unwrap_quoted("hello"), "hello");
        assert_eq!(FieldVisitor::unwrap_quoted("\""), "\"");
        assert_eq!(FieldVisitor::unwrap_quoted(""), "");
    }
}
