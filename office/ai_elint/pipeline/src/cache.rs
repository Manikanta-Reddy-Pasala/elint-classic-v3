/// In-memory known-emitter cache.
///
/// Previously backed by Dragonfly (Redis protocol).  Dragonfly has been
/// removed from the stack — tracks now live only in process memory and are
/// re-seeded from Postgres on restart.

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Known emitter track — medoid vector + metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitterTrack {
    pub signal_id: i64,
    pub vector: Vec<f32>,
    pub freq: f64,
    pub angle: f64,
    pub power: f64,
    pub modulation: String,
    pub cardinality: i64,
}

/// In-memory cache of known emitter tracks.
pub struct Cache {
    pub tracks: DashMap<i64, EmitterTrack>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            tracks: DashMap::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn load_track(&self, signal_id: i64) -> Option<EmitterTrack> {
        match self.tracks.get(&signal_id) {
            Some(t) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(t.value().clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub fn save_track(&self, track: &EmitterTrack) {
        self.tracks.insert(track.signal_id, track.clone());
    }

    /// EMA-update the medoid vector for a known track.
    pub fn ema_update_track(&self, signal_id: i64, embedding: &[f32], alpha: f32) {
        if let Some(mut t) = self.tracks.get_mut(&signal_id) {
            let alpha = alpha.min(1.0).max(0.0);
            for (d, &nv) in embedding.iter().enumerate().take(t.vector.len()) {
                t.vector[d] = alpha * nv + (1.0 - alpha) * t.vector[d];
            }
            t.cardinality += 1;
        }
    }

    pub fn load_all_tracks(&self) -> Vec<EmitterTrack> {
        self.tracks.iter().map(|t| t.value().clone()).collect()
    }

    pub fn save(&self) {
        // No persistence — the cache is rebuilt from Postgres on restart.
    }

    pub fn start_heartbeat(&self) {}

    /// Log cache stats on shutdown.
    pub fn log_stats(&self) {
        info!(
            "cache: shutdown stats — tracks={}, hits={}, misses={}",
            self.tracks.len(),
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed)
        );
    }
}
