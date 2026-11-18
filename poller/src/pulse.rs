//! Pulse struct and xorshift64 generator — no external RNG crate.
//! Optimized for 1M+ records/s: minimal allocations, u64 toa_ns, u8 modulation enum.

use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

/// Modulation type as a compact u8 enum.
#[derive(Serialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Modulation {
    Psk,
    Fsk,
    Am,
    Pm,
    Cw,
    Unknown,
    Qam,
}

impl std::fmt::Display for Modulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modulation::Psk => write!(f, "PSK"),
            Modulation::Fsk => write!(f, "FSK"),
            Modulation::Am => write!(f, "AM"),
            Modulation::Pm => write!(f, "PM"),
            Modulation::Cw => write!(f, "CW"),
            Modulation::Unknown => write!(f, "UNKNOWN"),
            Modulation::Qam => write!(f, "QAM"),
        }
    }
}

pub const MODULATIONS: [Modulation; 7] = [
    Modulation::Psk,
    Modulation::Fsk,
    Modulation::Am,
    Modulation::Pm,
    Modulation::Cw,
    Modulation::Unknown,
    Modulation::Qam,
];

#[derive(Serialize)]
pub struct Pulse {
    pub id: u64,
    pub power_dbm: f32,
    pub center_frequency_hz: f64,
    pub time_of_arrival_ns: u64,
    pub duration_ns: f64,
    pub bandwidth_hz: f64,
    pub direction_of_arrival_deg: f32,
    pub snr_db: f32,
    pub modulation: Modulation,
    pub is_cw: u8,
    pub receiver_id: u8,
}

/// xorshift64 — deterministic, no external dependency.
pub struct Rng(u64);

impl Rng {
    pub fn next_f64(&mut self) -> f64 {
        let s = &mut self.0;
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s % 1_000_000) as f64 / 1_000_000.0
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
    pub fn next_modulation(&mut self) -> Modulation {
        let idx = (self.next_f64() * MODULATIONS.len() as f64) as usize % MODULATIONS.len();
        MODULATIONS[idx]
    }
}

pub struct PulseGenerator {
    rng: Rng,
    next_id: u64,
    total: u64,
    base_ns: u64,
}

impl PulseGenerator {
    pub fn new(total: u64) -> Self {
        let base_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self {
            rng: Rng(42),
            next_id: 1,
            total,
            base_ns,
        }
    }
}

impl Iterator for PulseGenerator {
    type Item = Pulse;

    #[inline]
    fn next(&mut self) -> Option<Pulse> {
        if self.next_id > self.total {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;

        let rng = &mut self.rng;
        Some(Pulse {
            id,
            power_dbm: rng.range(-80.0, -20.0) as f32,
            center_frequency_hz: rng.range(1.0e8, 1.2e10),
            time_of_arrival_ns: (self.base_ns + id * 1000) % 1_000_000_000_000,
            duration_ns: rng.range(500.0, 50_000.0),
            bandwidth_hz: rng.range(1.0e5, 1.0e8),
            direction_of_arrival_deg: rng.range(0.0, 360.0) as f32,
            snr_db: rng.range(10.0, 40.0) as f32,
            modulation: rng.next_modulation(),
            is_cw: if rng.next_f64() < 0.1 { 1 } else { 0 },
            receiver_id: 1,
        })
    }
}
