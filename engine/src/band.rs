//! Band routing — S / L / X frequency bands (ported from Go `band` pkg).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    X,
    S,
    L,
}

impl Band {
    pub fn as_str(&self) -> &'static str {
        match self {
            Band::X => "x",
            Band::S => "s",
            Band::L => "l",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandSpec {
    pub name: Band,
    pub min: f64,
    pub max: f64,
}

pub fn specs_default() -> Vec<BandSpec> {
    vec![
        BandSpec { name: Band::L, min: 1.0e9, max: 2.0e9 },
        BandSpec { name: Band::S, min: 2.0e9, max: 4.0e9 },
        BandSpec { name: Band::X, min: 8.0e9, max: 12.0e9 },
    ]
}

/// Read ELINT_BANDS env: "name:min:max;name:min:max;..."
pub fn specs_from_env() -> Vec<BandSpec> {
    let v = std::env::var("ELINT_BANDS").unwrap_or_default();
    if v.is_empty() {
        return specs_default();
    }
    let mut specs = Vec::new();
    for part in v.split(';') {
        let seg: Vec<&str> = part.split(':').collect();
        if seg.len() != 3 {
            continue;
        }
        let name = match seg[0].trim().to_lowercase().as_str() {
            "x" => Band::X,
            "s" => Band::S,
            "l" => Band::L,
            _ => continue,
        };
        let Ok(lo) = seg[1].parse::<f64>() else { continue };
        let Ok(hi) = seg[2].parse::<f64>() else { continue };
        specs.push(BandSpec { name, min: lo, max: hi });
    }
    if specs.is_empty() {
        specs_default()
    } else {
        specs
    }
}

pub fn for_freq(specs: &[BandSpec], freq_hz: f64) -> Option<Band> {
    for s in specs {
        if freq_hz >= s.min && freq_hz < s.max {
            return Some(s.name);
        }
    }
    None
}
