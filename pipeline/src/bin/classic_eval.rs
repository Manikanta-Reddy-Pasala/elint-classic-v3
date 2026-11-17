//! Offline classic runner for experiments: CSV -> classic detect -> one JSON
//! line per file. The engine never sees `emitter_id`; it is read here ONLY to
//! report which true emitters each found group holds (scoring).
//!
//! usage: classic_eval <file.csv>...
use ai_processor::classic_v2::{detect, freq_mode, ClassicCfg};
use ai_processor::onnx_flow::FlowPulse;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let cfg = ClassicCfg::from_env();
    for path in std::env::args().skip(1) {
        let f = std::fs::File::open(&path)?;
        let mut lines = BufReader::new(f).lines();
        let head = lines.next().unwrap()?;
        let cols: Vec<&str> = head.split(',').collect();
        let ix = |n: &str| cols.iter().position(|c| *c == n);
        let (it, ip, ifr, ib, id, ido, ity, isn, ie) = (
            ix("time_of_arrival_ns").unwrap(),
            ix("power_dbm").unwrap(),
            ix("center_frequency_hz").unwrap(),
            ix("bandwidth_hz").unwrap(),
            ix("duration_ns").unwrap(),
            ix("direction_of_arrival_deg").unwrap(),
            ix("type").unwrap(),
            ix("snr_db"),
            ix("emitter_id").unwrap(),
        );
        let num = |s: &str| s.parse::<f64>().unwrap_or(f64::NAN);
        let mut pulses = Vec::new();
        let mut truth = Vec::new();
        for (row, l) in lines.enumerate() {
            let l = l?;
            let v: Vec<&str> = l.split(',').collect();
            if v.len() < cols.len() {
                continue;
            }
            pulses.push(FlowPulse {
                input_id: row as i64 + 1,
                freq_hz: num(v[ifr]),
                power_dbm: num(v[ip]),
                bandwidth_hz: num(v[ib]),
                duration_ns: num(v[id]),
                doa_deg: num(v[ido]),
                snr_db: isn.map(|i| num(v[i])).unwrap_or(0.0),
                toa_ns: num(v[it]),
                modulation: v[ity].to_string(),
            });
            truth.push(v[ie].parse::<f64>().map(|x| x as i64).unwrap_or(-9));
        }
        let t0 = Instant::now();
        let em = detect(&pulses, &cfg);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut out = Vec::new();
        for (lib, idx) in &em {
            let mut t: BTreeMap<i64, usize> = BTreeMap::new();
            for &i in idx {
                *t.entry(truth[i]).or_default() += 1;
            }
            let ps: Vec<&FlowPulse> = idx.iter().map(|&i| &pulses[i]).collect();
            out.push(serde_json::json!({"lib": lib, "n": idx.len(), "truth": t, "fmode": freq_mode(&ps)}));
        }
        let mut tt: BTreeMap<i64, usize> = BTreeMap::new();
        for &e in &truth {
            *tt.entry(e).or_default() += 1;
        }
        println!("{}", serde_json::json!({"file": path, "pulses": pulses.len(), "ms": ms, "truth_counts": tt, "emitters": out}));
    }
    Ok(())
}
