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
        let cols: Vec<String> = split_csv(&head);
        let ix = |n: &str| cols.iter().position(|c| c == n);
        let (it, ip, ifr, ib, id, ido, ity, isn, ie) = (
            ix("time_of_arrival_ns").unwrap(),
            ix("power_dbm").unwrap(),
            ix("center_frequency_hz").unwrap(),
            ix("bandwidth_hz").unwrap(),
            ix("duration_ns").unwrap(),
            ix("direction_of_arrival_deg").unwrap(),
            ix("type").or(ix("modulation")).unwrap(),
            ix("snr_db"),
            ix("emitter_id"),
        );
        let num = |s: &str| s.parse::<f64>().unwrap_or(f64::NAN);
        let toa = |s: &str| s.parse::<f64>().unwrap_or_else(|_| parse_ts_ns(s));
        let mut pulses = Vec::new();
        let mut truth = Vec::new();
        for (row, l) in lines.enumerate() {
            let l = l?;
            let vs = split_csv(&l);
            let v: Vec<&str> = vs.iter().map(|x| x.as_str()).collect();
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
                toa_ns: toa(v[it]),
                modulation: v[ity].to_ascii_lowercase(),
            });
            truth.push(ie.and_then(|i| v[i].parse::<f64>().ok()).map(|x| x as i64).unwrap_or(-9));
        }
        // real captures: absolute timestamps -> relative to the first pulse
        let t0n = pulses.iter().map(|p| p.toa_ns).fold(f64::INFINITY, f64::min);
        if t0n.is_finite() && t0n > 1e15 {
            for p in pulses.iter_mut() {
                p.toa_ns -= t0n;
            }
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
            let mut dv: Vec<f64> = ps.iter().map(|p| p.doa_deg).filter(|x| x.is_finite()).collect();
            dv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| if dv.is_empty() { f64::NAN } else { dv[((dv.len() - 1) as f64 * f) as usize] };
            let mut fv: Vec<f64> = ps.iter().map(|p| p.freq_hz / 1e6).collect();
            fv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let fq = |f: f64| fv[((fv.len() - 1) as f64 * f) as usize];
            out.push(serde_json::json!({"lib": lib, "n": idx.len(), "truth": t, "fmode": freq_mode(&ps),
                "doa_q10_50_90": [q(0.1), q(0.5), q(0.9)], "f_q5_95": [fq(0.05), fq(0.95)]}));
        }
        let mut tt: BTreeMap<i64, usize> = BTreeMap::new();
        for &e in &truth {
            *tt.entry(e).or_default() += 1;
        }
        println!("{}", serde_json::json!({"file": path, "pulses": pulses.len(), "ms": ms, "truth_counts": tt, "emitters": out}));
    }
    Ok(())
}

/// Minimal CSV split: commas outside double quotes; quotes stripped.
fn split_csv(l: &str) -> Vec<String> {
    let (mut out, mut cur, mut q) = (Vec::new(), String::new(), false);
    for ch in l.chars() {
        match ch {
            '"' => q = !q,
            ',' if !q => out.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

/// "YYYY-MM-DD HH:MM:SS.fffffffff[+00:00]" -> ns since 1970 (UTC, no tz shift).
fn parse_ts_ns(s: &str) -> f64 {
    let s = s.trim();
    let (d, t) = match s.split_once(|c| c == ' ' || c == 'T') {
        Some(x) => x,
        None => return f64::NAN,
    };
    let dp: Vec<i64> = d.split('-').filter_map(|x| x.parse().ok()).collect();
    if dp.len() != 3 {
        return f64::NAN;
    }
    let t = t.split(|c| c == '+' || c == 'Z').next().unwrap_or("");
    let tp: Vec<&str> = t.split(':').collect();
    if tp.len() != 3 {
        return f64::NAN;
    }
    let (y, m, dd) = (dp[0] - (dp[1] <= 2) as i64, dp[1], dp[2]);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + dd - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let (sec, frac) = tp[2].split_once('.').unwrap_or((tp[2], "0"));
    let frac_ns: f64 = format!("{:0<9}", &frac[..frac.len().min(9)]).parse().unwrap_or(0.0);
    let secs = days * 86400 + tp[0].parse::<i64>().unwrap_or(0) * 3600 + tp[1].parse::<i64>().unwrap_or(0) * 60 + sec.parse::<i64>().unwrap_or(0);
    secs as f64 * 1e9 + frac_ns
}
