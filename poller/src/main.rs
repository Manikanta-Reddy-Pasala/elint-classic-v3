//! CDC poller — pull 1M PDWs/s from ClickHouse `signals.pdw` into the bus.
//!
//! Modes (CH_CDC env):
//!   - `1`   (default): CDC — read stored rows from ClickHouse, publish to Kafka
//!   - `0`            : legacy synthetic generator (no ClickHouse)
//!   - `live`      : data simulator — continuously INSERT pulses into
//!     `signals.pdw` while publishing them to Kafka in the same run.
//!     Simulates a live capture feed; the CDC poller can read the rows
//!     back from the table in parallel.

use std::sync::Arc;

use anyhow::Result;
use log::info;

mod ch_read;
mod config;
mod publish;
mod pulse;
mod simulator;




use config::{Config, ReadMode};
use pulse::PulseGenerator;
use simulator::{insert_batch_sync, SimCounters};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cfg = Config::from_env();
    info!("mode={:?} pulse_count={} ch_url={} fetch_limit={}",
        cfg.read_mode, cfg.pulse_count, cfg.ch_url, cfg.fetch_limit);

    match cfg.read_mode {
        ReadMode::Cdc      => { let _ = publish::publish(&cfg, &mut PulseGenerator::new(cfg.pulse_count))?; }
        ReadMode::Generate => { let _ = publish::publish(&cfg, &mut PulseGenerator::new(cfg.pulse_count))?; }
        ReadMode::Live     => { run_live(&cfg)?; }
    }
    Ok(())
}

/// Live simulator: N threads each synthesize a slice of the pulse range,
/// insert them into ClickHouse `signals.pdw`, then publish the same pulses
/// to Kafka.  The done-marker is sent after all threads finish.
fn run_live(cfg: &Config) -> Result<()> {
    let ntasks = cfg.num_sim_tasks.max(1) as u64;
    let per_task = cfg.pulse_count.div_ceil(ntasks);
    let counters = Arc::new(SimCounters::new());
    let cfg_owned: Config = cfg.clone();

    info!("[simulator] live mode: {ntasks} tasks, ~{per_task} pulses each");

    let mut handles = Vec::with_capacity(ntasks as usize);
    for t in 0..ntasks {
        let cfg_t = cfg_owned.clone();
        let counters = counters.clone();
        let producer = Arc::new(publish::make_producer(&cfg_t).unwrap());
        let start_id = t * per_task + 1;
        let end_id   = ((t + 1) * per_task).min(cfg.pulse_count);

        handles.push(std::thread::spawn(move || -> Result<()> {
            sim_worker(&cfg_t, &counters, &producer, start_id, end_id)
        }));
    }

    for h in handles {
        let res = h.join()
            .map_err(|e| anyhow::anyhow!("sim worker panicked: {:?}", e))?;
        if let Err(e) = res {
            log::error!("sim worker error: {e}");
        }
    }

    publish::send_done(&publish::make_producer(cfg)?);
    let (ins, pub_) = counters.summary();
    info!("[simulator] done: inserted={ins}  published={pub_}");
    Ok(())
}

fn sim_worker(
    cfg: &Config,
    counters: &SimCounters,
    producer: &Arc<publish::TProducer>,
    start_id: u64,
    end_id: u64,
) -> Result<()> {
    // Generate pulses for this task's id range [start_id, end_id].
    // PulseGenerator::new(total) yields ids 1..=total, so we skip to start_id.
    let mut gen = PulseGenerator::new(end_id);
    for _ in 0..(start_id - 1) {
        let _ = gen.next();
    }

    let batch_size = cfg.fetch_limit as usize;
    let mut batch: Vec<pulse::Pulse> = Vec::with_capacity(batch_size);

    loop {
        batch.clear();
        for _ in 0..batch_size {
            match gen.next() {
                Some(p) if p.id <= end_id => batch.push(p),
                _ => break,
            }
        }
        if batch.is_empty() {
            break;
        }

        // 1. Insert into ClickHouse signals.pdw (simulates the live capture feed)
        let n = insert_batch_sync(cfg, &batch)?;
        counters.add_inserted(n as u64);

        // 2. Publish the same batch to Kafka elint.pdw
        let published = publish::publish_pulses(producer, &batch)?;
        counters.add_published(published);
    }
    Ok(())
}
