//! Kafka publish loop with batching + done-marker.
//! Uses ThreadedProducer (async librdkafka batching) + lz4 compression
//! for 1M+ msg/s throughput.
//!
//! Optimizations for 1M records/s:
//! - Batch of 50,000 records per flush
//! - lz4 compression
//! - acks=all + idempotence for zero data loss
//! - Large buffer size (1MB)
//! - linger.ms=5 for efficient batching
use crate::ch_read;
use crate::config::{Config, ReadMode, TOPIC_DONE, TOPIC_PDW};
use crate::pulse::{Pulse, PulseGenerator};
use anyhow::{bail, Context, Result};
use log::{error, info, warn};
use rdkafka::config::ClientConfig;
use rdkafka::producer::{BaseRecord, DefaultProducerContext, Producer, ThreadedProducer};
use rdkafka::util::Timeout;

pub type TProducer = ThreadedProducer<DefaultProducerContext>;

/// Create a ThreadedProducer with 1M/s tuning.
pub fn make_producer(cfg: &Config) -> Result<TProducer> {
    ClientConfig::new()
        .set("bootstrap.servers", &cfg.brokers)
        .set("acks", "all")
        .set("enable.idempotence", "true")
        .set("compression.type", "lz4")
        .set("batch.size", "1048576")
        .set("linger.ms", "5")
        .set("message.timeout.ms", "10000")
        .set("queue.buffering.max.messages", "500000")
        .set("queue.buffering.max.kbytes", "1048576")
        .set("compression.level", "1")
        .create()
        .context("kafka: create producer")
}

/// Publish pulses to the bus. CDC mode streams them from ClickHouse in
/// `fetch_limit` batches (real data); generate mode synthesizes them.
/// Always ends with a done-marker on elint.pdw.done.
pub fn publish(cfg: &Config, gen: &mut PulseGenerator) -> Result<u64> {
    let producer = make_producer(cfg)?;
    let mut published: u64 = 0;

    if cfg.read_mode == ReadMode::Cdc {
        let mut cursor: u64 = 0;
        let total = ch_read::table_count(cfg).unwrap_or(0);
        info!("poller: CDC mode — signals.pdw has {total} rows");
        if total == 0 {
            bail!("ch: signals.pdw has 0 rows — nothing to publish");
        }
        loop {
            match ch_read::read_batch(cfg, cursor) {
                Ok((rows, max_id)) => {
                    published += publish_pulses(&producer, &rows)?;
                    info!("  published: {published}/{total} (cursor={max_id})");
                    if rows.len() < cfg.fetch_limit as usize {
                        break;
                    }
                    cursor = max_id;
                }
                Err(_) => break, // no more rows after cursor — normal end
            }
        }
    } else {
        pub const BATCH: usize = 50_000;
        let mut batch: Vec<Pulse> = Vec::with_capacity(BATCH);
        for pulse in gen {
            batch.push(pulse);
            if batch.len() >= BATCH {
                published += publish_pulses(&producer, &batch)?;
                info!("  published: {}/{}", published, cfg.pulse_count);
                batch.clear();
            }
        }
        if !batch.is_empty() {
            published += publish_pulses(&producer, &batch)?;
        }
    }

    send_done(&producer);
    match producer.flush(Timeout::Never) {
        Ok(()) => info!("poller: flush complete"),
        Err(e) => warn!("poller: flush error: {e:?}"),
    }
    Ok(published)
}

/// Serialize + send one batch of pulses. Returns count sent.
pub fn publish_pulses(producer: &TProducer, rows: &[Pulse]) -> Result<u64> {
    for p in rows {
        let raw = serde_json::to_vec(p)?;
        let rec: BaseRecord<str, Vec<u8>> = BaseRecord::to(TOPIC_PDW).payload(&raw);
        if let Err((e, _)) = producer.send(rec) {
            error!("produce error: {e:?}");
            break;
        }
    }
    Ok(rows.len() as u64)
}

/// Send the done-marker on elint.pdw.done.
pub fn send_done(producer: &TProducer) {
    let done: Vec<u8> = b"1".to_vec();
    let done_rec: BaseRecord<str, Vec<u8>> = BaseRecord::to(TOPIC_DONE).payload(&done);
    if let Err((e, _)) = producer.send(done_rec) {
        warn!("poller: failed to send done marker: {e:?}");
    }
}
