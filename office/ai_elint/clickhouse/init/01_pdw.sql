-- 1M/s tuning for ClickHouse: async inserts, bigger blocks, parallel merges.
-- Server-level settings are applied via users.d/00_perf.xml (mounted to
-- /etc/clickhouse-server/config.d) — SET GLOBAL in initdb.d is not allowed.
CREATE DATABASE IF NOT EXISTS signals;

CREATE TABLE IF NOT EXISTS signals.pdw
(
    id                       UInt64,
    power_dbm                Float32,
    center_frequency_hz      Float64,
    time_of_arrival_ns       String,
    duration_ns              Float64,
    bandwidth_hz             Float64,
    direction_of_arrival_deg Float32,
    snr_db                   Float32,
    modulation               LowCardinality(String),
    source                   LowCardinality(String),
    is_cw                    UInt8,
    receiver_id              UInt8 DEFAULT 1
)
ENGINE = MergeTree
ORDER BY (id);

CREATE TABLE IF NOT EXISTS signals.clusters
(
    cluster_id    String,
    emitter_id    String,
    signal_id     String,
    pdw_count     UInt32,
    mean_freq_hz  Float64,
    mean_power    Float32,
    pdw_ids       Array(UInt64),
    created_at    DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (cluster_id);
