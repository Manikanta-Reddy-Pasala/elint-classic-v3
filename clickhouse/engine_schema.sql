-- ClickHouse schema the engine works with. Loaded by ./elint.sh (one statement per ';' line).
CREATE DATABASE IF NOT EXISTS signals
;
CREATE TABLE IF NOT EXISTS signals.clusters
(
    `cluster_id` UInt64,
    `activity_ids` Array(String),
    `created_at` DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY cluster_id
SETTINGS index_granularity = 8192
;
CREATE TABLE IF NOT EXISTS signals.csv_input
(
    `event_id` UInt64,
    `power_dbm` Float64,
    `center_frequency_hz` Float64,
    `bandwidth_hz` Float64,
    `duration_ns` Float64,
    `elevation_deg` Float32,
    `pulse_repetition_interval_ns` Float64,
    `emitter_id` UInt32,
    `source` String,
    `dtoa_ns` Int64,
    `snr_db` Float64,
    `sidelobe_blanking_flag_as_int8` Int64,
    `time_of_arrival_ns` Int64,
    `direction_of_arrival_deg` Float64,
    `type` String,
    `id` Int64
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8192
;
CREATE TABLE IF NOT EXISTS signals.pdw
(
    `id` UInt64,
    `power_dbm` Float32,
    `center_frequency_hz` Float64,
    `time_of_arrival_ns` String,
    `duration_ns` Float64,
    `bandwidth_hz` Float64,
    `direction_of_arrival_deg` Float32,
    `snr_db` Float32,
    `modulation` LowCardinality(String),
    `source` LowCardinality(String),
    `is_cw` UInt8,
    `receiver_id` UInt8 DEFAULT 1
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8192
;
