CREATE TABLE IF NOT EXISTS emitters (
    id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name VARCHAR(255) NOT NULL,
    signature_matched_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name VARCHAR(255) NOT NULL DEFAULT '',
    pdw_count INTEGER DEFAULT 0,
    toa_ns BIGINT,
    signal_id INTEGER,
    frequency_hz BIGINT,
    min_frequency_hz BIGINT,
    max_frequency_hz BIGINT,
    std_frequency_hz DOUBLE PRECISION,
    power_dbm DOUBLE PRECISION,
    min_power_dbm DOUBLE PRECISION,
    max_power_dbm DOUBLE PRECISION,
    std_power_dbm DOUBLE PRECISION,
    pulse_width_us DOUBLE PRECISION,
    min_pulse_width_us DOUBLE PRECISION,
    max_pulse_width_us DOUBLE PRECISION,
    std_pulse_width_us DOUBLE PRECISION,
    pulse_repetition_interval_us DOUBLE PRECISION,
    min_pulse_repetition_interval_us DOUBLE PRECISION,
    max_pulse_repetition_interval_us DOUBLE PRECISION,
    std_pulse_repetition_interval_us DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    frequency_hz BIGINT,
    bandwidth_hz BIGINT,
    receiver_id INTEGER DEFAULT 1,
    power_dbm DOUBLE PRECISION,
    snr_db DOUBLE PRECISION,
    doa_deg DOUBLE PRECISION,
    emitter_id INTEGER,
    is_ai BOOLEAN DEFAULT TRUE,
    modulation_type_id INTEGER,
    toa_nanoseconds BIGINT,
    duration_ms DOUBLE PRECISION,
    emission_type VARCHAR(32) DEFAULT 'PULSE',
    toa_seconds TIMESTAMPTZ DEFAULT NOW()
);

-- Seed 10 known emitters
INSERT INTO emitters (name, signature_matched_id) VALUES
    ('PSR_1', 22), ('PSR_2', 23), ('PSR_3', 24), ('PSR_4', 25),
    ('PSR_5_Mod1', 26), ('PSR_5_Mod2', 28), ('MF_03', 32),
    ('MF_05', 33), ('EW_03', 29), ('MF_06', 38)
ON CONFLICT DO NOTHING;
