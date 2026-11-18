CREATE TABLE IF NOT EXISTS elint_signals (
    signal_id   VARCHAR(64) PRIMARY KEY,
    emitter_id  VARCHAR(64) NOT NULL REFERENCES elint_emitters(emitter_id),
    band        VARCHAR(32),
    modulation  VARCHAR(32),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
