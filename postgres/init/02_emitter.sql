CREATE TABLE IF NOT EXISTS elint_emitters (
    emitter_id VARCHAR(64) PRIMARY KEY,
    cluster_id VARCHAR(64) NOT NULL REFERENCES elint_clusters(cluster_id),
    location   VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
