CREATE TABLE IF NOT EXISTS elint_clusters (
    cluster_id VARCHAR(64) PRIMARY KEY,
    name       VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
