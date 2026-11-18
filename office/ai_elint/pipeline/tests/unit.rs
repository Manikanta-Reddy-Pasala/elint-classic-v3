// Integration test harness for pipeline unit tests.
// Each module pulls in a test file from src/tests/ via #[path].

#[path = "../src/tests/cache_test.rs"]
mod cache;
#[path = "../src/tests/ch_read_test.rs"]
mod ch_read;
#[path = "../src/tests/cluster_merge_test.rs"]
mod cluster_merge;
#[path = "../src/tests/config_test.rs"]
mod config;
#[path = "../src/tests/db_test.rs"]
mod db;
#[path = "../src/tests/engine_test.rs"]
mod engine;
#[path = "../src/tests/features_test.rs"]
mod features;
#[path = "../src/tests/global_match_test.rs"]
mod global_match;
#[path = "../src/tests/hdbscan_test.rs"]
mod hdbscan;
#[path = "../src/tests/inference_test.rs"]
mod inference;
#[path = "../src/tests/known_match_test.rs"]
mod known_match;
#[path = "../src/tests/refresh_logger_test.rs"]
mod refresh_logger;
#[path = "../src/tests/refresh_test.rs"]
mod refresh;
