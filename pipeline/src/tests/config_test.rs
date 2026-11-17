/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::config::*;
    use std::env;

    /// from_env reads the whole process env, and `cargo test` runs the
    /// #[test] fns in parallel — so these env tests are serialised on a
    /// process-wide lock.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_defaults_when_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        for k in [
            "ONNX_PATH", "ONNX_SESSION_COUNT", "ONNX_THREADS", "KNOWN_THRESHOLD",
            "ANGLE_GATE", "EMA_ALPHA", "BATCH_SIZE", "MIN_CLUSTER_SIZE",
            "MAX_SAMPLES", "PG_URL", "CH_URL",
        ] {
            std::env::remove_var(k);
        }
        let c = Config::from_env();
        assert_eq!(c.onnx_path, "/models/transformer_model_known_v16.onnx");
        assert_eq!(c.onnx_session_count, 2);
        assert_eq!(c.onnx_threads, 6);
        assert!((c.known_threshold - 0.80).abs() < f32::EPSILON);
        assert!((c.angle_gate - 40.0).abs() < f32::EPSILON);
        assert!((c.ema_alpha - 0.1).abs() < f32::EPSILON);
        assert_eq!(c.batch_size, 2000);
        assert_eq!(c.min_cluster_size, 10);
        assert_eq!(c.max_samples, 0);
        assert_eq!(c.pg_url, "postgresql://elint:elint@127.0.0.1:5432/sigint");
        assert_eq!(c.ch_url, "http://127.0.0.1:8123");
    }

    #[test]
    fn from_env_reads_overrides() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("ONNX_SESSION_COUNT", "3");
        env::set_var("ONNX_THREADS", "8");
        env::set_var("KNOWN_THRESHOLD", "0.95");
        env::set_var("BATCH_SIZE", "42");
        env::set_var("CH_URL", "http://ch.lab:9999");
        let c = Config::from_env();
        assert_eq!(c.onnx_session_count, 3);
        assert_eq!(c.onnx_threads, 8);
        assert!((c.known_threshold - 0.95).abs() < f32::EPSILON);
        assert_eq!(c.batch_size, 42);
        assert_eq!(c.ch_url, "http://ch.lab:9999");
        for k in ["ONNX_SESSION_COUNT", "ONNX_THREADS", "KNOWN_THRESHOLD", "BATCH_SIZE", "CH_URL"] {
            std::env::remove_var(k);
        }
    }

    #[test]
    fn from_env_bad_numeric_falls_back_to_default() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("ONNX_SESSION_COUNT", "not-a-number");
        env::set_var("BATCH_SIZE", "12.7");
        let c = Config::from_env();
        assert_eq!(c.onnx_session_count, 2);
        assert_eq!(c.batch_size, 2000);
        std::env::remove_var("ONNX_SESSION_COUNT");
        std::env::remove_var("BATCH_SIZE");
    }

}
