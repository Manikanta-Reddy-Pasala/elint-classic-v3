/// Tests for config (moved out of inline #[cfg(test)]).
mod tests {
    use cdc_poller::config::*;

/// Set CH_CDC, read the config, restore the env — under the lock.
    fn read_mode_with_cdc(value: &str) -> ReadMode {
        let _g = ENV_LOCK.guard();
        if value.is_empty() {
            std::env::remove_var("CH_CDC");
        } else {
            std::env::set_var("CH_CDC", value);
        }
        let mode = Config::from_env().read_mode;
        std::env::remove_var("CH_CDC");
        mode
    }

    #[test]
    fn default_values() {
        let c = Config::default();
        assert_eq!(c.brokers, "127.0.0.1:9092");
        assert_eq!(c.pulse_count, 50_000);
        assert_eq!(c.read_mode, ReadMode::Cdc);
        assert_eq!(c.ch_url, "http://127.0.0.1:8123");
        assert_eq!(c.fetch_limit, 2_000);
        assert_eq!(c.num_sim_tasks, 4);
    }

    #[test]
    fn read_mode_from_env_cdc() {
        assert_eq!(read_mode_with_cdc("1"), ReadMode::Cdc);
    }

    #[test]
    fn read_mode_from_env_generate() {
        assert_eq!(read_mode_with_cdc("0"), ReadMode::Generate);
    }

    #[test]
    fn read_mode_from_env_live() {
        assert_eq!(read_mode_with_cdc("live"), ReadMode::Live);
    }

    #[test]
    fn read_mode_defaults_to_cdc() {
        assert_eq!(read_mode_with_cdc(""), ReadMode::Cdc);
    }

    #[test]
    fn topic_constants() {
        assert_eq!(TOPIC_PDW, "elint.pdw");
        assert_eq!(TOPIC_DONE, "elint.pdw.done");
    }
}
