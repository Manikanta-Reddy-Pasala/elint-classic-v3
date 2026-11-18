/// Tests for simulator (moved out of inline #[cfg(test)]).
mod tests {
    use cdc_poller::simulator::*;

use cdc_poller::config::Config;
    use cdc_poller::pulse::{Pulse, Modulation};

    #[test]
    fn build_insert_sql_single() {
        let p = vec![Pulse {
            id: 1,
            power_dbm: -50.0,
            center_frequency_hz: 1.0e9,
            time_of_arrival_ns: 1000,
            duration_ns: 100.0,
            bandwidth_hz: 1.0e6,
            direction_of_arrival_deg: 90.0,
            snr_db: 20.0,
            modulation: Modulation::Psk,
            is_cw: 0,
            receiver_id: 1,
        }];
        let sql = build_insert_sql(&p);
        assert!(sql.starts_with("INSERT INTO signals.pdw"));
        assert!(sql.contains("(1,"));
        assert!(sql.contains("'PSK'"));
    }

    #[test]
    fn build_insert_sql_multiple() {
        let p = vec![
            Pulse {
                id: 1, power_dbm: -50.0, center_frequency_hz: 1.0e9,
                time_of_arrival_ns: 1000, duration_ns: 100.0,
                bandwidth_hz: 1.0e6, direction_of_arrival_deg: 90.0,
                snr_db: 20.0, modulation: Modulation::Psk, is_cw: 0, receiver_id: 1,
            },
            Pulse {
                id: 2, power_dbm: -30.0, center_frequency_hz: 2.0e9,
                time_of_arrival_ns: 2000, duration_ns: 200.0,
                bandwidth_hz: 2.0e6, direction_of_arrival_deg: 45.0,
                snr_db: 30.0, modulation: Modulation::Fsk, is_cw: 1, receiver_id: 2,
            },
        ];
        let sql = build_insert_sql(&p);
        // Two rows: the second one appears as " (2, ..." (space before the paren).
        let count = sql.matches("(2,").count();
        assert!(count >= 1, "expected second row, sql = {}", sql);
        assert!(sql.contains("'FSK'"));
    }

    #[test]
    fn host_port_with_scheme() {
        let cfg = Config {
            ch_url: "http://10.0.0.5:9999".into(),
            ..Default::default()
        };
        let (h, p) = host_port(&cfg);
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 9999);
    }

    #[test]
    fn host_port_default() {
        let cfg = Config {
            ch_url: "http://10.0.0.5".into(),
            ..Default::default()
        };
        let (h, p) = host_port(&cfg);
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 8123);
    }

    #[test]
    fn insert_empty_batch() {
        let cfg = Config::default();
        // Empty batch returns 0 without touching the network.
        assert_eq!(insert_batch_sync(&cfg, &[]).unwrap(), 0);
    }
}
