/// Tests for publish (moved out of inline #[cfg(test)]).
mod tests {
    use cdc_poller::config::Config;
    use cdc_poller::pulse::Pulse;
    use rdkafka::config::ClientConfig;
    use cdc_poller::publish::*;

use cdc_poller::pulse::Modulation;

    fn test_producer() -> TProducer {
        ClientConfig::new()
            .set("bootstrap.servers", "127.0.0.1:19092")
            .set("message.timeout.ms", "100")
            .create()
            .expect("test producer")
    }

    fn test_pulse(id: u64) -> Pulse {
        Pulse {
            id,
            power_dbm: -40.0,
            center_frequency_hz: 2.0e9,
            time_of_arrival_ns: 1_000,
            duration_ns: 100.0,
            bandwidth_hz: 1.0e6,
            direction_of_arrival_deg: 90.0,
            snr_db: 25.0,
            modulation: Modulation::Psk,
            is_cw: 0,
            receiver_id: 1,
        }
    }

    #[test]
    fn publish_pulses_serializes_each_row() {
        let producer = test_producer();
        let rows = vec![test_pulse(1), test_pulse(2), test_pulse(3)];
        // No broker on the port — sends time out and are logged as
        // errors, but every row must still be attempted and counted.
        let n = publish_pulses(&producer, &rows).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn publish_pulses_empty_returns_zero() {
        let producer = test_producer();
        assert_eq!(publish_pulses(&producer, &[]).unwrap(), 0);
    }

    #[test]
    fn send_done_does_not_panic_without_broker() {
        let producer = test_producer();
        send_done(&producer);
    }
}
