/// Tests for pulse (moved out of inline #[cfg(test)]).
mod tests {
    use cdc_poller::pulse::*;

#[test]
    fn generator_yields_exact_count() {
        let mut gen = PulseGenerator::new(10);
        let ids: Vec<u64> = gen.by_ref().map(|p| p.id).collect();
        assert_eq!(ids, (1..=10).collect::<Vec<u64>>());
    }

    #[test]
    fn generator_respects_u64_total() {
        // 1M records: the scale target.
        let mut gen = PulseGenerator::new(1_000_000);
        let count = gen.by_ref().count();
        assert_eq!(count, 1_000_000);
    }

    #[test]
    fn toa_is_monotonic_and_bounded() {
        let gen = PulseGenerator::new(100);
        let mut last = 0u64;
        for p in gen {
            assert!(p.time_of_arrival_ns > last);
            assert!(p.time_of_arrival_ns < 1_000_000_000_000);
            last = p.time_of_arrival_ns;
        }
    }

    #[test]
    fn modulation_is_known_variant() {
        let gen = PulseGenerator::new(500);
        for p in gen {
            let s = format!("{}", p.modulation);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = PulseGenerator::new(10);
        let mut b = PulseGenerator::new(10);
        // base_ns may differ by microseconds between the two constructors,
        // but the derived frequency/power values are driven by the same
        // seed and must match.
        let fa = a.next().unwrap().center_frequency_hz;
        let fb = b.next().unwrap().center_frequency_hz;
        assert!((fa - fb).abs() < f64::EPSILON);
    }
}
