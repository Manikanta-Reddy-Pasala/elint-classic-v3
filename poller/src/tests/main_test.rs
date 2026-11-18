/// Tests for main (moved out of inline #[cfg(test)]).
mod tests {
    use cdc_poller::simulator::SimCounters;
    use std::sync::Arc;

#[test]
    fn sim_counters_start_at_zero() {
        let c = SimCounters::new();
        assert_eq!(c.summary(), (0, 0));
    }

    #[test]
    fn sim_counters_accumulate_inserted_and_published() {
        let c = SimCounters::new();
        c.add_inserted(100);
        c.add_inserted(50);
        c.add_published(100);
        assert_eq!(c.summary(), (150, 100));
    }

    #[test]
    fn sim_counters_are_shared_via_arc() {
        let c = Arc::new(SimCounters::new());
        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = c.clone();
            handles.push(std::thread::spawn(move || {
                c.add_inserted(25);
                c.add_published(25);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(c.summary(), (100, 100));
    }
}
