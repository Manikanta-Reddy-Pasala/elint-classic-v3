/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::cache::*;

    fn sample_track(id: i64, vec: &[f32]) -> EmitterTrack {
        EmitterTrack {
            signal_id: id,
            vector: vec.to_vec(),
            freq: 100.0,
            angle: 45.0,
            power: 5.0,
            modulation: "AM".into(),
            cardinality: 1,
        }
    }

    #[test]
    fn new_cache_is_empty() {
        let c = Cache::new();
        assert_eq!(c.tracks.len(), 0);
        assert!(c.load_track(1).is_none());
    }

    #[test]
    fn save_and_load_track() {
        let c = Cache::new();
        let t = sample_track(42, &[1.0, 0.0, 0.0]);
        c.save_track(&t);
        let loaded = c.load_track(42).unwrap();
        assert_eq!(loaded.signal_id, 42);
        assert_eq!(loaded.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(loaded.modulation, "AM");
    }

    #[test]
    fn load_missing_returns_none() {
        let c = Cache::new();
        assert!(c.load_track(999).is_none());
    }

    #[test]
    fn ema_update_changes_vector() {
        let c = Cache::new();
        let t = sample_track(1, &[1.0, 0.0]);
        c.save_track(&t);
        c.ema_update_track(1, &[0.0, 1.0], 1.0);
        let loaded = c.load_track(1).unwrap();
        assert!((loaded.vector[0] - 0.0).abs() < 1e-6);
        assert!((loaded.vector[1] - 1.0).abs() < 1e-6);
        assert_eq!(loaded.cardinality, 2);
    }

    #[test]
    fn ema_alpha_zero_keeps_old() {
        let c = Cache::new();
        let t = sample_track(1, &[1.0, 0.0]);
        c.save_track(&t);
        c.ema_update_track(1, &[0.0, 1.0], 0.0);
        let loaded = c.load_track(1).unwrap();
        assert!((loaded.vector[0] - 1.0).abs() < 1e-6);
        assert!((loaded.vector[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn load_all_tracks_returns_all() {
        let c = Cache::new();
        c.save_track(&sample_track(1, &[1.0, 0.0]));
        c.save_track(&sample_track(2, &[0.0, 1.0]));
        assert_eq!(c.load_all_tracks().len(), 2);
    }

}
