/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::known_match::*;

    use ai_processor::cache::{Cache, EmitterTrack};
    use std::sync::Arc;

    fn track(id: i64, vec: Vec<f32>, angle: f64) -> EmitterTrack {
        EmitterTrack {
            signal_id: id,
            vector: vec,
            freq: 100.0,
            angle,
            power: 5.0,
            modulation: "AM".into(),
            cardinality: 1,
        }
    }

    #[test]
    fn empty_embeddings_returns_empty() {
        let r = match_known(&[], &[], &[track(1, vec![1.0, 0.0], 0.0)], 0.8, 40.0);
        assert!(r.is_empty());
    }

    #[test]
    fn no_tracks_all_unmatched() {
        let emb = vec![vec![1.0, 0.0]];
        let r = match_known(&emb, &[0.0], &[], 0.8, 40.0);
        assert_eq!(r.len(), 1);
        assert!(!r[0].matched);
        assert_eq!(r[0].signal_id, -1);
    }

    #[test]
    fn perfect_match_passes() {
        let emb = vec![vec![1.0, 0.0]];
        let t = track(7, vec![1.0, 0.0], 0.0);
        let r = match_known(&emb, &[0.0], &[t], 0.8, 40.0);
        assert!(r[0].matched);
        assert_eq!(r[0].signal_id, 7);
        assert!((r[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn low_score_fails_threshold() {
        let emb = vec![vec![0.0, 1.0]];
        let t = track(7, vec![1.0, 0.0], 0.0);
        let r = match_known(&emb, &[0.0], &[t], 0.8, 40.0);
        assert!(!r[0].matched);
    }

    #[test]
    fn angle_gate_blocks_match() {
        let emb = vec![vec![1.0, 0.0]];
        let t = track(7, vec![1.0, 0.0], 90.0);
        let r = match_known(&emb, &[0.0], &[t], 0.1, 40.0);
        assert!(!r[0].matched);
    }

    #[test]
    fn angle_within_gate_passes() {
        let emb = vec![vec![1.0, 0.0]];
        let t = track(7, vec![1.0, 0.0], 30.0);
        let r = match_known(&emb, &[0.0], &[t], 0.1, 40.0);
        assert!(r[0].matched);
    }

    #[test]
    fn angle_diff_wraps_at_360() {
        assert!((angle_diff_deg(350.0, 10.0) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn best_of_multiple_tracks() {
        let emb = vec![vec![1.0, 0.0]];
        let t1 = track(1, vec![0.0, 1.0], 0.0);
        let t2 = track(2, vec![1.0, 0.0], 0.0);
        let r = match_known(&emb, &[0.0], &[t1, t2], 0.5, 40.0);
        assert_eq!(r[0].signal_id, 2);
        assert!(r[0].matched);
    }

    #[test]
    fn apply_ema_updates_cache() {
        let cache = Arc::new(Cache::new());
        cache.save_track(&track(1, vec![1.0, 0.0], 0.0));
        apply_ema(&cache, 1, &[0.0, 1.0], 1.0);
        let loaded = cache.load_track(1).unwrap();
        assert!((loaded.vector[1] - 1.0).abs() < 1e-6);
    }

}
