/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::hdbscan::*;

    fn normalize(mut v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() { *x /= n; }
        v
    }

    #[test]
    fn single_point_is_noise() {
        let r = hdbscan_cosine(&[vec![1.0f32, 0.0, 0.0]], 5, 0);
        assert_eq!(r.labels.len(), 1);
        assert_eq!(r.labels[0], -1);
    }

    #[test]
    fn empty_input() {
        let r = hdbscan_cosine(&[], 5, 0);
        assert!(r.labels.is_empty());
    }

    #[test]
    fn deterministic_on_same_data() {
        let mut rng = 0x12345678u64;
        fn xs(s: &mut u64) -> f32 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; (*s & 0xFFFF) as f32 / 65535.0 }
        let dim = 32;
        let pts: Vec<Vec<f32>> = (0..30).map(|_| {
            let mut v = vec![0f32; dim];
            for x in v.iter_mut() { *x = xs(&mut rng); }
            v
        }).collect();
        let r1 = hdbscan_cosine(&pts, 5, 0);
        let r2 = hdbscan_cosine(&pts, 5, 0);
        assert_eq!(r1.labels, r2.labels);
    }

    #[test]
    fn label_count_matches_input() {
        let pts: Vec<Vec<f32>> = (0..10).map(|i| {
            let mut v = vec![0f32; 8];
            v[0] = 1.0; v[1] = i as f32 * 0.01;
            normalize(v)
        }).collect();
        let r = hdbscan_cosine(&pts, 3, 0);
        assert_eq!(r.labels.len(), 10);
        for l in &r.labels { assert!(*l >= -1); }
    }

    #[test]
    fn all_identical_points() {
        let pts: Vec<Vec<f32>> = vec![vec![1.0f32, 0.0, 0.0]; 20];
        let r = hdbscan_cosine(&pts, 3, 0);
        assert_eq!(r.labels.len(), 20);
        let unique: std::collections::HashSet<_> = r.labels.iter().collect();
        assert!(unique.len() <= 2);
    }

}
