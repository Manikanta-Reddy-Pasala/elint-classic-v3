/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::global_match::*;

    #[test]
    fn sample_no_cap_returns_all() {
        assert_eq!(sample_indices(10, 0, 0), (0..10).collect::<Vec<usize>>());
    }

    #[test]
    fn sample_cap_larger_than_n_keeps_all() {
        assert_eq!(sample_indices(5, 100, 0), (0..5).collect::<Vec<usize>>());
    }

    #[test]
    fn sample_stride_hits_max_s() {
        let idx = sample_indices(100, 10, 0);
        assert_eq!(idx.len(), 10);
        assert_eq!(idx[0], 0);
        assert_eq!(idx.last().unwrap(), &90);
        // strictly increasing
        assert!(idx.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn sample_global_cap_shrinks() {
        let idx = sample_indices(1000, 100, 10);
        assert!(idx.len() <= 10);
        assert!(!idx.is_empty());
    }

    #[test]
    fn sample_empty_input() {
        assert!(sample_indices(0, 10, 0).is_empty());
    }

}
