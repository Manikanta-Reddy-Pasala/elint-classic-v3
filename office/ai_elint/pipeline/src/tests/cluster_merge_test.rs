/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::cluster_merge::*;

    fn norm(v: &mut Vec<f32>) {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-12 { for x in v.iter_mut() { *x /= n; } }
    }

    #[test]
    fn empty_medoids_returns_empty() {
        assert!(merge_clusters(&[], &[], &[], &[], &[], &[], 0.7).is_empty());
    }

    #[test]
    fn single_cluster_no_merge() {
        let mut m = vec![1.0f32, 0.0, 0.0];
        norm(&mut m);
        let r = merge_clusters(&[m.clone()], &[10], &[45.0], &[1.0], &[2.75e9], &[vec![1, 2, 3]], 0.7);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0.size, 10);
        assert_eq!(r[0].0.member_ids, vec![1, 2, 3]);
    }

    #[test]
    fn two_similar_clusters_merge() {
        let mut m1 = vec![1.0f32, 0.0, 0.0];
        let mut m2 = vec![0.999f32, 0.001, 0.0];
        norm(&mut m1); norm(&mut m2);
        let r = merge_clusters(&[m1, m2], &[10, 5], &[45.0, 50.0], &[1.0, 1.0], &[2.75e9, 2.85e9], &[vec![1, 2], vec![3, 4]], 0.7);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0.size, 15);
        assert_eq!(r[0].0.member_ids.len(), 4);
    }

    #[test]
    fn two_dissimilar_clusters_stay_separate() {
        let m1 = vec![1.0f32, 0.0, 0.0];
        let m2 = vec![0.0f32, 1.0, 0.0];
        let r = merge_clusters(&[m1, m2], &[10, 5], &[45.0, 180.0], &[1.0, 2.0], &[2.75e9, 2.85e9], &[vec![1], vec![2]], 0.7);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn three_clusters_merge_pair() {
        let mut m1 = vec![1.0f32, 0.0, 0.0];
        let mut m2 = vec![0.99f32, 0.01, 0.0];
        let m3 = vec![0.0f32, 0.0, 1.0];
        norm(&mut m1); norm(&mut m2);
        let r = merge_clusters(&[m1, m2, m3], &[10, 10, 5], &[45.0, 50.0, 180.0], &[1.0, 1.0, 2.0], &[2.75e9, 2.80e9, 2.90e9], &[vec![1], vec![2], vec![3]], 0.7);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn member_ids_flattened() {
        let m = vec![1.0f32, 0.0, 0.0];
        let r = merge_clusters(&[m], &[3], &[0.0], &[1.0], &[2.75e9], &[vec![10, 20, 30]], 0.7);
        assert_eq!(r[0].0.member_ids, vec![10, 20, 30]);
    }

    #[test]
    fn angle_diff_deg_circular() {
        assert!((angle_diff_deg(0.0, 10.0) - 10.0).abs() < 1e-6);
        assert!((angle_diff_deg(350.0, 10.0) - 20.0).abs() < 1e-6);
        assert!((angle_diff_deg(90.0, 270.0) - 180.0).abs() < 1e-6);
    }

}
