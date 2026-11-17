/// Agglomerative merge + POC score gate.
///
/// Ports `cluster_engine.py::group_similar_clusters` and
/// `cluster_refinement.py::get_cluster_score`.
///
/// The pairwise similarity matrix is built in parallel (rayon) — it is the
/// dominant cost when `n` clusters and 1024-dim medoids are involved.
use rayon::prelude::*;
use serde::Serialize;

/// One merged cluster after agglomerative merge.
#[derive(Debug, Clone, Serialize)]
pub struct MergedCluster {
    pub cluster_id: i64,
    pub medoid: Vec<f32>,
    pub size: usize,
    pub angle: f64,
    pub modulation: f64,
    pub score: f64,
    pub member_ids: Vec<i64>,
    pub mean_freq_hz: f64,
}

/// Agglomeratively merge cluster medoids that are within `threshold`
/// cosine distance of each other (POC CLUSTER_MERGE_THRESHOLD = 0.7).
/// Uses connected-components on the pairwise similarity matrix.
///
/// Returns `Vec<(MergedCluster, Vec<usize>)>` where the second element
/// is the list of input cluster indices that make up the merged cluster
/// (callers use it to track parallel per-cluster payloads, e.g. the
/// CH `activity_ids` for each member cluster).
pub fn merge_clusters(
    medoids: &[Vec<f32>],
    sizes: &[usize],
    angles: &[f64],
    mods: &[f64],
    freqs: &[f64],
    member_ids_per_cluster: &[Vec<i64>],
    threshold: f64,
) -> Vec<(MergedCluster, Vec<usize>)> {
    let sim_norm: Vec<Vec<f32>> = normalize_all(medoids);
    merge_clusters_normed(&sim_norm, sizes, angles, mods, freqs, member_ids_per_cluster, threshold)
}

/// Like `merge_clusters`, but the medoids must ALREADY be L2-normalised.
/// Skips the per-medoid normalisation pass.
pub fn merge_clusters_normed(
    medoids: &[Vec<f32>],
    sizes: &[usize],
    angles: &[f64],
    mods: &[f64],
    freqs: &[f64],
    member_ids_per_cluster: &[Vec<i64>],
    threshold: f64,
) -> Vec<(MergedCluster, Vec<usize>)> {
    let n = medoids.len();
    if n == 0 {
        return Vec::new();
    }

    // Build pairwise cosine-similarity matrix (parallel over rows; each (i, j)
    // cell is written by exactly one task, so per-cell access is disjoint).
    // Embedding dimension read from the first medoid (1024 for v16 model).
    //
    // Fast path: the medoids are L2-normalised (built as normalised
    // centroids in the callers), so the dot product IS the cosine
    // similarity — no per-pair normalisation needed. The pre-merge
    // dedup already collapses near-identical medoids, so the matrix
    // here is tiny in practice.
    let dim = medoids[0].len();
    // SAFETY: `vec![0f32; ...]` is fully initialised, aligned, and lives for
    // the whole function; each (i, j) pair with i <= j is owned by task i, so
    // the two cells (i*n+j) and (j*n+i) are disjoint from every other task's.
    let vec = vec![0f32; n * n];
    (0..n).into_par_iter().for_each(|i| {
        // SAFETY: pointer is derived from the live Vec above; each write
        // targets cells no other task touches.
        let sim_ptr: *mut f32 = unsafe { vec.as_ptr().cast_mut() };
        for j in i..n {
            let mut dot: f32 = 0.0;
            for d in 0..dim {
                dot += medoids[i][d] * medoids[j][d];
            }
            unsafe {
                *sim_ptr.add(i * n + j) = dot;
                *sim_ptr.add(j * n + i) = dot;
            }
        }
    });
    let sim = vec;

    // Connected components: edge if sim >= threshold.
    let mut parent: Vec<i64> = (0..n as i64).collect();
    pub fn find(p: &mut [i64], x: i64) -> i64 {
        if p[x as usize] != x {
            p[x as usize] = find(p, p[x as usize]);
        }
        p[x as usize]
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if sim[i * n + j] as f64 >= threshold {
                let ri = find(&mut parent, i as i64);
                let rj = find(&mut parent, j as i64);
                if ri != rj {
                    parent[ri as usize] = rj;
                }
            }
        }
    }

    // Group by root.
    let mut groups: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i as i64);
        groups.entry(root).or_default().push(i);
    }

    let mut merged: Vec<(MergedCluster, Vec<usize>)> = Vec::new();
    let mut next_id = 0i64;
    for (_root, members) in groups.into_iter() {
        // Size-weighted centroid.
        let total_size: usize = members.iter().map(|&m| sizes[m]).sum();
        let mut centroid = vec![0f64; dim];
        for &m in &members {
            let w = (sizes[m] as f64) / (total_size as f64 + 1e-9);
            for d in 0..dim {
                centroid[d] += w * medoids[m][d] as f64;
            }
        }
        // L2-normalise.
        let norm: f64 = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in centroid.iter_mut() {
                *x /= norm;
            }
        }
        let medoid: Vec<f32> = centroid.iter().map(|&x| x as f32).collect();

        // Power-weighted major angle (POC get_cluster_angle_score).
        let mut angle_sum = 0f64;
        let mut weight_sum = 0f64;
        for &m in &members {
            let w = sizes[m] as f64;
            angle_sum += angles[m] * w;
            weight_sum += w;
        }
        let major_angle = if weight_sum > 0.0 { angle_sum / weight_sum } else { 0.0 };

        // Angle score: % of pulses within +/- 40 deg of major angle.
        let mut within = 0usize;
        for &m in &members {
            let diff = angle_diff_deg(major_angle, angles[m]);
            if diff <= 40.0 {
                within += sizes[m];
            }
        }
        let angle_score = if total_size > 0 { within as f64 / total_size as f64 } else { 0.0 };

        // Modulation score: 1 - (Shannon entropy / log2(unique_mods)).
        let mut mod_counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for &m in &members {
            let key = (mods[m] * 1000.0) as i64;
            *mod_counts.entry(key).or_insert(0) += sizes[m];
        }
        let n_mods = mod_counts.len().max(1);
        let mut entropy = 0f64;
        for &count in mod_counts.values() {
            let p = count as f64 / total_size.max(1) as f64;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        let max_entropy = (n_mods as f64).log2().max(1e-9);
        let mod_score = (1.0 - entropy / max_entropy).clamp(0.0, 1.0);

        // Medoid similarity: mean dot(medoid, each member medoid).
        let medoid_sim: f64 = members
            .iter()
            .map(|&m| {
                let mut dot = 0f32;
                for d in 0..dim {
                    dot += medoid[d] * medoids[m][d];
                }
                dot as f64
            })
            .sum::<f64>()
            / members.len() as f64;

        // POC score = (medoid_sim + angle_score + mod_score) / 3
        let score = (medoid_sim + angle_score + mod_score) / 3.0;

        // Flatten member IDs.
        let mut all_ids = Vec::new();
        for &m in &members {
            all_ids.extend_from_slice(&member_ids_per_cluster[m]);
        }

        // Size-weighted mean frequency.
        let mut freq_sum = 0f64;
        for &m in &members {
            freq_sum += freqs[m] * (sizes[m] as f64);
        }
        let mean_freq = if total_size > 0 { freq_sum / total_size as f64 } else { 0.0 };

        merged.push((
            MergedCluster {
                cluster_id: next_id,
                medoid,
                size: total_size,
                angle: major_angle,
                modulation: mods[members[0]],
                score,
                member_ids: all_ids,
                mean_freq_hz: mean_freq,
            },
            members,
        ));
        next_id += 1;
    }

    merged
}

/// L2-normalise every medoid in place (parallel over rows).
fn normalize_all(medoids: &[Vec<f32>]) -> Vec<Vec<f32>> {
    medoids
        .par_iter()
        .map(|m| {
            let mut v = m.clone();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 1e-12 {
                for x in v.iter_mut() {
                    *x /= n;
                }
            }
            v
        })
        .collect()
}

/// Circular angle difference in degrees (0..180).
pub fn angle_diff_deg(a: f64, b: f64) -> f64 {
    let mut d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        d = 360.0 - d;
    }
    d
}

/// Cosine similarity between two equal-length vectors (0..1).
/// Used for emitter dedup in global_match.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
